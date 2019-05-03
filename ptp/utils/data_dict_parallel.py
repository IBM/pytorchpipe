#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) tkornuta, IBM Corporation 2019
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch

from torch.nn.parallel._functions import Scatter, Gather
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.parallel_apply import parallel_apply


from ptp.data_types.data_dict import DataDict


def datadict_scatter(inputs, target_gpus, dim=0):
    r"""
    Slices tensors into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not tensors.
    """
    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            return Scatter.apply(target_gpus, None, dim, obj)
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        if isinstance(obj, DataDict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))    
        return [obj for targets in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None


def datadict_scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
    r"""Scatter with support for kwargs dictionary"""
    inputs = datadict_scatter(inputs, target_gpus, dim) if inputs else []
    kwargs = datadict_scatter(kwargs, target_gpus, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


def datadict_gather(outputs, target_device, dim=0):
    r"""
    Gathers tensors from different GPUs on a specified device
      (-1 means the CPU).
    """
    def gather_map(outputs):
        out = outputs[0]
        if isinstance(out, torch.Tensor):
            return Gather.apply(target_device, dim, *outputs)
        if out is None:
            return None

        if isinstance(out, DataDict):
            if not all((len(out) == len(d) for d in outputs)):
                raise ValueError('All dicts must have the same number of keys')
            return type(out)(((k, gather_map([d[k] for d in outputs]))
                              for k in out))

        if isinstance(out, dict):
            if not all((len(out) == len(d) for d in outputs)):
                raise ValueError('All dicts must have the same number of keys')
            return type(out)(((k, gather_map([d[k] for d in outputs]))
                              for k in out))

        return type(out)(map(gather_map, zip(*outputs)))

    # Recursive function calls like this create reference cycles.
    # Setting the function to None clears the refcycle.
    try:
        return gather_map(outputs)
    finally:
        gather_map = None


class DataDictParallel(torch.nn.DataParallel):
    """
    Modified DataParallel wrapper enabling operation on DataDicts.
    
    .. warning:
        Compatible with PyTorch v1.0.1 !!

    """
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DataDictParallel, self).__init__(module, device_ids, output_device, dim)

    def forward(self, *inputs, **kwargs):
        """
        Performs "parallelized forward" pass by scattering batch into several batches, distributing models on different GPUs, performing parallel pass and gathering results into a single (returned) DataDict.

        ..warning:
            As the "external" operations are changing inputs to tuple of DataDicts, extension of main DataDict must be done "outside" of this method.
        """

        # Simple processing.
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        # One device - also easy.
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])

        # Preprocessing: get only the inputs important for to the wrapped model (optimization).
        inputs_tuple = []
        for i, item in enumerate(inputs):
            input_dict = DataDict({key: value for key,value in item.items() if key in self.module.input_data_definitions().keys()})
            inputs_tuple.append(input_dict)
        # Convert to tuple.
        inputs_tuple = tuple(inputs_tuple)

        # Scatter inputs into several tuples.
        inputs_tuple, kwargs = self.scatter(inputs_tuple, kwargs, self.device_ids)

        # Create replicas of the module on all devices.
        replicas = self.replicate(self.module, self.device_ids[:len(inputs_tuple)])

        # Pass scattered inputs throught those replicas.
        self.parallel_apply(replicas, inputs_tuple, kwargs)

        # Gather tuple. This cannot be done "in place"!
        gathered_tuple = self.gather(inputs_tuple, self.output_device)

        # Return 0-th tuple, i.e. a single DataDict on device 0.
        return gathered_tuple[0]


    def replicate(self, module, device_ids):
        return replicate(module, device_ids)

    def scatter(self, inputs, kwargs, device_ids):
        return datadict_scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])

    def gather(self, outputs, output_device):
        return datadict_gather(outputs, output_device, dim=self.dim)

    def add_statistics(self, stat_col):
        """
        Adds statistics for the wrapped model.

        :param stat_col: ``StatisticsCollector``.
        """
        self.module.add_statistics(stat_col)


    def collect_statistics(self, stat_col, data_dict):
        """
        Collects statistics for the wrapped model.

        :param stat_col: :py:class:`ptp.utils.StatisticsCollector`.

        :param data_dict: ``DataDict`` containing inputs, targets etc.
        :type data_dict: :py:class:`ptp.core_types.DataDict`
        """
        self.module.collect_statistics(stat_col, data_dict)


    def add_aggregators(self, stat_agg):
        """
        Aggregates statistics for the wrapped model.

        :param stat_agg: ``StatisticsAggregator``.
        """
        self.module.add_aggregators(stat_agg)


    def aggregate_statistics(self, stat_col, stat_agg):
        """
        Aggregates statistics for the wrapped model.

        :param stat_col: ``StatisticsCollector``

        :param stat_agg: ``StatisticsAggregator``
        """
        self.module.aggregate_statistics(stat_col, stat_agg)
