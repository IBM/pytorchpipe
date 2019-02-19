#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) IBM Corporation 2019
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

__author__ = "Vincent Marois, Tomasz Kornuta"

import torch
import logging
import collections

logger = logging.Logger('DataDict')


class DataDict(collections.MutableMapping):
    """
    - Mapping: A container object that supports arbitrary key lookups and implements the methods ``__getitem__``, \
    ``__iter__`` and ``__len__``.

    - Mutable objects can change their value but keep their id() -> ease modifying existing keys' value.

    DataDict: Dict used for storing batches of data by problems.

    **This is the main object class used to share data between a problem class and a model class through a worker.**
    """

    def __init__(self, *args, **kwargs):
        """
        DataDict constructor. Can be initialized in different ways:

            >>> data_dict = DataDict()
            >>> data_dict = DataDict({'inputs': torch.tensor(), 'targets': numpy.ndarray()})
            >>> # etc.

        :param args: Used to pass a non-keyworded, variable-length argument list.

        :param kwargs: Used to pass a keyworded, variable-length argument list.
        """
        self.__dict__.update(*args, **kwargs)

    def __setitem__(self, key, value, addkey=False):
        """
        key:value setter function.

        :param key: Dict Key.

        :param value: Associated value.

        :param addkey: Indicate whether or not it is authorized to add a new key `on-the-fly`.\
        Default: ``False``.
        :type addkey: bool

        .. warning::

            `addkey` is set to ``False`` by default as setting it to ``True`` removes flexibility of the\
            ``DataDict``. Indeed, there are some cases where adding a key `on-the-fly` to a ``DataDict`` is\
            useful (e.g. for plotting pre-processing).


        """
        if addkey and key not in self.keys():
            logger.error('KeyError: Cannot modify a non-existing key.')
            raise KeyError('Cannot modify a non-existing key.')
        else:
            self.__dict__[key] = value

    def __getitem__(self, key):
        """
        Value getter function.

        :param key: Dict Key.

        :return: Associated Value.

        """
        return self.__dict__[key]

    def __delitem__(self, key, override=False):
        """
        Delete a key:value pair.

        .. warning::

            By default, it is not authorized to delete an existing key. Set `override` to ``True`` to ignore this\
            restriction.

        :param key: Dict Key.

        :param override: Indicate whether or not to lift the ban of non-deletion of any key.
        :type override: bool

        """
        if not override:
            logger.error('KeyError: Not authorizing the deletion of a key.')
            raise KeyError('Not authorizing the deletion of a key.')
        else:
            del self.__dict__[key]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __str__(self):
        """
        :return: A simple Dict representation of ``DataDict``.

        """
        return str(self.__dict__)

    def __repr__(self):
        """
        :return: Echoes class, id, & reproducible representation in the Read–Eval–Print Loop.

        """
        return '{}, DataDict({})'.format(super(DataDict, self).__repr__(), self.__dict__)

    def numpy(self):
        """
        Converts the DataDict to numpy objects.

        .. note::

            The ``torch.tensor`` (s) contained in `self` are converted using ``torch.Tensor.numpy()`` : \
            This tensor and the returned ndarray share the same underlying storage. \
            Changes to ``self`` tensor will be reflected in the ndarray and vice versa.

            If an element of ``self`` is not a ``torch.tensor``, it is returned as is.


        :return: Converted DataDict.

        """
        numpy_datadict = self.__class__({key: None for key in self.keys()})

        for key in self:
            if isinstance(self[key], torch.Tensor):
                numpy_datadict[key] = self[key].numpy()
            else:
                numpy_datadict[key] = self[key]

        return numpy_datadict

    def cpu(self):
        """
        Moves the DataDict to memory accessible to the CPU.

        .. note::

            The ``torch.tensor`` (s) contained in `self` are converted using ``torch.Tensor.cpu()`` .
            If an element of `self` is not a ``torch.tensor``, it is returned as is, \
            i.e. We only move the ``torch.tensor`` (s) contained in `self`.


        :return: Converted DataDict.

        """
        cpu_datadict = self.__class__({key: None for key in self.keys()})

        for key in self:
            if isinstance(self[key], torch.Tensor):
                cpu_datadict[key] = self[key].cpu()
            else:
                cpu_datadict[key] = self[key]

        return cpu_datadict

    def cuda(self, device=None, non_blocking=False):
        """
        Returns a copy of this object in CUDA memory.

        .. note::

            Wraps call to ``torch.Tensor.cuda()``: If this object is already in CUDA memory and on the correct device, \
            then no copy is performed and the original object is returned.
            If an element of `self` is not a ``torch.tensor``, it is returned as is, \
            i.e. We only move the ``torch.tensor`` (s) contained in `self`. \


        :param device: The destination GPU device. Defaults to the current CUDA device.
        :type device: torch.device

        :param non_blocking: If True and the source is in pinned memory, the copy will be asynchronous with respect to \
        the host. Otherwise, the argument has no effect. Default: ``False``.
        :type non_blocking: bool

        """
        cuda_datadict = self.__class__({key: None for key in self.keys()})
        for key in self:
            if isinstance(self[key], torch.Tensor):
                cuda_datadict[key] = self[key].cuda(device=device, non_blocking=non_blocking)
            else:
                cuda_datadict[key] = self[key]

        return cuda_datadict

    def detach(self):
        """
        Returns a new DataDict, detached from the current graph.
        The result will never require gradient.

        .. note::
            Wraps call to ``torch.Tensor.detach()`` : the ``torch.tensor`` (s) in the returned ``DataDict`` use the same\
            data tensor(s) as the original one(s).
            In-place modifications on either of them will be seen, and may trigger errors in correctness checks.

        """
        detached_datadict = self.__class__({key: None for key in self.keys()})
        for key in self:
            if isinstance(self[key], torch.Tensor):
                detached_datadict[key] = self[key].detach()
            else:
                detached_datadict[key] = self[key]

        return detached_datadict


if __name__ == '__main__':
    """Unit test for DataDict"""

    data_definitions = {'inputs': {'size': [-1, -1], 'type': [torch.Tensor]},
                        'targets': {'size': [-1], 'type': [torch.Tensor]}
                        }

    datadict = DataDict({key: None for key in data_definitions.keys()})

    #datadict['inputs'] = torch.ones([64, 20, 512]).type(torch.FloatTensor)
    #datadict['targets'] = torch.ones([64, 20]).type(torch.FloatTensor)

    print(repr(datadict))

