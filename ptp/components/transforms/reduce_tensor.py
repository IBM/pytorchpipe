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

__author__ = "Tomasz Kornuta"

import torch

from ptp.components.component import Component
from ptp.data_types.data_definition import DataDefinition
from ptp.configuration.config_parsing import get_value_from_dictionary


class ReduceTensor(Component):
    """
    Class responsible for reducing tensor using indicated reduction method along a given dimension.

    """

    def __init__(self, name, config):
        """
        Initializes object.

        :param name: Name of the component loaded from the configuration file.
        :type name: str

        :param config: Dictionary of parameters (read from the configuration ``.yaml`` file).
        :type config: :py:class:`ptp.configuration.ConfigInterface`

        """
        # Call constructors of parent classes.
        Component.__init__(self, name, ReduceTensor, config)

        # Set key mappings.
        self.key_inputs = self.stream_keys["inputs"]
        self.key_outputs = self.stream_keys["outputs"]
        
        # Get number of input dimensions from configuration.
        self.num_inputs_dims = self.config["num_inputs_dims"]
        # Get size of a single input item (last dimension) from globals.
        self.input_size =  self.globals["input_size"]

        # Get reduction tparamsype from configuration.
        self.dim =  self.config["reduction_dim"]
        self.keepdim =  self.config["keepdim"]

        # Set reduction type.
        rt = get_value_from_dictionary(
            "reduction_type", self.config,
            'sum | mean | min | max | argmin | argmax'.split(" | ")
            )
        reduction_types = {}
        reduction_types["sum"] = torch.sum
        reduction_types["mean"] = torch.mean
        reduction_types["min"] = torch.min
        reduction_types["max"] = torch.max
        reduction_types["argmin"] = torch.argmin
        reduction_types["argmax"] = torch.argmax

        self.reduction = reduction_types[rt]


    def input_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of input data that are required by the component.

        :return: dictionary containing input data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        # Generate the description of input stream.
        dims_desc  = ["DIM {}".format(i) for i in range(self.num_inputs_dims-1)]
        desc = "Batch of outputs [" + " x ".join(dims_desc) + "]"
        return {
            self.key_inputs: DataDefinition(
                [-1]*(self.num_inputs_dims-1) + [self.input_size],
                [torch.Tensor],
                desc)
            }

    def output_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of output data produced the component.

        :return: Empty dictionary.
        """
        # Generate the dimensions and description of output stream.
        if self.keepdim:
            dims = [-1]*(self.num_inputs_dims-1) + [self.input_size]
            dims[self.dim] = 1
            dims_desc  = ["DIM {}".format(i) for i in range(self.num_inputs_dims)]
            dims_desc[self.dim] = "1"
            desc = "Batch of outputs [" + " x ".join(dims_desc) + "]"
        else:
            dims = [-1]*(self.num_inputs_dims-2) + [self.input_size]
            dims_desc  = ["DIM {}".format(i) for i in range(self.num_inputs_dims-1)]
            desc = "Batch of outputs [" + " x ".join(dims_desc) + "]"
        return {
            self.key_outputs: DataDefinition(
                dims,
                [torch.Tensor],
                desc)
            }


    def __call__(self, data_streams):
        """
        Encodes "inputs" in the format of a single tensor.
        Stores reshaped tensor in "outputs" field of in data_streams.

        :param data_streams: :py:class:`ptp.utils.DataStreams` object containing (among others):

            - "inputs": expected input field containing tensor [BATCH_SIZE x ...]

            - "outputs": added output field containing tensor [BATCH_SIZE x ...] 
        """
        # Get inputs to be encoded.
        inputs = data_streams[self.key_inputs]

        outputs =  self.reduction(inputs, self.dim, self.keepdim)

        # Create the returned dict.
        data_streams.publish({self.key_outputs: outputs})

