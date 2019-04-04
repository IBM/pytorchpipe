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


class ListToTensor(Component):
    """
    Class responsible for transforming list (of lists) to a tensor.

    """

    def __init__(self, name, config):
        """
        Initializes object.

        :param name: Loss name.
        :type name: str

        :param config: Dictionary of parameters (read from the configuration ``.yaml`` file).
        :type config: :py:class:`ptp.configuration.ConfigInterface`

        """
        # Call constructors of parent classes.
        Component.__init__(self, name, ListToTensor, config)

        # Set key mappings.
        self.key_inputs = self.stream_keys["inputs"]
        self.key_outputs = self.stream_keys["outputs"]
        
        # Get number of input dimensions from configuration.
        self.num_inputs_dims = self.config["num_inputs_dims"]

        # Get size of a single input item (last dimension) from globals.
        self.input_size =  self.globals["input_size"]



    def input_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of input data that are required by the component.

        :return: dictionary containing input data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {
            self.key_inputs: DataDefinition(
                [-1]*(self.num_inputs_dims-1) + [self.input_size],
                [list]*(self.num_inputs_dims-1) + [torch.Tensor],
                "Batch of inputs [DIM 1 x DIM 2 x ... x INPUT_SIZE]")
            }

    def output_data_definitions(self):
        """ 
        Function returns a empty dictionary with definitions of output data produced the component.

        :return: Empty dictionary.
        """
        return {
            self.key_outputs: DataDefinition(
                [-1]*(self.num_inputs_dims-1) + [self.input_size],
                [torch.Tensor],
                "Batch of outputs [DIM 1 x DIM 2 x ... x INPUT_SIZE]")
            }


    def __call__(self, data_dict):
        """
        Encodes "inputs" in the format of a single tensor.
        Stores reshaped tensor in "outputs" field of in data_dict.

        :param data_dict: :py:class:`ptp.utils.DataDict` object containing (among others):

            - "inputs": expected input field containing tensor [BATCH_SIZE x ...]

            - "outputs": added output field containing tensor [BATCH_SIZE x ...] 
        """
        # Get inputs to be encoded.
        inputs = data_dict[self.key_inputs]

        # Change to tensor.
        if self.num_inputs_dims == 1:
            # CASE: Single tensor.
            outputs = input
        elif self.num_inputs_dims == 2:
            # CASE: List of tensors.

            # This needs testing - padding?
            outputs = torch.stack(inputs)

        elif self.num_inputs_dims == 3:
            # CASE: List of lists of tensors.

            # Get type.
            ttype = type(inputs[0][0])
            # Generate tensor that will be added as padding - all zeros.
            pad_tensor = ttype([0]*self.input_size)

            # Get max length of lists.
            max_len = max([len(lst) for lst in inputs])

            # List of stacked tensors.
            stacked_tensor_lst = []

            # Iterate over list of lists.
            for lst in inputs:
                # "Manual" padding of each inner list.
                if len(lst) < max_len:
                    lst.extend([pad_tensor] * (max_len - len(lst)))
                # Stack inner list.
                stacked_tensor = torch.stack(lst)
                stacked_tensor_lst.append(stacked_tensor)
            # Finally, pad the result.
            outputs = torch.stack(stacked_tensor_lst)

        # Create the returned dict.
        data_dict.extend({self.key_outputs: outputs})

