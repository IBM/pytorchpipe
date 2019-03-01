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
import math

from ptp.components.component import Component
from ptp.data_types.data_definition import DataDefinition


class ReshapeTensor(Component):
    """
    Class responsible for reshaping the input tensor.

    """

    def __init__(self, name, params):
        """
        Initializes object.

        :param name: Loss name.
        :type name: str

        :param params: Dictionary of parameters (read from the configuration ``.yaml`` file).
        :type params: :py:class:`ptp.utils.ParamInterface`

        """
        # Call constructors of parent classes.
        Component.__init__(self, name, params)

        # Set key mappings.
        self.key_inputs = self.mapkey("inputs")
        self.key_outputs = self.mapkey("outputs")
        self.key_output_size = self.mapkey("output_size")

        # Get input and output shapes from configuration.
        self.input_dims = [int(x) for x in self.params["input_dims"]]
        self.output_dims = [int(x) for x in self.params["output_dims"]]

        # Set global variable - all dimensions ASIDE OF BATCH.
        self.app_state[self.key_output_size] = self.output_dims[1:]

    def input_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of input data that are required by the component.

        :return: dictionary containing input data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {
            self.key_inputs: DataDefinition(self.input_dims, [torch.Tensor], "Batch of inputs [BATCH_SIZE x ...]")
            }

    def output_data_definitions(self):
        """ 
        Function returns a empty dictionary with definitions of output data produced the component.

        :return: Empty dictionary.
        """
        return {
            self.key_outputs: DataDefinition(self.output_dims, [torch.Tensor], "Batch of outputs [BATCH_SIZE x ... ]"),
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

        # 
        outputs = inputs.view(self.output_dims) 
        # Create the returned dict.
        data_dict.extend({self.key_outputs: outputs})

