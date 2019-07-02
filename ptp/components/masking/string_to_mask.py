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


class StringToMask(Component):
    """
    Class responsible for producing masks for strings using the provided word mappings.
    """

    def __init__(self, name, config):
        """
        Initializes object. Loads key and word mappings.

        :param name: Loss name.
        :type name: str

        :param config: Dictionary of parameters (read from the configuration ``.yaml`` file).
        :type config: :py:class:`ptp.configuration.ConfigInterface`

        """
        # Call constructors of parent classes.
        Component.__init__(self, name, StringToMask, config)

        # Get key mappings.
        self.key_strings = self.stream_keys["strings"]
        self.key_masks = self.stream_keys["masks"]
        
        # Retrieve word mappings from globals.
        self.word_to_ix = self.globals["word_mappings"]

    def input_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of input data that are required by the component.

        :return: dictionary containing input data definitions (each of type :py:class:`ptp.data_types.DataDefinition`).
        """
        return {
            self.key_strings: DataDefinition([-1, 1], [list, str], "Batch of strings, each being treated as a single 'vocabulary entry' (word) [BATCH_SIZE] x [STRING]")
            }

    def output_data_definitions(self):
        """ 
        Function returns a empty dictionary with definitions of output data produced the component.

        :return: Empty dictionary.
        """
        return {
            self.key_masks: DataDefinition([-1], [torch.Tensor], "Batch of masks [BATCH_SIZE]")
            }


    def __call__(self, data_streams):
        """
        Encodes "inputs" in the format of a single tensor.
        Stores reshaped tensor in "outputs" field of in data_streams.

        :param data_streams: :py:class:`ptp.utils.DataStreams` object containing (among others):

            - "inputs": expected input field containing tensor [BATCH_SIZE x ...]

            - "outputs": added output field containing tensor [BATCH_SIZE x ...] 
        """
        # Get inputs strings.
        strings = data_streams[self.key_strings]

        masks = torch.zeros(len(strings), requires_grad=False).type(self.app_state.ByteTensor)

        # Process samples 1 by 1.
        for i,sample in enumerate(strings):
            assert not isinstance(sample, (list,)), "This masking component requires input 'string' to contain a single word"
            # Process single token.
            if sample in self.word_to_ix.keys():
                masks[i] = 1
        
        # Extend the dict by returned stream with masks.
        data_streams.publish({
            self.key_masks: masks
            })

