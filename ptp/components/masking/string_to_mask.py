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
    Additionally, it returns the associated string indices.
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
        self.key_string_indices = self.stream_keys["string_indices"]
        
        # Retrieve word mappings from globals.
        self.word_to_ix = self.globals["word_mappings"]

        # Get value from configuration.
        self.out_of_vocabulary_value = self.config["out_of_vocabulary_value"]

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
            self.key_masks: DataDefinition([-1], [torch.Tensor], "Batch of masks [BATCH_SIZE]"),
            self.key_string_indices: DataDefinition([-1], [torch.Tensor], "Batch of indices corresponging to inputs strings when using provided word mappings [BATCH_SIZE]")
            }


    def __call__(self, data_dict):
        """
        Encodes "inputs" in the format of a single tensor.
        Stores reshaped tensor in "outputs" field of in data_dict.

        :param data_dict: :py:class:`ptp.utils.DataDict` object containing (among others):

            - "inputs": expected input field containing tensor [BATCH_SIZE x ...]

            - "outputs": added output field containing tensor [BATCH_SIZE x ...] 
        """
        # Get inputs strings.
        strings = data_dict[self.key_strings]

        masks = torch.zeros(len(strings), requires_grad=False).type(self.app_state.ByteTensor)

        outputs_list = []
        # Process samples 1 by 1.
        for i,sample in enumerate(strings):
            assert not isinstance(sample, (list,)), 'This encoder requires input sample to contain a single word'
            # Process single token.
            if sample in self.word_to_ix.keys():
                output_sample = self.word_to_ix[sample]
                masks[i] = 1
            else:
                # Word out of vocabulary.
                output_sample = self.out_of_vocabulary_value
            outputs_list.append(output_sample)
        # Transform to tensor.
        output_indices = torch.tensor(outputs_list, requires_grad=False).type(self.app_state.LongTensor)
        
        #print("strings ", strings)
        #print("masks ", masks)
        #print("indices ", output_indices)

        # Create the returned dict.
        data_dict.extend({
            self.key_masks: masks,
            self.key_string_indices: output_indices
            })

