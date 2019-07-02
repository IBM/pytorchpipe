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
from ptp.components.mixins.word_mappings import WordMappings
from ptp.data_types.data_definition import DataDefinition


class WordDecoder(Component, WordMappings):
    """
    Class responsible for decoding of samples encoded in the form of vectors ("probability distributions").
    """
    def __init__(self, name, config):
        """
        Initializes the component.

        :param name: Component name (read from configuration file).
        :type name: str

        :param config: Dictionary of parameters (read from the configuration ``.yaml`` file).
        :type config: :py:class:`ptp.configuration.ConfigInterface`

        """
        # Call constructor(s) of parent class(es) - in the right order!
        Component.__init__(self, name, WordDecoder, config)
        WordMappings.__init__(self)

        # Construct reverse mapping for faster processing.
        self.ix_to_word = dict((v,k) for k,v in self.word_to_ix.items())

        # Set key mappings.
        self.key_inputs = self.stream_keys["inputs"]
        self.key_outputs = self.stream_keys["outputs"]


    def input_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of input data that are required by the component.

        :return: dictionary containing input data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {
            self.key_inputs: DataDefinition([-1, -1], [torch.Tensor], "Batch of words, each represented as a vector (probability distribution) [BATCH_SIZE x ITEM_SIZE] (agnostic to item size)"),
            }

    def output_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of output data produced the component.

        :return: dictionary containing output data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {
            self.key_outputs: DataDefinition([-1, 1], [list, str], "Batch of words, each represented as a single string [BATCH_SIZE] x [string]")
            }

    def __call__(self, data_streams):
        """
        Encodes "inputs" in the format of a single word.
        Stores result in "outputs" field of in data_streams.

        :param data_streams: :py:class:`ptp.utils.DataStreams` object containing (among others):

            - "inputs": expected input field containing tensor [BATCH_SIZE x ITEM_SIZE]

            - "outputs": added output field containing list of words [BATCH_SIZE] x [string] 
        """
        # Get inputs to be encoded.
        inputs = data_streams[self.key_inputs]
        outputs_list = []
        # Process samples 1 by 1.
        for sample in inputs.chunk(inputs.size(0), 0):
            # Process single token.
            max_index = sample.squeeze(0).argmax(dim=0).item() 
            output_sample = self.ix_to_word[max_index]
            outputs_list.append(output_sample)
        # Create the returned dict.
        data_streams.publish({self.key_outputs: outputs_list})

