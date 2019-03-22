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


class BOWEncoder(Component):
    """
    Simple Bag-of-word type encoder that encodes the sentence (in the form of a list of encoded words) into a vector.
    
    .. warning::
        BoW transformation is inreversible, thus decode-related methods in fact return original inputs.
    """
    def  __init__(self, name, params):
        """
        Initializes the bag-of-word encoded by creating dictionary mapping ALL words from training, validation and test sets into unique indices.

        :param name: Component name (read from configuration file).
        :type name: str

        :param params: Dictionary of parameters (read from configuration ``.yaml`` file).
        :type params: :py:class:`ptp.utils.ParamInterface`

        """
        # Call constructors of parent classes.
        Component.__init__(self, name, BOWEncoder, params)

        # Default name mappings for all encoders.
        self.key_inputs = self.get_stream_key("inputs")
        self.key_outputs = self.get_stream_key("outputs")
        self.key_bow_size = self.get_global_key("bow_size")

        # Retrieve output size from global params.
        self.bow_size = self.app_state[self.key_bow_size]

    def input_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of input data that are required by the component.

        :return: dictionary containing input data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {
            self.key_inputs: DataDefinition([-1, -1, self.bow_size], [list, list, torch.Tensor], "Batch of sentences, each represented as a list of vectors [BATCH_SIZE] x [SEQ_LENGTH] x [ITEM_SIZE] (agnostic to item size)")
            }

    def output_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of output data produced the component.

        :return: dictionary containing output data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {
            self.key_outputs: DataDefinition([-1, self.bow_size], [torch.Tensor], "Batch of sentences, each represented as a single vector [BATCH_SIZE x ITEM_SIZE] (agnostic to item size)")
            }

    def __call__(self, data_dict):
        """
        Encodes batch, or, in fact, only one field of batch ("inputs").
        Stores result in "outputs" field of data_dict.

        :param data_dict: :py:class:`ptp.utils.DataDict` object containing (among others):

            - "inputs": expected input containing list of (list of tokens) [BATCH SIZE] x [SEQ_LEN] x [ITEM_SIZE]
            - "outputs": added output tensor with encoded words [BATCH_SIZE x ITEM_SIZE]
        """
        # Get inputs to be encoded.
        inputs = data_dict[self.key_inputs]
        outputs_list = []
        # Process samples 1 by one.
        for sample in inputs:
            # Encode sample
            output = self.encode_sample(sample)
            # Add to list plus unsqueeze inputs dimension(!)
            outputs_list.append( output.unsqueeze(0) )
        # Concatenate output tensors.
        outputs = torch.cat(outputs_list, dim=0)
        # Add result to the data dict.
        data_dict.extend({self.key_outputs: outputs})

    def encode_sample(self, list_of_tokens):
        """
        Generates a bag-of-word vector of length `bow_size`.

        :param list_of_tokens: List of tokens [SEQ_LENGTH] x [ITEM_SIZE]
        :return: torch.LongTensor [ITEM_SIZE]
        """
        # Create output.
        output = list_of_tokens[0]
        # "Adds" tokens.
        for token in list_of_tokens[1:]:
            output += token
        return output
