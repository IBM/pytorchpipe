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

from ptp.components.text.token_encoder import TokenEncoder
from ptp.data_types.data_definition import DataDefinition


class SentenceIndexer(TokenEncoder):
    """
    Class responsible for encoding of sequences of words into list of indices.
    Those can be letter embedded, encoded with 1-hot encoding or else.
    """
    def __init__(self, name, params):
        # Call constructors of parent classes.
        TokenEncoder.__init__(self, name, params)

        # Export vocabulary size to global params.
        self.key_vocab_size = self.mapkey("sentence_vocab_size")
        self.app_state[self.key_vocab_size] = len(self.word_to_ix)

        self.logger.info("Initializing sentence indexer with vocabulary size '{}' = {}".format(self.key_vocab_size, len(self.word_to_ix)))


    def input_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of input data that are required by the component.

        :return: dictionary containing input data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {
            self.key_inputs: DataDefinition([-1, -1, 1], [list, list, str], "Batch of sentences, each represented as a list of words [BATCH_SIZE] x [SEQ_LENGTH] x [string]"),
            }

    def output_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of output data produced the component.

        :return: dictionary containing output data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {
            self.key_outputs: DataDefinition([-1, -1], [torch.Tensor], "Batch of sentences represented as a single tensor of indices [BATCH_SIZE x SEQ_LENGTH]"),
            }

    def __call__(self, data_dict):
        """
        Encodes "inputs" in the format of list of tokens (for a single sample)
        Stores result in "encoded_inputs" field of in data_dict.

        :param data_dict: :py:class:`ptp.utils.DataDict` object containing (among others):

            - "inputs": expected input field containing list of words [BATCH_SIZE] x [SEQ_SIZE] x [string]

            - "encoded_targets": added output field containing list of indices [BATCH_SIZE x SEQ_SIZE] 
        """
        # Get inputs to be encoded.
        inputs = data_dict[self.key_inputs]
        outputs_list = []
        # Process samples 1 by one.
        for sample in inputs:
            assert isinstance(sample, (list,)), 'This encoder requires input sample to contain a list of words'
            # Process list.
            output_sample = []
            # Encode sample (list of words)
            for token in sample:
                # Get index.
                output_index = self.word_to_ix[token]
                # Add index to outputs.
                output_sample.append( output_index )

            outputs_list.append(output_sample)

        # Transform the list of lists to tensor.
        output = torch.LongTensor(outputs_list)
        # Create the returned dict.
        data_dict.extend({self.key_outputs: output})
