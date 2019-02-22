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

from ptp.text.token_encoder import TokenEncoder
from ptp.core_types.data_definition import DataDefinition


class LabelEncoder(TokenEncoder):
    """
    Class responsible for encoding of samples consisting of labels (into indices, that can be latter used for loss calculation, PyTorch-style).
    """
    def __init__(self, name, params):
        # Call constructors of parent classes.
        TokenEncoder.__init__(self, name, params)

        # Export token size to global params.
        self.key_token_size = self.mapkey("label_token_size")
        self.app_state[self.key_token_size] = len(self.word_to_ix)

    def input_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of input data that are required by the component.

        :return: dictionary containing input data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {
            self.key_inputs: DataDefinition([-1, 1], [list, str], "Batch of labels (words), each represented as a single string [BATCH_SIZE] x [string]"),
            }

    def output_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of output data produced the component.

        :return: dictionary containing output data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {
            self.key_outputs: DataDefinition([-1, 1], [list, int], "Batch of labels, each represented as a single index [BATCH_SIZE] x [index]")
            }

    def __call__(self, data_dict):
        """
        Encodes "inputs" in the format of a single word.
        Stores result in "outputs" field of in data_dict.

        :param data_dict: :py:class:`ptp.utils.DataDict` object containing (among others):

            - "inputs": expected input field containing list of words [BATCH_SIZE] x x [string]

            - "outputs": added output field containing list of indices  [BATCH_SIZE] x [1] 
        """
        # Get inputs to be encoded.
        inputs = data_dict[self.key_inputs]
        outputs_list = []
        # Process samples 1 by 1.
        for sample in inputs:
            assert not isinstance(sample, (list,)), 'This encoder requires input sample to contain a single word'
            # Process single token.
            output_sample = self.word_to_ix[sample]
            outputs_list.append(output_sample)
        # Create the returned dict.
        data_dict.extend({self.key_outputs: outputs_list})
