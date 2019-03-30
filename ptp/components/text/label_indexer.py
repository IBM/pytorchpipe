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


class LabelIndexer(TokenEncoder):
    """
    Class responsible for changing of samples consisting of single words/labels into indices (that e.g. can be latter used for loss calculation, PyTorch-style).
    """
    def __init__(self, name, config):
        """
        Initializes the component.

        :param name: Component name (read from configuration file).
        :type name: str

        :param config: Dictionary of parameters (read from the configuration ``.yaml`` file).
        :type config: :py:class:`ptp.configuration.ConfigInterface`

        """
        # Call constructors of parent classes.
        TokenEncoder.__init__(self, name, LabelIndexer, config)

        # Export vocabulary size to global variables.
        self.globals["label_vocab_size"] = len(self.word_to_ix)

        self.logger.info("Initializing sentence indexer with vocabulary size '{}' = {}".format(self.global_keys["label_vocab_size"], len(self.word_to_ix)))

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
            self.key_outputs: DataDefinition([-1], [torch.Tensor], "Batch of labels, each represented as a single index [BATCH_SIZE]")
            }

    def __call__(self, data_dict):
        """
        Encodes "inputs" in the format of a single word.
        Stores result in "outputs" field of in data_dict.

        :param data_dict: :py:class:`ptp.utils.DataDict` object containing (among others):

            - "inputs": expected input field containing list of words [BATCH_SIZE] x x [string]

            - "outputs": added output field containing list of indices  [BATCH_SIZE] 
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
        # Transform to tensor.
        output_tensor = torch.tensor(outputs_list)
        # Create the returned dict.
        data_dict.extend({self.key_outputs: output_tensor})
