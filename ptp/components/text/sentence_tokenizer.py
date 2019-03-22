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

from ptp.components.component import Component
from ptp.data_types.data_definition import DataDefinition


class SentenceTokenizer(Component):
    """
    Class responsible for tokenizing the sentence.
    """
    def __init__(self, name, params):
        """
        Initializes the component.

        :param name: Component name (read from configuration file).
        :type name: str

        :param params: Dictionary of parameters (read from the configuration ``.yaml`` file).
        :type params: :py:class:`ptp.utils.ParamInterface`

        """
        # Call constructors of parent classes.
        Component.__init__(self, name, SentenceTokenizer, params)

        # Read the actual configuration.
        self.mode_detokenize = params['detokenize']

        # Set key mappings.
        self.key_inputs = self.get_stream_key("inputs")
        self.key_outputs = self.get_stream_key("outputs")

        if self.mode_detokenize:
            # list of strings -> sentence.
            self.processor = self.detokenize_sample
        else:
            # sentence -> list of strings.
            self.processor = self.tokenize_sample
        # Ok, we are ready to go!


    def input_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of input data that are required by the component.

        :return: dictionary containing input data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        if self.mode_detokenize == False:
            return { self.key_inputs: DataDefinition([-1, 1], [list, str], "Batch of sentences, each represented as a single string [BATCH_SIZE] x [string]") }
        else:
            return { self.key_inputs: DataDefinition([-1, -1, 1], [list, list, str], "Batch of tokenized sentences, each represented as a list of words [BATCH_SIZE] x [SEQ_LENGTH] x [string]") }


    def output_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of output data produced the component.

        :return: dictionary containing output data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        if self.mode_detokenize == False:
            return { self.key_outputs: DataDefinition([-1, -1, 1], [list, list, str], "Batch of tokenized sentences, each represented as a list of words [BATCH_SIZE] x [SEQ_LENGTH] x [string]") }
        else:
            return { self.key_outputs: DataDefinition([-1, 1], [list, str], "Batch of sentences, each represented as a single string [BATCH_SIZE] x [string]") }


    def tokenize_sample(self, sample):
        """
        Changes sample (sentence) into list of tokens (words).

        :param sample: sentence (string).

        :return: list of words (strings).
        """
        return sample.split()

    def detokenize_sample(self, sample):
        """
        Changes list of tokens (words) into sentence.

        :param sample: list of words (strings).

        :return: sentence (string).
        """
        return ' '.join([str(x) for x in sample])

    def __call__(self, data_dict):
        """
        Encodes batch, or, in fact, only one field of bach ("inputs").
        Stores result in "encoded_inputs" field of in data_dict.

        :param data_dict: :py:class:`ptp.utils.DataDict` object containing (among others):

            - "inputs": expected input field containing list of words

            - "encoded_targets": added field containing output, tensor with encoded samples [BATCH_SIZE x 1] 
        """
        # Get inputs to be encoded.
        inputs = data_dict[self.key_inputs]
        outputs_list = []
        # Process samples 1 by one.
        for sample in inputs:
            output = self.processor(sample)
            # Add to outputs.
            outputs_list.append( output )
        # Create the returned dict.
        data_dict.extend({self.key_outputs: outputs_list})
