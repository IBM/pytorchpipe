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
from ptp.components.utils.word_mappings import pad_trunc_list


class SentenceIndexer(Component, WordMappings):
    """
    Class responsible for encoding of sequences of words into list of indices.
    Those can be letter embedded, encoded with 1-hot encoding or else.

    Additianally, when 'reverse' mode is on, it works in the oposite direction, i.e. changing tensor witl indices into list of words.
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
        Component.__init__(self, name, SentenceIndexer, config)
        WordMappings.__init__(self)

        # Set key mappings.
        self.key_inputs = self.stream_keys["inputs"]
        self.key_outputs = self.stream_keys["outputs"]

        # Read mode from the configuration.
        self.mode_reverse = self.config['reverse']

        # Force padding to a fixed length
        self.fixed_padding = self.config['fixed_padding']

        # Wether to add <EOS> at the end of sequence
        self.enable_eos_token = self.config['eos_token']

        if self.mode_reverse:
            # We will need reverse (index:word) mapping.
            self.ix_to_word = dict((v,k) for k,v in self.word_to_ix.items())

        # Get inputs distributions/indices flag.
        self.use_input_distributions = self.config["use_input_distributions"]



    def input_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of input data that are required by the component.

        :return: dictionary containing input data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        if self.mode_reverse:
            if self.use_input_distributions:
                return {
                    self.key_inputs: DataDefinition([-1, -1, -1], [torch.Tensor], "Batch of sentences represented as a single tensor with batch of probability distributions [BATCH_SIZE x SEQ_LENGTH x ITEM_SIZE]"),
                    }
            else: 
                return {
                    self.key_inputs: DataDefinition([-1, -1], [torch.Tensor], "Batch of sentences represented as a single tensor of indices of particular words [BATCH_SIZE x SEQ_LENGTH]"),
                    }
        else: 
            return {
                self.key_inputs: DataDefinition([-1, -1, 1], [list, list, str], "Batch of sentences, each represented as a list of words [BATCH_SIZE] x [SEQ_LENGTH] x [string]"),
                }

    def output_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of output data produced the component.

        :return: dictionary containing output data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        if self.mode_reverse:
            return {
                self.key_outputs: DataDefinition([-1, -1, 1], [list, list, str], "Batch of sentences, each represented as a list of words [BATCH_SIZE] x [SEQ_LENGTH] x [string]"),
                }
        else: 
            return {
                self.key_outputs: DataDefinition([-1, -1], [torch.Tensor], "Batch of sentences represented as a single tensor of indices of particular words  [BATCH_SIZE x SEQ_LENGTH]"),
                }


    def __call__(self, data_streams):
        """
        Encodes inputs into outputs.
        Depending on the mode (set by 'reverse' config param) calls sentences_to_tensor() (when False) or tensor_to_sentences() (when set to True).

        :param data_streams: :py:class:`ptp.datatypes.DataStreams` object.
        """
        if self.mode_reverse:
            if self.use_input_distributions:
                # Produce list of words.
                self.tensor_distributions_to_sentences(data_streams)
            else:
                # Produce list of words.
                self.tensor_indices_to_sentences(data_streams)
        else:
            # Produce indices.
            self.sentences_to_tensor(data_streams)


    def sentences_to_tensor(self, data_streams):
        """
        Encodes "inputs" in the format of batch of list of words into a single tensor with corresponding indices.

        :param data_streams: :py:class:`ptp.datatypes.DataStreams` object containing (among others):

            - "inputs": expected input field containing list of lists of words [BATCH_SIZE] x [SEQ_SIZE] x [string]

            - "outputs": added output field containing tensor with indices [BATCH_SIZE x SEQ_SIZE] 
        """
        # Get inputs to be encoded.
        inputs = data_streams[self.key_inputs]

        # Get index of padding.
        pad_index = self.word_to_ix['<PAD>']
        eos_index = self.word_to_ix['<EOS>'] if self.enable_eos_token else None

        outputs_list = []
        # Process sentences 1 by 1.
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

            # Apply fixed padding to all sequences if requested
            # Otherwise let torch.nn.utils.rnn.pad_sequence handle it and choose a dynamic padding
            if self.fixed_padding > 0:
                pad_trunc_list(output_sample, self.fixed_padding, padding_value=pad_index, eos_value=eos_index)

            outputs_list.append(self.app_state.LongTensor(output_sample))

        # Transform the list of lists to tensor.
        # output = self.app_state.LongTensor(outputs_list)
        output = torch.nn.utils.rnn.pad_sequence(outputs_list, batch_first=True, padding_value=pad_index)
        # Create the returned dict.
        data_streams.publish({self.key_outputs: output})

    def tensor_indices_to_sentences(self, data_streams):
        """
        Encodes "inputs" in the format of tensor with indices into a batch of list of words.

        :param data_streams: :py:class:`ptp.datatypes.DataStreams` object containing (among others):

            - "inputs": added output field containing tensor with indices [BATCH_SIZE x SEQ_SIZE] 

            - "outputs": expected input field containing list of lists of words [BATCH_SIZE] x [SEQ_SIZE] x [string]

        """
        # Get inputs to be changed to words.
        inputs = data_streams[self.key_inputs].data.cpu().numpy().tolist()

        outputs_list = []
        # Process samples 1 by 1.
        for sample in inputs:
            # Process list.
            output_sample = []
            # "Decode" sample (list of indices).
            for token in sample:
                # Get word.
                output_word = self.ix_to_word[token]
                # Add index to outputs.
                output_sample.append( output_word )

            # Add sentence to batch.
            outputs_list.append(output_sample)

        # Create the returned dict.
        data_streams.publish({self.key_outputs: outputs_list})

    def tensor_distributions_to_sentences(self, data_streams):
        """
        Encodes "inputs" in the format of tensor with probability distributions into a batch of list of words.

        :param data_streams: :py:class:`ptp.datatypes.DataStreams` object containing (among others):

            - "inputs": added output field containing tensor with indices [BATCH_SIZE x SEQ_SIZE x ITEM_SIZE] 

            - "outputs": expected input field containing list of lists of words [BATCH_SIZE] x [SEQ_SIZE] x [string]

        """
        # Get inputs to be changed to words.
        inputs = data_streams[self.key_inputs].max(2)[1].data.cpu().numpy().tolist()

        outputs_list = []
        # Process samples 1 by 1.
        for sample in inputs:
            # Process list.
            output_sample = []
            # "Decode" sample (list of indices).
            for token in sample:

                # Get word.
                output_word = self.ix_to_word[token]
                # Add index to outputs.
                output_sample.append( output_word )

            # Add sentence to batch.
            outputs_list.append(output_sample)

        # Create the returned dict.
        data_streams.publish({self.key_outputs: outputs_list})
