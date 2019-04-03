#!/usr/bin/env python3
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


import os
import torch
import numpy as np

from ptp.components.models.model import Model
from ptp.data_types.data_definition import DataDefinition

import ptp.components.utils.io as io
import ptp.components.utils.word_mappings as wm
import ptp.components.utils.embeddings as emb


class SentenceEmbeddings(Model):
    """
    Model responsible of embedding of whole sentences.

    For this purpose it:
        - creates its own vocabulary,
        - splits sentences into tokens (using NLTK tokenizer),
        - uses vocabulary to first transform tokens (one by one) to indices (first) and dense vectors (next).

    Optionally, it can load pretrained word embeddings (currently GloVe).

    """ 
    def __init__(self, name, config):
        """
        Initializes the ``SentenceEmbeddings`` layer.

        :param name: Name of the model (taken from the configuration file).

        :param config: Parameters read from configuration file.
        :type config: ``ptp.configuration.ConfigInterface``
        """
        super(SentenceEmbeddings, self).__init__(name, SentenceEmbeddings, config)

        # Set key mappings.
        self.key_inputs = self.stream_keys["inputs"]
        self.key_outputs = self.stream_keys["outputs"]

        # Read the actual configuration.
        self.data_folder = os.path.expanduser(self.config['data_folder'])

        # Source and resulting (indexed) vocabulary.
        self.source_vocabulary_files = self.config['source_vocabulary_files']
        self.vocabulary_mappings_file = self.config['vocabulary_mappings_file']
        # Regenerate vocabulary.
        self.mode_regenerate = self.config['regenerate']

        # Retrieve embeddings size from configuration and export it to globals.
        self.embeddings_size = self.config['embeddings_size']
        self.globals["embeddings_size"] = self.embeddings_size

        # Initialize the vocabulary.
        vocabulary_mappings_file_path = os.path.expanduser(self.data_folder) + "/" + self.vocabulary_mappings_file

        # Check whether we want to (re)generate new  or load existing encodings.
        if self.mode_regenerate or not os.path.exists(vocabulary_mappings_file_path):
            # Generate new vocabulary.
            self.word_to_ix = wm.initialize_word_mappings_from_source_files(self.logger, self.data_folder, self.source_files)
            assert (len(self.word_to_ix) > 0), "The created vocabulary cannot be empty!"
            # Ok, save mappings, so next time we will simply load them.
            io.save_mappings_to_csv_file(self.data_folder, self.vocabulary_mappings_file, self.word_to_ix, ['word', 'index'])
        else:
            # Load encodings.
            self.word_to_ix = io.load_mappings_from_csv_file(self.data_folder, self.vocabulary_mappings_file)
            assert (len(self.word_to_ix) > 0), "The loaded encodings list is empty!"

        # Check if additional tokens are present.
        self.additional_tokens = self.config["additional_tokens"].split(',')
        for word in self.additional_tokens:
            # If new token.
            if word != '' and word not in self.word_to_ix:
                self.word_to_ix[word] = len(self.word_to_ix)

        # Export vocabulary and its length to globals.
        self.globals["vocabulary"] = self.word_to_ix
        self.globals["vocabulary_size"] = len(self.word_to_ix)

        # Create the embeddings layer.
        self.logger.info("Initializing embeddings layer with vocabulary size = {} and embeddings size = {}".format(len(self.word_to_ix), self.embeddings_size))
        self.embeddings = torch.nn.Embedding(len(self.word_to_ix), self.embeddings_size, padding_idx=0) # Index of self.word_to_ix['<PAD>']

        # Load the embeddings first.
        if self.config["pretrained_embeddings"] != '':
            emb_vectors = self.load_pretrained_glove_embeddings(self.data_folder, self.config["pretrained_embeddings"], self.word_to_ix, self.embeddings_size)
            self.embeddings.weight = torch.nn.Parameter(emb_vectors)


    def input_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of input data that are required by the component.

        :return: dictionary containing input data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {
            self.key_inputs: DataDefinition([-1, -1, 1], [list, list, str], "Batch of sentences, each represented as a list of words [BATCH_SIZE] x [SEQ_LENGTH] x [string]")
            }


    def output_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of output data produced the component.

        :return: dictionary containing output data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {
            self.key_outputs: DataDefinition([-1, -1, self.embeddings_size], [torch.Tensor], "Batch of embedded sentences [BATCH_SIZE x SENTENCE_LENGTH x EMBEDDING_SIZE]")
            }


    def forward(self, data_dict):
        """
        Forward pass  - performs embedding.

        :param data_dict: DataDict({'images',**}), where:

            - inputs: expected tokenized sentences [BATCH_SIZE x SENTENCE_LENGTH] x [string]
            - outputs: added embedded sentences [BATCH_SIZE x SENTENCE_LENGTH x EMBEDDING_SIZE]

        :type data_dict: ``miprometheus.utils.DataDict``
        """

        # Unpack DataDict.
        inputs = data_dict[self.key_inputs]

        indices_list = []
        # Process samples 1 by one.
        for sample in inputs:
            assert isinstance(sample, (list,)), 'This embedder requires input sample to contain a list of words'
            # Process list.
            output_sample = []
            # Encode sample (list of words)
            for token in sample:
                # Get index.
                output_index = self.word_to_ix[token]
                # Add index to outputs.
                output_sample.append( output_index )

            indices_list.append(torch.tensor(output_sample).type(self.app_state.LongTensor))

        # Transform the list of lists to tensor - use padding.
        # indices = torch.LongTensor(indices_list)

        # Sort data by seq_length.
        indices_list.sort(key=lambda x: len(x), reverse=True)

        # Get lengths.
        #seq_lengths = [len(x) for x in indices_list]

        # Pad the indices list.
        padded_indices = torch.nn.utils.rnn.pad_sequence(indices_list, batch_first=True)

        # Embedd indices.
        embedds = self.embeddings(padded_indices)

        # Pack embedded sentences.
        #packed_embedds = torch.nn.utils.rnn.pack_padded_sequence(embeds, lengths= seq_lengths)#, batch_first=True)

        # Add embeddings to datadict.
        data_dict.extend({self.key_outputs: embedds})
