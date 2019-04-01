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

import ptp.utils.io_utils as io
from ptp.components.models.model import Model
from ptp.data_types.data_definition import DataDefinition


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
            self.word_to_ix = self.initialize_word_mappings(self.data_folder, self.source_files)
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
            if word not in self.word_to_ix:
                self.word_to_ix[word] = len(self.word_to_ix)

        # Export vocabulary and its length to globals.
        self.globals["vocabulary"] = self.word_to_ix
        self.globals["vocabulary_size"] = len(self.word_to_ix)

        # Create the embeddings layer.
        self.logger.info("Initializing embeddings layer with vocabulary size = {} and embeddings size = {}".format(len(self.word_to_ix), self.embeddings_size))
        self.embeddings = torch.nn.Embedding(len(self.word_to_ix), self.embeddings_size)

        # Load the embeddings first.
        if self.config["pretrained_embeddings"] != '':
            emb_vectors = self.load_pretrained_glove_embeddings(self.data_folder, self.config["pretrained_embeddings"], self.word_to_ix, self.embeddings_size)
            self.embeddings.weight = torch.nn.Parameter(emb_vectors)



    def initialize_word_mappings(self, data_folder, source_files):
        """
        Load list of files (containing raw text) and creates a vocabulary from all words (tokens).
        Indexing starts from 0.

        :return: Dictionary with mapping "word-to-index".
        """
        assert len(source_files) > 0, 'Cannot create dictionary: "source_files" is empty, please provide comma separated list of files to be processed'
        # Get absolute path.
        data_folder = os.path.expanduser(data_folder)

        # Dictionary word_to_ix maps each word in the vocab to a unique integer.
        word_to_ix = {}
        # Add special word (10 spaces), so the "real" enumeration will start from 1!
        word_to_ix['          '] = 0

        for filename in source_files.split(','):
            # filename + path.
            fn = data_folder+ '/' + filename
            if not os.path.exists(fn):
                self.logger.warning("Cannot load tokens files from {} because file does not exist".format(fn))
                continue
            # File exists, try to parse.
            content = open(fn).read()
            # Parse tokens.
            for word in content.split():
                # If new token.
                if word not in word_to_ix:
                    word_to_ix[word] = len(word_to_ix)
        return word_to_ix


    def load_pretrained_glove_embeddings(self, data_folder, embeddings_name, word_to_ix, embeddings_size):
        """
        Loads the pretrained embeddings from GloVe project.

        :return: Array with loaded (or random) vectors.
        """
        # https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
        # http://ronny.rest/blog/post_2017_08_04_glove/

        # Check th presence of the file.
        # Available options.
        # https://nlp.stanford.edu/projects/glove/
        pretrained_embeddings_urls = {}
        pretrained_embeddings_urls["glove.6B.50d.txt"] = ("http://nlp.stanford.edu/data/glove.6B.zip", "glove.6B.zip")
        pretrained_embeddings_urls["glove.6B.100d.txt"] = ("http://nlp.stanford.edu/data/glove.6B.zip", "glove.6B.zip")
        pretrained_embeddings_urls["glove.6B.200d.txt"] = ("http://nlp.stanford.edu/data/glove.6B.zip", "glove.6B.zip")
        pretrained_embeddings_urls["glove.6B.300d.txt"] = ("http://nlp.stanford.edu/data/glove.6B.zip", "glove.6B.zip")
        pretrained_embeddings_urls["glove.42B.300d.txt"] = ("http://nlp.stanford.edu/data/glove.42B.300d.zip", "glove.42B.300d.zip")
        pretrained_embeddings_urls["glove.840B.300d.txt"] = ("http://nlp.stanford.edu/data/glove.840B.300d.zip", "glove.840B.300d.zip")
        pretrained_embeddings_urls["glove.twitter.27B.txt"] = ("http://nlp.stanford.edu/data/glove.twitter.27B.zip", "glove.twitter.27B.zip")

        if (embeddings_name not in pretrained_embeddings_urls.keys()):
            self.logger.error("Cannot load the indicated pretrained embeddings (current '{}' must be one of {})".format(embeddings_name, pretrained_embeddings_urls.keys()))
            exit(1)

        # Check presence of the file.
        if not io.check_file_existence(data_folder, embeddings_name):
            # Download and extract wikitext zip.
            io.download_extract_zip_file(self.logger, data_folder, pretrained_embeddings_urls[embeddings_name][0], pretrained_embeddings_urls[embeddings_name][1])
        else: 
            self.logger.info("File '{}' containing pretrained embeddings found in '{}' folder".format(embeddings_name, data_folder))

        num_loaded_embs = 0
        # Set zeros for words "out of vocabulary"
        # embeddings = np.zeros((len(word_to_ix), embeddings_size))
        embeddings = np.random.normal(scale=0.6, size=(len(word_to_ix), embeddings_size))
        # Open the embeddings file.
        with open(os.path.join(data_folder, embeddings_name)) as f:
            # Parse file 
            for line in f.readlines():
                values = line.split()
                # Get word.
                word = values[0]
                # Get index.
                index = word_to_ix.get(word)
                if index:
                    vector = np.array(values[1:], dtype='float32')
                    assert (len(vector) == embeddings_size), "Embeddings size must be equal to the size of pretrained embeddings!"
                    # Ok, set vector.
                    embeddings[index] = vector
                    # Increment counter.
                    num_loaded_embs += 1
        
        self.logger.info("Loaded {} pretrained embeddings from {}".format(num_loaded_embs, embeddings_name))

        # Return matrix with embeddings.
        return torch.from_numpy(embeddings).float()



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

            indices_list.append(output_sample)

        # Transform the list of lists to tensor.
        indices = torch.LongTensor(indices_list)

        # Embedd.
        embeds = self.embeddings(indices)

        # Add embeddings to datadict.
        data_dict.extend({self.key_outputs: embeds})
