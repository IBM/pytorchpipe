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
import numpy as np
import torch
import ptp.components.utils.io as io


def load_pretrained_embeddings(logger, folder, embeddings_name, word_to_ix, embeddings_size):
    """
    Creates embedding vector for words from the provided (word:index) mappings (dictionary).

    Loads the pretrained embeddings from the GloVe project - for the words found in the dictionary.
    
    For words out of dictionary initializes random vectors.

    Available embeddings:
        - glove.6B.50d.txt
        - glove.6B.100d.txt
        - glove.6B.200d.txt
        - glove.6B.300d.txt
        - glove.42B.300d.txt
        - glove.840B.300d.txt
        - glove.twitter.27B.txt
        - fasttext.mimic.300d.txt

    :param logger: Logger object.

    :param folder: Relative path to to the folder.
    :type folder: str

    :param word_to_ix: (word:index) mappings
    :type word_to_ix: dict

    :param embeddings_size: Embeddings size. Warning: must match the length of vector in the selected file.

    :return: Torch tensor with loaded (or random) vectors.
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
    pretrained_embeddings_urls["mimic.fastText.no_clean.300d.pickled"] = ("https://mednli.blob.core.windows.net/shared/word_embeddings/mimic.fastText.no_clean.300d.pickled","mimic.fastText.no_clean.300d.pickled")

    if (embeddings_name not in pretrained_embeddings_urls.keys()):
        logger.error("Cannot load the indicated pretrained embeddings (current '{}' must be one of {})".format(embeddings_name, pretrained_embeddings_urls.keys()))
        exit(1)

    logger.info("Initializing embeddings in folder {}".format(folder))

    num_loaded_embs = 0
    # Set random embeddings for words "out of vocabulary".
    # embeddings = np.zeros((len(word_to_ix), embeddings_size))
    embeddings = np.random.normal(scale=0.6, size=(len(word_to_ix), embeddings_size))

    # Open the embeddings file.
    if embeddings_name == "mimic.fastText.no_clean.300d.pickled":
        # Check if pickle exists.
        file_name = pretrained_embeddings_urls[embeddings_name][1]
        if not io.check_file_existence(folder, file_name):
            # Try to download the pickle.
            url = pretrained_embeddings_urls[embeddings_name][0]
            logger.info("Downloading file '{}' from {}".format(file_name, url))
            io.download(folder, file_name, url)
        else:
            logger.info("File '{}' found in {}".format(embeddings_name, folder))

        # Load word embeddings map.
        word_embedding_map = io.load_pickle(logger, os.path.join(folder, file_name))
        # Iterate over map and cherry pick the vectors that fit our vocabulary.
        for w, index in word_to_ix.items():
            if w in word_embedding_map:
                vector = word_embedding_map[w]
                assert (len(vector) == embeddings_size), "Embeddings size must be equal to the size of pretrained embeddings!"
                embeddings[index] = vector
                num_loaded_embs += 1

    else:

        # Check presence of the file.
        if not io.check_file_existence(folder, embeddings_name):
            # Download and extract wikitext zip.
            io.download_extract_zip_file(logger, folder, pretrained_embeddings_urls[embeddings_name][0], pretrained_embeddings_urls[embeddings_name][1])
        else: 
            logger.info("File '{}' containing pretrained embeddings found in '{}' folder".format(embeddings_name, folder))


        with open(os.path.join(folder, embeddings_name)) as f:
            # Parse file and cherry pick the vectors that fit our vocabulary.
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
    
    logger.info("Loaded {} pretrained embeddings for vocabulary of size {} from {}".format(num_loaded_embs, len(word_to_ix), embeddings_name))

    # Return matrix with embeddings.
    return torch.from_numpy(embeddings).float()
