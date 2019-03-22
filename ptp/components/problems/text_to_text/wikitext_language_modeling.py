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

from nltk.tokenize import WhitespaceTokenizer

import ptp.utils.io_utils as io
from ptp.configuration import ConfigurationError
from ptp.components.problems.problem import Problem
from ptp.data_types.data_definition import DataDefinition


class WikiTextLanguageModeling(Problem):
    """
    Language modeling problem using WikiText-2 (_dataset2) / WikiText-103 (_dataset103) datasets, featured at the Salesforce _website.

    Problem downloads the files, loads the file associated with a given subset (train/valid/test), concatenates all sencentes and tokenizes them using NLTK's WhitespaceTokenizer.
    
    Resulting tokens are then passed to samples (source/target) as list of tokens of a given length (set by the user in configuration file).

    Associated paper: Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer Sentinel Mixture Models (2016) (_arxiv)

    .. _website: https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/
    .. _dataset2: https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip
    .. _dataset103: https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip
    .. _arxiv: https://arxiv.org/abs/1609.07843
    """
    def __init__(self, name, params):
        """
        The init method downloads the required files, loads the file associated with a given subset (train/valid/test), 
        concatenates all sencentes and tokenizes them using NLTK's WhitespaceTokenizer.

        It also stores the intermediate results, so for example, it file with tokenized set is found, it simply loads it.

        :param name: Name of the component.

        :param class_type: Class type of the component.

        :param params: Dictionary of parameters (read from configuration ``.yaml`` file).
        """
        # Call constructor of parent classes.
        Problem.__init__(self, name, WikiTextLanguageModeling, params) 

        # Set streams key mappings.
        self.key_sources = self.get_stream_key("sources")
        self.key_targets = self.get_stream_key("targets")

        # Get absolute path to data folder.
        self.data_folder = os.path.expanduser(self.params['data_folder'])

        # Get dataset.
        if (self.params['dataset'] is None) or (self.params['dataset'] not in ["wikitext-2", "wikitext-103"]):
            raise ConfigurationError("Problem supports two 'dataset' options: 'wikitext-2', 'wikitext-103' ")
        dataset = self.params['dataset']

        # Get (sub)set: train/valid/test.
        if (self.params['subset'] is None) or (self.params['subset'] not in ['train', 'valid', 'test']):
            raise ConfigurationError("Problem supports three 'subset' options: 'train', 'valid', 'test' ")
        subset = self.params['subset']

        # Check if file with tokenized words exists.
        filename_tokenized_words = "wiki."+self.params['subset']+".tokenized_words"

        if not io.check_files_existence(self.data_folder, filename_tokenized_words):
            # If not, we must generate (and save it) using source files.

            # Names of files used by this problem.
            filenames = ["wiki.train.tokens", "wiki.valid.tokens", "wiki.test.tokens"]

            # Initialize dataset if files do not exist.
            if not io.check_files_existence(self.data_folder, filenames):
                # Set url and source filename depending on dataset.
                if dataset == "wikitext-2":
                    url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip"
                    zipfile_name = "wikitext-2-v1.zip"
                else: 
                    url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip"
                    zipfile_name = "wikitext-103-v1.zip"

                # Download and extract wikitext zip.
                io.download_extract_zip_file(self.logger, self.data_folder, url, zipfile_name)

                # Move extracted files to the right folder.
                io.move_files_between_dirs(self.logger, os.path.join(self.data_folder, dataset) , self.data_folder, filenames)
            else:
                self.logger.info("Files {} found in folder '{}'".format(filenames, self.data_folder))


            # Load the whole sentences.
            sentences = io.load_string_list_from_txt_file(self.data_folder, "wiki."+subset+".tokens")
            self.logger.info("Loaded {} sentences from the 'wiki.{}.tokens' subset".format(len(sentences), subset))

            # Generate text full of tokens.
            self.logger.info("Please wait, using NLTK to tokenize the loaded sentences...")
            # Create a single text by replacing newlines with <eos> tokens.
            text = " <eos> ".join(sentences)
            # Tokenize.
            tokenizer = WhitespaceTokenizer()
            self.tokens = tokenizer.tokenize(text)
            # Save fo file.
            io.save_string_list_to_txt_file(self.data_folder, filename_tokenized_words, self.tokens)
            self.logger.info("Created text consisting of {} tokens and saved it to '{}'".format(len(self.tokens), filename_tokenized_words))
        else:
            # Ok, file with tokens exists, load it.
            self.tokens = io.load_string_list_from_txt_file(self.data_folder, filename_tokenized_words)
            self.logger.info("Load text consisting of {} tokens from '{}'".format(len(self.tokens), filename_tokenized_words))

        # Get the required sample length.
        self.sentence_length = self.params['sentence_length']
        # Calculate the size of dataset.
        self.dataset_length = len(self.tokens) - self.sentence_length - 1 # as target is "shifted" by 1.

        # Display exemplary sample.
        self.logger.info("Exemplary sample:\n  source: {}\n  target: {}".format(self.tokens[0:self.sentence_length], self.tokens[1:self.sentence_length+1]))
        




    def output_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of output data produced the component.

        :return: dictionary containing output data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {
            self.key_indices: DataDefinition([-1, 1], [list, int], "Batch of sample indices [BATCH_SIZE] x [1]"),
            self.key_sources: DataDefinition([-1, self.sentence_length, 1], [list, list, str], "Batch of input sentences, each consisting of several words [BATCH_SIZE] x [SENTENCE_LENGTH] x [string]"),
            self.key_targets: DataDefinition([-1, self.sentence_length, 1], [list, list, str], "Batch of target sentences, each consisting of several words [BATCH_SIZE] x [SENTENCE_LENGTH] x [string]")
            }


    def __len__(self):
        """
        Returns the "size" of the "problem" (total number of samples).

        :return: The size of the problem.
        """
        return self.dataset_length


    def __getitem__(self, index):
        """
        Getter method to access the dataset and return a sample.

        :param index: index of the sample to return.
        :type index: int

        :return: ``DataDict({'indices', sources','targets'})``

        """
        # Return data_dict.
        data_dict = self.create_data_dict(index)
        data_dict[self.key_sources] = self.tokens[index:index+self.sentence_length]
        data_dict[self.key_targets] = self.tokens[index+1:index+self.sentence_length+1] # target is "shifted" by 1.
        #print("problem: index = {} source = {} target = {}".format(index, data_dict[self.key_sources], data_dict[self.key_targets]))
        return data_dict

    def collate_fn(self, batch):
        """
        Generates a batch of samples from a list of individuals samples retrieved by :py:func:`__getitem__`.

        :param batch: List of :py:class:`ptp.utils.DataDict` retrieved by :py:func:`__getitem__`
        :type batch: list

        :return: DataDict containing the created batch.

        """
        # Collate indices.
        data_dict = self.create_data_dict([sample[self.key_indices] for sample in batch])
        # Collate sources.
        data_dict[self.key_sources] = [sample[self.key_sources] for sample in batch]
        data_dict[self.key_targets] = [sample[self.key_targets] for sample in batch]
        return data_dict

