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

__author__ = "Alexis Asseman"

import os
import random
import tempfile
import unicodedata
import re

from nltk.tokenize import WhitespaceTokenizer

import ptp.components.utils.io as io
from ptp.configuration import ConfigurationError
from ptp.components.problems.problem import Problem
from ptp.data_types.data_definition import DataDefinition


class TranslationPairs(Problem):
    """
    Bilingual sentence pairs from http://www.manythings.org/anki/.
    Only some pairs are included here, but many more are available on the website.
    Will download the requested language pair if necessary, normalize and tokenize the sentences, and will cut the data into train, valid, test sets.

    Resulting tokens that are shorter than the specified length are then passed to samples (source/target) as list of tokens (set by the user in configuration file).
    """
    def __init__(self, name, config):
        """
        The init method downloads the required files, loads the file associated with a given subset (train/valid/test), 
        concatenates all sencentes and tokenizes them using NLTK's WhitespaceTokenizer.

        :param name: Name of the component.

        :param class_type: Class type of the component.

        :param config: Dictionary of parameters (read from configuration ``.yaml`` file).
        """
        # Call constructor of parent classes.
        Problem.__init__(self, name, TranslationPairs, config) 

        # Set streams key mappings.
        self.key_sources = self.stream_keys["sources"]
        self.key_targets = self.stream_keys["targets"]

        # Get absolute path to data folder.
        self.data_folder = os.path.expanduser(self.config['data_folder'])

        # Get dataset.
        if (self.config['dataset'] is None) or (self.config['dataset'] not in ["eng-fra", "eng-pol"]):
            raise ConfigurationError("Problem supports only 'dataset' options: 'eng-fra', 'eng-pol'")
        dataset = self.config['dataset']

        # Get (sub)set: train/valid/test.
        if (self.config['subset'] is None) or (self.config['subset'] not in ['train', 'valid', 'test']):
            raise ConfigurationError("Problem supports one 'subset' options: 'train', 'valid', 'test' ")
        subset = self.config['subset']

        # Extract source and target language name
        self.lang_source = self.config['dataset'].split('-')[0]
        self.lang_target = self.config['dataset'].split('-')[1]


        # Names of files used by this problem.
        filenames = [
            self.lang_source + ".train.txt",
            self.lang_target + ".train.txt", 
            self.lang_source + ".valid.txt", 
            self.lang_target + ".valid.txt", 
            self.lang_source + ".test.txt", 
            self.lang_target + ".test.txt"
            ]

        # Initialize dataset if files do not exist.
        if not io.check_files_existence(os.path.join(self.data_folder, dataset), filenames):
            # Set url and source filename depending on dataset.
            url = "https://www.manythings.org/anki/" + self.lang_target + "-" + self.lang_source + ".zip"
            zipfile_name = "translate_" + self.lang_target + "_" + self.lang_source + ".zip"

            with tempfile.TemporaryDirectory() as tmpdirname:
                # Download and extract wikitext zip.
                io.download_extract_zip_file(self.logger, tmpdirname, url, zipfile_name)

                # Create train, valid, test files from the downloaded file
                lines = io.load_string_list_from_txt_file(tmpdirname, self.lang_target + ".txt")

                # Shuffle the lines
                random.seed(42)
                random.shuffle(lines)

                # Split english and french pairs
                lines_source = [self.normalizeString(l.split('\t')[0]) for l in lines]
                lines_target = [self.normalizeString(l.split('\t')[1]) for l in lines]

                # Cut dataset into train (90%), valid (5%), test (5%) files
                test_index = len(lines) // 20
                valid_index = test_index + (len(lines) // 20)

                os.makedirs(os.path.join(self.data_folder, dataset), exist_ok=True)
                
                with open(os.path.join(os.path.join(self.data_folder, dataset), self.lang_source + ".test.txt"), mode='w+') as f:
                    f.write('\n'.join(lines_source[0:test_index]))
                with open(os.path.join(os.path.join(self.data_folder, dataset), self.lang_target + ".test.txt"), mode='w+') as f:
                    f.write('\n'.join(lines_target[0:test_index]))

                with open(os.path.join(os.path.join(self.data_folder, dataset), self.lang_source + ".valid.txt"), mode='w+') as f:
                    f.write('\n'.join(lines_source[test_index:valid_index]))
                with open(os.path.join(os.path.join(self.data_folder, dataset), self.lang_target + ".valid.txt"), mode='w+') as f:
                    f.write('\n'.join(lines_target[test_index:valid_index]))

                with open(os.path.join(os.path.join(self.data_folder, dataset), self.lang_source + ".train.txt"), mode='w+') as f:
                    f.write('\n'.join(lines_source[valid_index:]))
                with open(os.path.join(os.path.join(self.data_folder, dataset), self.lang_target + ".train.txt"), mode='w+') as f:
                    f.write('\n'.join(lines_target[valid_index:]))

        else:
            self.logger.info("Files {} found in folder '{}'".format(filenames, self.data_folder))


        # Load the lines
        lines_source = io.load_string_list_from_txt_file(os.path.join(self.data_folder, dataset), self.lang_source + "."+subset+".txt")
        lines_target = io.load_string_list_from_txt_file(os.path.join(self.data_folder, dataset), self.lang_target + "."+subset+".txt")

        # Get the required sample length.
        self.sentence_length = self.config['sentence_length']

        # Separate into src - tgt sentence pairs + tokenize
        tokenizer = WhitespaceTokenizer()
        self.sentences_source = []
        self.sentences_target = []
        for s_src, s_tgt in zip(lines_source, lines_target):
            src = tokenizer.tokenize(s_src)
            tgt = tokenizer.tokenize(s_tgt)
            # Keep only the pairs that are shorter or equal to the requested length
            # If self.sentence_length < 0, then give all the pairs regardless of length
            if (len(src) <= self.sentence_length and len(tgt) <= self.sentence_length) \
                or self.sentence_length < 0:
                self.sentences_source += [src]
                self.sentences_target += [tgt]

        self.logger.info("Load text consisting of {} sentences".format(len(self.sentences_source)))

        # Calculate the size of dataset.
        self.dataset_length = len(self.sentences_source)

        # Display exemplary sample.
        self.logger.info("Exemplary sample:\n  source: {}\n  target: {}".format(self.sentences_source[0], self.sentences_target[0]))
        

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

    # Turn a Unicode string to plain ASCII, thanks to
    # https://stackoverflow.com/a/518232/2809427
    @staticmethod
    def unicodeToAscii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    # Lowercase, trim, and remove non-letter characters
    # https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
    def normalizeString(self, s):
        s = self.unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

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
        data_dict[self.key_sources] = self.sentences_source[index]
        data_dict[self.key_targets] = self.sentences_target[index]
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

