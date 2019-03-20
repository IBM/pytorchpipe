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


from ptp.components.problems.problem import Problem
from ptp.data_types.data_definition import DataDefinition

import os
import zipfile

import ptp.utils.io_utils as io


class WiLYNGramLanguageModeling(Problem):
    """
    N-gram Language Modeling problem.
    By default it is using sentences from the WiLI benchmark _dataset taken from the paper: Thoma, Martin. "The WiLI benchmark dataset for written language identification." arXiv preprint arXiv:1801.07779 (2018). (_arxiv)

    .. _dataset: https://zenodo.org/record/841984
    .. _arxiv: https://arxiv.org/abs/1801.07779
    """
    def __init__(self, name, params):
        # Call constructors of parent classes.
        Problem.__init__(self, name, params) 

        # Set key mappings.
        self.key_inputs = self.mapkey("inputs")
        self.key_targets = self.mapkey("targets")

        # Set default parameters.
        self.params.add_default_params({
                'data_folder': '~/data/language_identification/wily',
                'use_train_data': True,
                'context': 2,
            })  

        # Get absolute path.
        self.data_folder = os.path.expanduser(self.params['data_folder'])

        # Get size of context.
        self.context = self.params['context']

        # Select set.
        if self.params['use_train_data']:
            inputs_file = "x_train.txt"
            ngrams_file =  "ngrams_train.txt"
        else:
            inputs_file = "x_test.txt"
            ngrams_file =  "ngrams_test.txt"

        # Check if we can load ngrams.
        if not io.check_file_existence(self.data_folder, ngrams_file):
            # Sadly not, we have to generate them.
            if not io.check_file_existence(self.data_folder, inputs_file):
                # Even worst - we have to download wily.
                self.initialize_dataset()

            # Load file.
            inputs = io.load_string_list_from_txt_file(self.data_folder, inputs_file)

            self.logger.info("Please wait, generating n-grams...")
            self.ngrams_sent = []
            # Now we have to split sentencese into n-grams.
            for sentence in inputs:
                # Split sentence into words.
                words = sentence.split()
                
                # Build a list of ngrams.
                for i in range(len(words) - self.context):
                    ngram = [words[j] for j in range(i, i+1+self.context)]
                    self.ngrams_sent.append(' '.join(ngram))

            # Assert that they are any ngrams there!
            assert len(self.ngrams_sent) > 0, "Number of n-grams generated on the basis of '{}' must be greater than 0!".format(inputs_file)
            # Done.
            self.logger.info("Generated {} n-grams, example:\n{}".format(len(self.ngrams_sent), self.ngrams_sent[0]))

            self.logger.info("Saving {} n-grams to file '{}'".format(len(self.ngrams_sent), ngrams_file))
            # N-grams generated, save them to file.
            io.save_string_list_to_txt_file(self.data_folder, ngrams_file, self.ngrams_sent)
        else:
            self.logger.info("Please wait, loading n-grams from file '{}'".format(ngrams_file))
            # Load file.
            self.ngrams_sent = io.load_string_list_from_txt_file(self.data_folder, ngrams_file)

            # Assert that they are equal in size!
            assert len(self.ngrams_sent) > 0, "Number of n-grams loaded from {} must be greater than 0!".format(ngrams_file)
            # Done.
            self.logger.info("Loaded {} n-grams, example:\n{}".format(len(self.ngrams_sent), self.ngrams_sent[0]))
        
        # Split words in n-grams.
        self.ngrams = [ngram.split() for ngram in self.ngrams_sent]
        

    def output_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of output data produced the component.

        :return: dictionary containing output data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {
            self.key_indices: DataDefinition([-1, 1], [list, int], "Batch of sample indices [BATCH_SIZE] x [1]"),
            self.key_inputs: DataDefinition([-1, self.context, 1], [list, list, str], "Batch of sentences, each being a context consisint of several words [BATCH_SIZE] x [CONTEXT_SIZE] x [WORD]"),
            self.key_targets: DataDefinition([-1, 1], [list, str], "Batch of targets, each being a single word [BATCH_SIZE] x [WORD]")
            }


    def __len__(self):
        """
        Returns the "size" of the "problem" (total number of samples).

        :return: The size of the problem.
        """
        return len(self.ngrams)


    def __getitem__(self, index):
        """
        Getter method to access the dataset and return a sample.

        :param index: index of the sample to return.
        :type index: int

        :return: ``DataDict({'inputs','targets'})``

        """
        # Return data_dict.
        data_dict = self.create_data_dict(index)
        data_dict[self.key_inputs] = self.ngrams[:self.context]
        data_dict[self.key_targets] = self.ngrams[-1] # Last word
        return data_dict


    def initialize_dataset(self):
        """
        Method downloads dataset from WiLI project url and extract the files.
        """
        self.logger.info("Initializing dataset in folder {}".format(self.data_folder))

        # Download url.
        url = "https://zenodo.org/record/841984/files/wili-2018.zip?download=1"
        zip_filename = "wili-2018.zip"

        if not io.check_file_existence(self.data_folder, zip_filename):
            self.logger.info("Downloading file {} containing WiLI dataset from {}".format(zip_filename, url))
            io.download(self.data_folder, zip_filename, url)
        else:
            self.logger.info("File {} found in {}".format(zip_filename, self.data_folder))


        # Extract data from zip.
        self.logger.info("Extracting dataset from {}".format(zip_filename))
        with zipfile.ZipFile(self.data_folder + "/" + zip_filename, 'r') as zip_ref:
            zip_ref.extractall(self.data_folder)

        self.logger.info("Initialization successfull")
