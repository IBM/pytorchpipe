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

import ptp.components.utils.word_mappings as wm
from ptp.components.component import Component


class WordMapping(Component):
    """
    Mixin class that handles the initialization of (word:index) mappings.
    """
    def __init__(self, name, class_type, config):
        """
        Initializes the (word:index) mappings.

        Loads parameters from configuration, 

        :param name: Component name (read from configuration file).
        :type name: str

        :param class_type: Class type of the component (derrived from this class).

        :param config: Dictionary of parameters (read from the configuration ``.yaml`` file).
        :type config: :py:class:`ptp.configuration.ConfigInterface`

        """
        # Call constructors of parent classes.
        Component.__init__(self, name, class_type, config)

        # Read the actual configuration.
        self.data_folder = os.path.expanduser(self.config['data_folder'])

        # Source and resulting (indexed) vocabulary.
        self.source_vocabulary_files = self.config['source_vocabulary_files']
        self.word_mappings_file = self.config['word_mappings_file']
        
        # Set aboslute path to file with word mappings.
        word_mappings_file_path = os.path.join(os.path.expanduser(self.data_folder), self.word_mappings_file)
        print(word_mappings_file_path)

        # Check if we want to load preprocessed mappings.
        if self.word_mappings_file != "" and os.path.exists(word_mappings_file_path) and not self.config['regenerate']:
            # Load word mappings.
            self.word_to_ix = wm.load_word_mappings_from_csv_file(self.logger, self.data_folder, self.word_mappings_file)
            assert (len(self.word_to_ix) > 0), "The loaded word mappings list is empty!"
        else:
            # Try to generate new word mappings from source files.
            self.word_to_ix = wm.generate_word_mappings_from_source_files(self.logger, self.data_folder, self.source_vocabulary_files)
            assert (len(self.word_to_ix) > 0), "The created word mappings cannot be empty!"
            # Ok, save mappings, so next time we will simply load them.
            wm.save_word_mappings_to_csv_file(self.logger, self.data_folder, self.word_mappings_file, self.word_to_ix)

        # Check if additional tokens are present.
        self.additional_tokens = self.config["additional_tokens"].split(',')
        for word in self.additional_tokens:
            # If new token.
            if word != '' and word not in self.word_to_ix:
                self.word_to_ix[word] = len(self.word_to_ix)

        self.logger.info("Initialized word mappings with vocabulary of size {}".format(len(self.word_to_ix)))

        # Export word mappings and vocabulary size to globals.
        self.globals["word_mappings"] = self.word_to_ix
        self.globals["vocabulary_size"] = len(self.word_to_ix)

