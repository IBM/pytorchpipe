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


class TokenEncoder(Component):
    """
    Abstract class responsible for encoding tokens. Please use derrived classes.
    """
    def __init__(self, name, class_type, config):
        """
        Initializes the component.

        :param name: Component name (read from configuration file).
        :type name: str

        :param class_type: Class type of the component (derrived from this class).

        :param config: Dictionary of parameters (read from the configuration ``.yaml`` file).
        :type config: :py:class:`ptp.configuration.ConfigInterface`

        """
        # Call constructors of parent classes.
        Component.__init__(self, name, class_type, config)

        # Read the actual configuration.
        self.data_folder = os.path.expanduser(config['data_folder'])
        self.source_files = config['source_files']
        self.encodings_file = config['encodings_file']
        self.mode_regenerate = config['regenerate']

        # Additional entries in the vocabulary.
        self.additional_tokens = config["additional_tokens"].split(',')

        # Default name mappings for all encoders.
        self.key_inputs = self.stream_keys["inputs"]
        self.key_outputs = self.stream_keys["outputs"]

        # Encodings file.
        encodings_file_path = self.data_folder + "/" + self.encodings_file

        # Check whether we want to (re)generate new  or load existing encodings.
        if self.mode_regenerate or not os.path.exists(encodings_file_path):
            # Generate new encodings.
            self.word_to_ix = wm.generate_word_mappings_from_source_files(self.logger, self.data_folder, self.source_files)
            assert (len(self.word_to_ix) > 0), "The created encodings list is empty!"
            # Ok, save mappings, so next time we will simply load them.
            wm.save_word_mappings_to_csv_file(self.logger, self.data_folder, self.encodings_file, self.word_to_ix)
        else:
            # Load encodings.
            self.word_to_ix = wm.load_word_mappings_from_csv_file(self.logger, self.data_folder, self.encodings_file)
            assert (len(self.word_to_ix) > 0), "The loaded encodings list is empty!"

        # Check if additional tokens are present.
        for word in self.additional_tokens:
            # If new token.
            if word not in self.word_to_ix:
                self.word_to_ix[word] = len(self.word_to_ix)
