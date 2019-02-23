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
import ptp.utils.io_utils as io

from ptp.core_types.component import Component


class TokenEncoder(Component):
    """
    Abstract class responsible for encoding tokens. Please use derrived classes.
    """
    def __init__(self, name, params):
        # Call constructors of parent classes.
        Component.__init__(self, name, params)

        # Set default parameters.
        params.add_default_params({
            'data_folder': '~/data/',
            'source_files': '', # Source files
            'encodings_file': 'default_encodings.csv', # File containing encodings
            'regenerate': False # True means that it will be regenerated despite the existence of file.
            })

        # Read the actual configuration.
        self.data_folder = params['data_folder']
        self.source_files = params['source_files']
        self.encodings_file = params['encodings_file']
        self.mode_regenerate = params['regenerate']

        # Default name mappings for all encoders.
        self.key_inputs = self.mapkey("inputs")
        self.key_outputs = self.mapkey("outputs")

        # Encodings file.
        encodings_file_path = os.path.expanduser(self.data_folder) + "/" + self.encodings_file

        # Check whether we want to (re)generate new  or load existing encodings.
        if self.mode_regenerate or not os.path.exists(encodings_file_path):
            # Generate new encodings.
            self.word_to_ix = self.create_encodings(self.data_folder, self.source_files)
            assert (len(self.word_to_ix) > 0), "The created encodings list is empty!"
            # Ok, save necodings, so next time we will simply load them.
            io.save_dict_to_csv_file(self.data_folder, self.encodings_file, self.word_to_ix, ['word', 'index'])
        else:
            # Load encodings.
            self.word_to_ix = io.load_dict_from_csv_file(self.data_folder, self.encodings_file)
            assert (len(self.word_to_ix) > 0), "The loaded encodings list is empty!"

        # Ok, we are ready to go!

    def create_encodings(self, data_folder, source_files):
        """
        Load list of files (containing raw text) and creates a dictionary from all words (tokens).
        Indexing starts from 0.

        :return: Dictionary with mapping "word-to-index".
        """
        assert len(source_files) > 0, 'Cannot create dictionary: "source_files" is empty, please provide comma separated list of files to be processed'
        # Get absolute path.
        data_folder = os.path.expanduser(data_folder)

        # Dictionary word_to_ix maps each word in the vocab to a unique integer.
        word_to_ix = {}

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
