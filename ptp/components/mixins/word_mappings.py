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
import csv

class WordMappings(object):
    """
    Mixin class that handles the initialization of (word:index) mappings.
    Assumes that it is mixed-in into class that is derived from the component.
    .. warning::
        Constructor (__init__) of the Component class has to be called before component of the mixin WordMapping class.

    """
    def __init__(self): #, name, class_type, config):
        """
        Initializes the (word:index) mappings.

        Assumes that Component was initialized in advance, which means that the self object possesses the following objects:
            - self.config
            - self.globals
            - self.logger

        """
        # Read the actual configuration.
        self.data_folder = os.path.expanduser(self.config['data_folder'])

        # Source and resulting (indexed) vocabulary.
        self.source_vocabulary_files = self.config['source_vocabulary_files']
        self.word_mappings_file = self.config['word_mappings_file']
        
        # Set aboslute path to file with word mappings.
        word_mappings_file_path = os.path.join(os.path.expanduser(self.data_folder), self.word_mappings_file)

        # Check if we want to export word mappings to globals.
        if self.config["import_word_mappings_from_globals"]:
            self.word_to_ix = self.globals["word_mappings"]
            assert (len(self.word_to_ix) > 0), "The word mappings imported from global variables are empty!"
            # We could also get vocabulary_size from globals... but what for;)

        elif self.word_mappings_file != "" and not self.config['regenerate']:
            if not os.path.exists(word_mappings_file_path):
                self.logger.warning("Cannot load word mappings from '{}' because the file does not exist".format(word_mappings_file_path))
                # This is a show stopper.
                exit(-1)

            # Try to load the preprocessed word mappings.
            self.word_to_ix = load_word_mappings_from_csv_file(self.logger, self.data_folder, self.word_mappings_file)
            assert (len(self.word_to_ix) > 0), "The word mappings loaded from file are empty!"

        else:
            # Try to generate new word mappings from source files.
            self.word_to_ix = generate_word_mappings_from_source_files(self.logger, self.data_folder, self.source_vocabulary_files)
            assert (len(self.word_to_ix) > 0), "The word mappings generated from sources are empty!"
            # Ok, save mappings, so next time we will simply load them.
            save_word_mappings_to_csv_file(self.logger, self.data_folder, self.word_mappings_file, self.word_to_ix)

        # Check if additional tokens are present.
        self.additional_tokens = self.config["additional_tokens"].split(',')
        for word in self.additional_tokens:
            # If new token.
            if word != '' and word not in self.word_to_ix:
                self.word_to_ix[word] = len(self.word_to_ix)

        if self.config["export_pad_index_to_globals"]:
            self.globals["pad_index"] = self.word_to_ix['<PAD>']

        self.logger.info("Initialized word mappings with vocabulary of size {}".format(len(self.word_to_ix)))

        # Check if we want to export word mappings to globals.
        if self.config["export_word_mappings_to_globals"]:
            self.globals["word_mappings"] = self.word_to_ix
            # Export vocabulary size to globals.
            self.globals["vocabulary_size"] = len(self.word_to_ix)



def load_word_mappings_from_csv_file(logger, folder, filename):
    """
    Loads (word:index) mappings from csv file.

    .. warning::
            There is an assumption that file will contain key:value pairs (no content checking for now!)

    :param logger: Logger object.

    :param folder: Relative path to to the folder.
    :type folder: str

    :param filename: File with encodings (absolute path + filename).

    :return: dictionary with word:index keys
    """        
    file_path = os.path.join(os.path.expanduser(folder), filename)

    if not os.path.exists(file_path):
        logger.warning("Cannot load word mappings from '{}' because the file does not exist".format(file_path))

    with open(file_path, mode='rt') as csvfile:
        # Check the presence of the header.
        sniffer = csv.Sniffer()
        first_bytes = str(csvfile.read(256))
        has_header = sniffer.has_header(first_bytes)
        # Rewind.
        csvfile.seek(0)  
        reader = csv.reader(csvfile)
        # Skip the header row.
        if has_header:
            next(reader)  
        # Read the remaining rows.
        word_to_ix = {rows[0]:int(rows[1]) for rows in reader}

    logger.info("Loaded mappings of size {}".format(len(word_to_ix)))
    return word_to_ix


def save_word_mappings_to_csv_file(logger, folder, filename, word_to_ix, fieldnames = ["word","index"]):
    """
    Saves (word:index) mappings dictionary to a file.

    :param logger: Logger object.

    :param folder: Relative path to to the folder.
    :type folder: str

    :param filename: Name of file with encodings.
    
    :param word_to_ix: Dictionary with word:index mappings to be saved.
    
    """
    # Expand path.
    folder = os.path.expanduser(folder)
    # Make sure directory exists.
    os.makedirs(os.path.dirname(folder +'/'), exist_ok=True)

    file_path = os.path.join(folder, filename)

    with open(file_path, mode='w+') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # Create header.
        writer.writeheader()

        # Write word-index pairs.
        for (k,v) in word_to_ix.items():
            #print("{} : {}".format(k,v))
            writer.writerow({fieldnames[0]:k, fieldnames[1]: v})

    logger.info("Saved mappings of size {} to file '{}'".format(len(word_to_ix), file_path))

def pad_trunc_list(l: list, length: int, padding_value = 0, eos_value = None):
    """
    Will apply padding / clipping to list to meet requested length.
    Works on the list in-place.

    :param l: List to manipulate

    :param length: Target length

    :param padding_value: Value to fill when padding. Default is int(0).

    :return: None
    """
    if len(l) < length:
        if eos_value is not None:
            l.append(eos_value)
        l.extend([padding_value]*(length-len(l)))
        
    elif len(l) > length:
        #print("pad_trunc_list to cat!: {}".format(len(l)))
        #exit(1)
        del l[length:]
        if eos_value is not None:
            l[length-1] = eos_value
