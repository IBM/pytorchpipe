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
import csv

from ptp.configuration.configuration_error import ConfigurationError

def generate_word_mappings_from_source_files(logger, folder, source_files):
    """
    Load list of files (containing raw text) and creates (word:index) mappings from all words (tokens).
    Indexing starts from 0.

    :param logger: Logger object.

    :param folder: Relative path to to the folder.
    :type folder: str

    :param source_files: Source files (separated by commas)

    :return: Dictionary with (word:index) mappings
    """
    # Check if there are any source files to load.
    if len(source_files) == 0:
        logger.error("Cannot create dictionary: list of vocabulary source files is empty, please provide comma separated list of files to be processed")
        exit(-1)

    # Get absolute path.
    folder = os.path.expanduser(folder)

    # Dictionary word_to_ix maps each word in the vocab to a unique integer.
    word_to_ix = {}
    # Add special word <PAD> that we will use that during padding.
    # As a result, the "real" enumeration will start from 1.
    word_to_ix['<PAD>'] = 0

    for filename in source_files.split(','):
        # filename + path.
        fn = folder+ '/' + filename
        if not os.path.exists(fn):
            logger.warning("Cannot load tokens files from '{}' because file does not exist".format(fn))
            continue
        # File exists, try to parse.
        content = open(fn).read()
        # Parse tokens.
        for word in content.split():
            # If new token.
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    logger.info("Generated mappings of size {}".format(len(word_to_ix)))
    return word_to_ix


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
