#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) IBM Corporation 2019
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
import urllib
import time
import sys
import logging

from ptp.utils.data_dict import DataDict
from ptp.utils.app_state import AppState


class Component(object):
    def __init__(self, name, params):
        """
        Initializes the component.

        This constructor:

        - stores a pointer to ``params``:

            >>> self.params = params

        - sets a problem name:

            >>> self.name = name

        - initializes the logger.

            >>> self.logger = logging.getLogger(self.name)        

        - sets the access to ``AppState``: for dtype, visualization flag etc.

            >>> self.app_state = AppState()

        :param name: Name of the component.

        :param params: Dictionary of parameters (read from configuration ``.yaml`` file).
        """
        self.name = name
        self.params = params

        # Initialize logger.
        self.logger = logging.getLogger(self.name)        

        # Set default (empty) data definitions and default_values.
        self.data_definitions = {}
        self.default_values =  {}

        # Initialize the "name mapping facility".
        params.add_default_params({"keymappings": {}})
        self.keymappings = params["keymappings"]

        # Get access to AppState: for globals, visualization flag etc.
        self.app_state = AppState()


    def mapkey(self, key_name):
        """
        Method responsible for checking whether name exists in the mappings.
        
        :key_name: name of the key to be mapped.

        :return: Mapped name or original key name (if it does not exist in mappings list).
        """
        return self.keymappings.get(key_name, key_name)

    def __call__(self, data_dict):
        """
        Method responsible for processing the data dict.

        :param data_dict: :py:class:`ptp.utils.DataDict` object containing data to encode and that will be extended with encoded results.
        """
        pass

    # Function to make check and download easier
    def check_and_download(self, file_folder_to_check, url=None, download_name='~/data/downloaded'):
        """
        Checks whether a file or folder exists at given path (relative to storage folder), \
        otherwise downloads files from the given URL.

        :param file_folder_to_check: Relative path to a file or folder to check to see if it exists.
        :type file_folder_to_check: str

        :param url: URL to download files from.
        :type url: str

        :param download_name: What to name the downloaded file. (DEFAULT: "downloaded").
        :type download_name: str

        :return: False if file was found, True if a download was necessary.

        """

        file_folder_to_check = os.path.expanduser(file_folder_to_check)
        if not (os.path.isfile(file_folder_to_check) or os.path.isdir(file_folder_to_check)):
            if url is not None:
                self.logger.info('Downloading {}'.format(url))
                urllib.request.urlretrieve(url, os.path.expanduser(download_name), reporthook)
                return True
            else:
                return True
        else:
            self.logger.info('Dataset found at {}'.format(file_folder_to_check))
            return False

# Progress bar function
def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
            start_time = time.time()
            return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                     (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()
