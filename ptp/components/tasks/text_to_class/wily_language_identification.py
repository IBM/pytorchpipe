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

import ptp.components.mixins.io as io
from .language_identification import LanguageIdentification


class WiLYLanguageIdentification(LanguageIdentification):
    """
    Language identification (classification) task.
    Using WiLI-2018 benchmark _dataset taken from the paper: Thoma, Martin. "The WiLI benchmark dataset for written language identification." arXiv preprint arXiv:1801.07779 (2018). (_arxiv)

    The dataset contains sentences from 235 languages.

    .. _dataset: https://zenodo.org/record/841984
    .. _arxiv: https://arxiv.org/abs/1801.07779
    """
    def __init__(self, name, config):
        """
        Initializes task object. Calls base constructor. Downloads the dataset if not present and loads the adequate files depending on the mode.

        :param name: Name of the component.

        :param class_type: Class type of the component.

        :param config: Dictionary of parameters (read from configuration ``.yaml`` file).
        """
        # Call constructors of parent classes.
        LanguageIdentification.__init__(self, name, WiLYLanguageIdentification, config) 

        # Get absolute path.
        self.data_folder = os.path.expanduser(self.config['data_folder'])

        # Generate the dataset (can be turned off).
        filenames = ["x_train.txt", "y_train.txt", "x_test.txt", "y_test.txt"]
        if not io.check_files_existence(self.data_folder, filenames):
            # Download and unpack.
            url = "https://zenodo.org/record/841984/files/wili-2018.zip?download=1"
            zipfile_name = "wili-2018.zip"
            io.download_extract_zip_file(self.logger, self.data_folder, url, zipfile_name)

        # Select set.
        if self.config['use_train_data']:
            inputs_file = "x_train.txt"
            targets_file = "y_train.txt"
        else:
            inputs_file = "x_test.txt"
            targets_file = "y_test.txt"

        # Load files.
        self.inputs = io.load_string_list_from_txt_file(self.data_folder, inputs_file)
        self.targets = io.load_string_list_from_txt_file(self.data_folder, targets_file)

        # Assert that they are equal in size!
        assert len(self.inputs) == len(self.targets), "Number of inputs loaded from {} not equal to number of targets loaded from {}!".format(inputs_file, targets_file)
