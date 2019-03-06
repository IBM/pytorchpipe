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


from .language_identification import LanguageIdentification

import os
import zipfile

import ptp.utils.io_utils as io


class WiLYLanguageIdentification(LanguageIdentification):
    """
    Language identification (classification) problem.
    Using WiLI benchmark _dataset taken from the paper: Thoma, Martin. "The WiLI benchmark dataset for written language identification." arXiv preprint arXiv:1801.07779 (2018). (_arxiv)

    .. _dataset: https://zenodo.org/record/841984
    .. _arxiv: https://arxiv.org/abs/1801.07779
    """
    def __init__(self, name, params):
        # Call constructors of parent classes.
        LanguageIdentification.__init__(self, name, params) 

        # Set default parameters.
        self.params.add_default_params({
                'data_folder': '~/data/language_identification/wily',
                'use_train_data': True
            })  

        # Get absolute path.
        self.data_folder = os.path.expanduser(self.params['data_folder'])

        # Generate the dataset (can be turned off).
        filenames = ["x_train.txt", "y_train.txt", "x_test.txt", "y_test.txt"]
        if not io.check_files_existence(self.data_folder, filenames):
            self.initialize_dataset()

        # Select set.
        if self.params['use_train_data']:
            inputs_file = "x_train.txt"
            targets_file = "y_train.txt"
        else:
            inputs_file = "x_test.txt"
            targets_file = "y_test.txt"

        # Load files.
        self.inputs = io.load_list_from_txt_file(self.data_folder, inputs_file)
        self.targets = io.load_list_from_txt_file(self.data_folder, targets_file)

        # Assert that they are equal in size!
        assert len(self.inputs) == len(self.targets), "Number of inputs loaded from {} not equal to number of targets loaded from {}!".format(inputs_file, targets_file)

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
