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

import ptp.utils.io_utils as io


class WillyLanguageIdentification(LanguageIdentification):
    """
    Language identification (classification) problem.
    Using WiLI benchmark _dataset taken from the paper: Thoma, Martin. "The WiLI benchmark dataset for written language identification." arXiv preprint arXiv:1801.07779 (2018). (_arxiv)

    .. _dataset: https://zenodo.org/record/841984
    .. _arxiv: https://arxiv.org/abs/1801.07779
    """
    def __init__(self, name, params):
        # Call constructors of parent classes.
        LanguageIdentification.__init__(self, name, params) 

        # Download url: https://zenodo.org/record/841984/files/wili-2018.zip?download=1

        # Set default parameters.
        self.params.add_default_params({
            'generate': True
            })
        # Generate the dataset (can be turned off).    
        if self.params['generate']:
            self.generate_dummy_dataset()

        if self.use_train_data:
            inputs_file = "x_training.txt"
            targets_file = "y_training.txt"
        else:
            inputs_file = "x_test.txt"
            targets_file = "y_test.txt"

        # Load files.
        self.inputs = io.load_list_from_txt_file(self.data_folder, inputs_file)
        self.targets = io.load_list_from_txt_file(self.data_folder, targets_file)

        # Assert that they are equal in size!
        assert len(self.inputs) == len(self.targets), "Number of inputs loaded from {} not equal to number of targets loaded from {}!".format(inputs_file, targets_file)

    def __len__(self):
        """
        Returns the "size" of the "problem" (total number of samples).

        :return: The size of the problem.
        """
        return len(self.inputs)