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

import ptp.utils.io_utils as io


class DummyLanguageIdentification(LanguageIdentification):
    """
    Simple Language identification (classification) problem.
    Data taken from the _example.

    .. _example: https://pytorch.org/tutorials/beginner/nlp/deep_learning_tutorial.html
    """
    def __init__(self, name, config):
        """
        Initializes the problem object. Calls base constructor and generates the files, if not present.

        :param name: Name of the component.

        :param class_type: Class type of the component.

        :param config: Dictionary of parameters (read from configuration ``.yaml`` file).
        """
        # Call constructors of parent classes.
        LanguageIdentification.__init__(self, name, DummyLanguageIdentification, config) 

        # Get absolute path.
        self.data_folder = os.path.expanduser(self.config['data_folder'])

        # Generate the dataset (can be turned off).
        filenames = ["x_training.txt", "y_training.txt", "x_test.txt", "y_test.txt"]
        if self.config['regenerate'] or not io.check_files_existence(self.data_folder, filenames):
            self.generate_dataset()

        # Select set.
        if self.config['use_train_data']:
            inputs_file = "x_training.txt"
            targets_file = "y_training.txt"
        else:
            inputs_file = "x_test.txt"
            targets_file = "y_test.txt"

        # Load files.
        self.inputs = io.load_string_list_from_txt_file(self.data_folder, inputs_file)
        self.targets = io.load_string_list_from_txt_file(self.data_folder, targets_file)

        # Assert that they are equal in size!
        assert len(self.inputs) == len(self.targets), "Number of inputs loaded from {} not equal to number of targets loaded from {}!".format(inputs_file, targets_file)


    def generate_dataset(self):
        """
        Method generates dummy dataset for language identification, few (sentence-language) pairs, training and text sets.
        """
        self.logger.info("Generating dummy dataset in {}".format(self.data_folder))

        # "Training" set.
        x_training_data = [
            "me gusta comer en la cafeteria",
            "Give it to me", 
            "No creo que sea una buena idea",
            "No it is not a good idea to get lost at sea"]
        io.save_string_list_to_txt_file(self.data_folder, 'x_training.txt', x_training_data)

        y_training_data = [
            "SPANISH",
            "ENGLISH",
            "SPANISH",
            "ENGLISH"]
        io.save_string_list_to_txt_file(self.data_folder, 'y_training.txt', y_training_data)

        # "Test" set.
        x_test_data = [
            "Yo creo que si",
            "it is lost on me"]
        io.save_string_list_to_txt_file(self.data_folder, 'x_test.txt', x_test_data)

        y_test_data = [
            "SPANISH",
            "ENGLISH"]
        io.save_string_list_to_txt_file(self.data_folder, 'y_test.txt', y_test_data)

        self.logger.info("Initialization successfull")
