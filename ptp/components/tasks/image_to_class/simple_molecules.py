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
import csv
from PIL import Image

import torch
from torchvision import transforms

from ptp.components.tasks.task import Task
from ptp.data_types.data_definition import DataDefinition

from ptp.configuration.config_parsing import get_value_from_dictionary
from ptp.configuration.configuration_error import ConfigurationError


class SimpleMolecules(Task):
    """
    Simple molecule classification task.

    """

    def __init__(self, name, config):
        """
        Initializes the task.

        .. warning::

            Resizing images might cause a significant slow down in batch generation.

        :param name: Task name.
        :type name: str

        :param config: Dictionary of parameters (read from the configuration ``.yaml`` file).
        :type config: :py:class:`ptp.configuration.ConfigInterface`
        """

        # Call base class constructors.
        super(SimpleMolecules, self).__init__(name, SimpleMolecules, config)

        # Get default key mappings.
        self.key_images = self.stream_keys["images"]
        self.key_targets = self.stream_keys["targets"]
        # Stream returning targets as words.
        self.key_labels = self.stream_keys["labels"]

        # Add transformations depending on the resizing option.
        if 'resize_image' in self.config:
            # Check the desired size.
            if len(self.config['resize_image']) != 2:
                self.logger.error("'resize_image' field must contain 2 values: the desired height and width")
                exit(-1)

            # Output image dimensions.
            self.height = self.config['resize_image'][0]
            self.width = self.config['resize_image'][1]

            # Up-scale and transform to tensors.
            self.image_transforms = transforms.Compose([transforms.Resize((self.height, self.width)), transforms.ToTensor()])

            self.logger.warning('Upscaling the images to [{}, {}]. Slows down batch generation.'.format(
                self.width, self.height))

        else:
            # Default  settings.
            self.width = 875
            self.height = 875
            # Simply turn to tensor.
            self.image_transforms = transforms.Compose([transforms.ToTensor()])

        # Set global variables - all dimensions ASIDE OF BATCH.
        self.globals["num_classes"] = 10
        self.globals["image_width"] = self.width
        self.globals["image_height"] = self.height
        self.globals["image_depth"] = 1

        # Class names.
        labels = 'Zero One Two Three Four Five Six Seven Eight Nine'.split(' ')
        # Export to globals.
        word_to_ix = {labels[i]: i for i in range(10)}
        self.globals["label_word_mappings"] = word_to_ix
        # Reverse mapping - for labels.
        self.ix_to_word = {value: key for (key, value) in word_to_ix.items()}

        # Get the absolute path.
        self.data_folder = os.path.expanduser(self.config['data_folder'])

        # Get the split.
        split = get_value_from_dictionary('split', self.config, "training | validation | test".split(" | "))

        # Set split-dependent data.
        if split == 'training':
            # Training split folder and file with data question.
            data_file = os.path.join(self.data_folder, 'ChemDATA_A_Dist_Labels_Set0.tsv')
            self.image_folder = os.path.join(self.data_folder, "ChemDATA_A_Dist")

        elif split == 'validation':
            # Training split folder and file with data question.
            data_file = os.path.join(self.data_folder, 'ChemDATA_A_Dist_Labels_Set1.tsv')
            self.image_folder = os.path.join(self.data_folder, "ChemDATA_A_Dist")

        elif split == 'test':
            # Training split folder and file with data question.
            data_file = os.path.join(self.data_folder, 'ChemDATA_A_Dist_Labels_Set2.tsv')
            self.image_folder = os.path.join(self.data_folder, "ChemDATA_A_Dist")

        else: 
            raise ConfigurationError("Split {} not supported yet".format(split))

        # Load dataset.
        self.dataset = self.load_dataset(data_file)
        
        # Display exemplary sample.
        i = 0
        sample = self.dataset[i]

        self.logger.info("Exemplary sample {}:\n  image_ids: {}\n  class {}".format(
            i,
            sample[1],
            sample[0]
            ))


    def load_dataset(self, source_data_file):
        """
        Loads the dataset from source file

        :param source_data_file: csv file containing label-image filename pairs.

        """
        self.logger.info("Loading dataset from:\n {}".format(source_data_file))
        dataset = []

        with open(source_data_file, 'r') as f:
            self.logger.info("Loading samples from '{}'...".format(source_data_file))
            dataset = list(csv.reader(f, delimiter='\t'))

        self.logger.info("Loaded split consisting of {} samples".format(len(dataset)))
        return dataset


    def __len__(self):
        """
        Returns the "size" of the "task" (total number of samples).

        :return: The size of the task.
        """
        return len(self.dataset)


    def output_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of output data produced the component.

        :return: dictionary containing output data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {
            self.key_indices: DataDefinition([-1, 1], [list, int], "Batch of sample indices [BATCH_SIZE] x [1]"),
            self.key_images: DataDefinition([-1, 1, self.height, self.width], [torch.Tensor], "Batch of images [BATCH_SIZE x IMAGE_DEPTH x IMAGE_HEIGHT x IMAGE_WIDTH]"),
            self.key_targets: DataDefinition([-1], [torch.Tensor], "Batch of targets, each being a single index [BATCH_SIZE]"),
            self.key_labels: DataDefinition([-1, 1], [list, str], "Batch of targets, each being a single word [BATCH_SIZE] x [STRING]")
            }

    def get_image(self, img_id):
        """
        Function loads and returns image along with its size.
        Additionally, it performs all the required transformations.

        :param img_id: Identifier of the images.
        :param img_folder: Path to the image.

        :return: image (Tensor)
        """

        # Load the image.
        img = Image.open(os.path.join(self.image_folder, img_id + '.png')) #.convert('RGB')

        # Apply transformations.
        img = self.image_transforms(img)

        # Return image.
        return img

    def __getitem__(self, index):
        """
        Getter method to access the dataset and return a sample.

        :param index: index of the sample to return.
        :type index: int

        :return: ``DataStreams({'images','targets'})``, with:

            - images: Image, resized if ``self.resize`` is set,
            - targets: Index of the target class
        """
        # Get image and target.
        (label, img_id) = self.dataset[index]
  
        # Load the image.
        img = self.get_image(img_id)

        target = int(label)

        # Return data_streams.
        data_streams = self.create_data_streams(index)
        data_streams[self.key_images] = img
        data_streams[self.key_targets] = target
        data_streams[self.key_labels] = label
        return data_streams
