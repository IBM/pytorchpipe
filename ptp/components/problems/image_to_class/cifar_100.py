#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) IBM Corporation 2018
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
import torch
from torchvision import datasets, transforms

from ptp.components.problems.problem import Problem
from ptp.data_types.data_definition import DataDefinition

class CIFAR100(Problem):
    """
    Classic CIFAR-100 image classification problem.

    Reference page: http://www.cs.toronto.edu/~kriz/cifar.html


    """

    def __init__(self, name, config):
        """
        Initializes the problem.

        .. warning::

            Resizing images might cause a significant slow down in batch generation.

        :param name: Problem name.
        :type name: str

        :param config: Dictionary of parameters (read from the configuration ``.yaml`` file).
        :type config: :py:class:`ptp.configuration.ConfigInterface`
        """

        # Call base class constructors.
        super(CIFAR100, self).__init__(name, CIFAR100, config)

        # Get default key mappings.
        self.key_inputs = self.stream_keys["inputs"]
        # Targets.
        self.key_coarse_targets = self.stream_keys["coarse_targets"]
        self.key_fine_targets = self.stream_keys["fine_targets"]
        # Streams returning targets as words (labels).
        self.key_coarse_labels = self.stream_keys["coarse_labels"]
        self.key_fine_labels = self.stream_keys["fine_labels"]

        # Get absolute path.
        data_folder = os.path.expanduser(self.config['data_folder'])

        # Retrieve parameters from the dictionary.
        self.use_train_data = self.config['use_train_data']

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
            transform = transforms.Compose([transforms.Resize((self.height, self.width)), transforms.ToTensor()])

            self.logger.warning('Upscaling the images to [{}, {}]. Slows down batch generation.'.format(
                self.width, self.height))

        else:
            # Default CIFAR-100 settings.
            self.width = 32
            self.height = 32
            # Simply turn to tensor.
            transform = transforms.Compose([transforms.ToTensor()])

        # Load the dataset. (PROBLEM WITH COARSE-TO-FINE LABEL MAPPING!)
        self.dataset = datasets.CIFAR100(root=data_folder, train=self.use_train_data, download=True, transform=transform)

        # Process labels.
        all_labels = {"aquatic_mammals": "beaver, dolphin, otter, seal, whale".split(", "),
            "fish": "aquarium_fish, flatfish, ray, shark, trout".split(", "),
            "flowers": "orchid, poppy, rose, sunflower, tulip".split(", "),
            "food_containers": "bottle, bowl, can, cup, plate".split(", "),
            "fruit_and_vegetables": "apple, mushroom, orange, pear, sweet_pepper".split(", "),
            "household_electrical_devices": "clock, keyboard, lamp, telephone, television".split(", "),
            "household_furniture": "bed, chair, couch, table, wardrobe".split(", "),
            "insects": "bee, beetle, butterfly, caterpillar, cockroach".split(", "),
            "large_carnivores": "bear, leopard, lion, tiger, wolf".split(", "),
            "large_man-made_outdoor_things": "bridge, castle, house, road, skyscraper".split(", "),
            "large_natural_outdoor_scenes": "cloud, forest, mountain, plain, sea".split(", "),
            "large_omnivores_and_herbivores": "camel, cattle, chimpanzee, elephant, kangaroo".split(", "),
            "medium-sized_mammals": "fox, porcupine, possum, raccoon, skunk".split(", "),
            "non-insect_invertebrates": "crab, lobster, snail, spider, worm".split(", "),
            "people": "baby, boy, girl, man, woman".split(", "),
            "reptiles": "crocodile, dinosaur, lizard, snake, turtle".split(", "),
            "small_mammals": "hamster, mouse, rabbit, shrew, squirrel".split(", "),
            "trees": "maple_tree, oak_tree, palm_tree, pine_tree, willow_tree".split(", "),
            "vehicles_1": "bicycle, bus, motorcycle, pickup_truck, train".split(", "),
            "vehicles_2": "lawn_mower, rocket, streetcar, tank, tractor".split(", ")}

        coarse_word_to_ix = {}
        fine_to_coarse_mapping = {}
        fine_labels = []
        for coarse_id, (key, values) in enumerate(all_labels.items()):
            # Add mapping from coarse category name to coarse id.
            coarse_word_to_ix[key] = coarse_id
            # Add mappings from fine category names to coarse id.
            for value in values:
                fine_to_coarse_mapping[value] = coarse_id
            # Add values to list of fine labels.
            fine_labels.extend(values)

        # Sort fine labels.
        fine_labels = sorted(fine_labels)

        # Generate fine word mappings.
        fine_word_to_ix = {fine_labels[i]: i for i in range(len(fine_labels))}
        # Export fine word mappings to globals.
        self.globals["fine_label_word_mappings"] = fine_word_to_ix
        # Reverse mapping - for labels.
        self.fine_ix_to_word = {value: key for (key, value) in fine_word_to_ix.items()}

        # Export coarse word mappings to globals.
        self.globals["coarse_label_word_mappings"] = coarse_word_to_ix
        # Reverse mapping - for labels.
        self.coarse_ix_to_word = {value: key for (key, value) in coarse_word_to_ix.items()}

        # Create fine to coarse id mapping.
        self.fine_to_coarse_id_mapping = {}
        for fine_label, fine_id in fine_word_to_ix.items():
            self.fine_to_coarse_id_mapping[fine_id] = fine_to_coarse_mapping[fine_label]
            #print(" {} ({}) : {} ".format(fine_label, fine_id, self.coarse_ix_to_word[fine_to_coarse_mapping[fine_label]]))

        # Set global variables - all dimensions ASIDE OF BATCH.
        self.globals["num_coarse_classes"] = len(coarse_word_to_ix)
        self.globals["num_fine_classes"] = len(fine_labels)
        self.globals["image_width"] = self.width
        self.globals["image_height"] = self.height
        self.globals["image_depth"] = 3

    def __len__(self):
        """
        Returns the "size" of the "problem" (total number of samples).

        :return: The size of the problem.
        """
        return len(self.dataset)

    def output_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of output data produced the component.

        :return: dictionary containing output data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {
            self.key_indices: DataDefinition([-1, 1], [list, int], "Batch of sample indices [BATCH_SIZE] x [1]"),
            self.key_inputs: DataDefinition([-1, 3, self.height, self.width], [torch.Tensor], "Batch of images [BATCH_SIZE x IMAGE_DEPTH x IMAGE_HEIGHT x IMAGE_WIDTH]"),
            self.key_coarse_targets: DataDefinition([-1], [torch.Tensor], "Batch of coarse targets, each being a single index [BATCH_SIZE]"),
            self.key_coarse_labels: DataDefinition([-1, 1], [list, str], "Batch of coarse labels, each being a single word [BATCH_SIZE] x [STRING]"),
            self.key_fine_targets: DataDefinition([-1], [torch.Tensor], "Batch of fine targets, each being a single index [BATCH_SIZE]"),
            self.key_fine_labels: DataDefinition([-1, 1], [list, str], "Batch of fine labels, each being a single word [BATCH_SIZE] x [STRING]")
            }


    def __getitem__(self, index):
        """
        Getter method to access the dataset and return a sample.

        :param index: index of the sample to return.
        :type index: int

        :return: ``DataStreams({'images','targets'})``, with:

            - images: Image, resized if ``self.resize`` is set,
            - targets: Index of the target class
        """
        # Get image and fine label id.
        image, fine_target = self.dataset.__getitem__(index)

        # Return data_streams.
        data_streams = self.create_data_streams(index)
        data_streams[self.key_inputs] = image
        # Targets.
        data_streams[self.key_coarse_targets] = self.fine_to_coarse_id_mapping[fine_target]
        data_streams[self.key_fine_targets] = fine_target
        # Labels.
        data_streams[self.key_coarse_labels] = self.coarse_ix_to_word[self.fine_to_coarse_id_mapping[fine_target]]
        data_streams[self.key_fine_labels] = self.fine_ix_to_word[fine_target]

        #print(data_streams)
        return data_streams
