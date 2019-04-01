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

__author__ = "Tomasz Kornuta, Younes Bouhadjar, Vincent Marois"

import os
import torch
from torchvision import datasets, transforms

from ptp.components.problems.image_to_class.image_to_class_problem import ImageToClassProblem
from ptp.data_types.data_definition import DataDefinition

class MNIST(ImageToClassProblem):
    """
    Classic MNIST digit classification problem.

    Please see reference here: http://yann.lecun.com/exdb/mnist/

    .. warning::

        The dataset is not originally split into a training set, validation set and test set; only\
        training and test set. 
        
        In order to split training into training and validation sets please set distinctive index ranges in configuration file.

    """

    def __init__(self, name, config):
        """
        Initializes the MNIST problem.

        .. warning::

            Resizing images might cause a significant slow down in batch generation.

        :param name: Problem name.
        :type name: str

        :param config: Dictionary of parameters (read from the configuration ``.yaml`` file).
        :type config: :py:class:`ptp.configuration.ConfigInterface`
        """

        # Call base class constructors.
        super(MNIST, self).__init__(name, MNIST, config)

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
            # Default MNIST settings.
            self.width = 28
            self.height = 28
            # Simply turn to tensor.
            transform = transforms.Compose([transforms.ToTensor()])

        # load the dataset
        self.dataset = datasets.MNIST(root=data_folder, train=self.use_train_data, download=True,
                                      transform=transform)

        # Set global variables - all dimensions ASIDE OF BATCH.
        self.globals["num_classes"] = 10
        self.globals["image_width"] = self.width
        self.globals["image_height"] = self.height
        self.globals["image_depth"] = 1

        # Class names.
        #self.labels = 'Zero One Two Three Four Five Six Seven Eight Nine'.split(' ')

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
            self.key_inputs: DataDefinition([-1, 1, self.height, self.width], [torch.Tensor], "Batch of images [BATCH_SIZE x IMAGE_DEPTH x IMAGE_HEIGHT x IMAGE_WIDTH]"),
            self.key_targets: DataDefinition([-1], [torch.Tensor], "Batch of targets, each being a single index [BATCH_SIZE]")
            }


    def __getitem__(self, index):
        """
        Getter method to access the dataset and return a sample.

        :param index: index of the sample to return.
        :type index: int

        :return: ``DataDict({'images','targets'})``, with:

            - images: Image, resized if ``self.resize`` is set,
            - targets: Index of the target class
        """
        # Get image and target.
        img, target = self.dataset.__getitem__(index)
  
        # Return data_dict.
        data_dict = self.create_data_dict(index)
        data_dict[self.key_inputs] = img
        data_dict[self.key_targets] = target
        return data_dict
