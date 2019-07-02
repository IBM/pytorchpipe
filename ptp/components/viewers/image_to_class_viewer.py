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

import numpy as np
import torch
import matplotlib.pyplot as plt

from ptp.components.component import Component
from ptp.data_types.data_definition import DataDefinition


class ImageToClassViewer(Component):
    """
    Utility for displaying contents image along with label and prediction (a single sample from the batch).
    """

    def __init__(self, name, config):
        """
        Initializes loss object.

        :param name: Loss name.
        :type name: str

        :param config: Dictionary of parameters (read from the configuration ``.yaml`` file).
        :type config: :py:class:`ptp.configuration.ConfigInterface`

        """
        # Call constructors of parent classes.
        Component.__init__(self, name, ImageToClassViewer, config)

        # Get default key mappings.
        self.key_indices = self.stream_keys["indices"]
        self.key_images = self.stream_keys["images"]
        self.key_labels = self.stream_keys["labels"]
        self.key_answers = self.stream_keys["answers"]

        # Get sample number.
        self.sample_number = self.config["sample_number"]
        

    def input_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of input data that are required by the component.

        :return: dictionary containing input data definitions (each of type :py:class:`ptp.data_types.DataDefinition`).
        """
        return {
            self.key_indices: DataDefinition([-1, 1], [list, int], "Batch of sample indices [BATCH_SIZE] x [1]"),
            self.key_images: DataDefinition([-1, -1, -1, -1], [torch.Tensor], "Batch of images [BATCH_SIZE x IMAGE_DEPTH x IMAGE_HEIGHT x IMAGE_WIDTH]"),
            self.key_labels: DataDefinition([-1, 1], [list, str], "Batch of target labels, each being a single word [BATCH_SIZE] x [STRING]"),
            self.key_answers: DataDefinition([-1, 1], [list, str], "Batch of predicted labels, each being a single word [BATCH_SIZE] x [STRING]")
            }

    def output_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of output data produced the component.

        :return: dictionary containing output data definitions (each of type :py:class:`ptp.data_types.DataDefinition`).
        """
        return {
            }

    def __call__(self, data_streams):
        """
        Shows a sample from the batch.

        :param data_streams: :py:class:`ptp.utils.DataStreams` object.

        """
        # Use worker interval.
        if self.app_state.episode % self.app_state.args.logging_interval == 0:

            # Get inputs
            indices = data_streams[self.key_indices]
            images = data_streams[self.key_images]
            labels = data_streams[self.key_labels]
            answers = data_streams[self.key_answers]

            # Get sample number.
            if self.sample_number == -1:
                # Random.
                sample_number = np.random.randint(0, len(images))
            else:
                sample_number = self.sample_number

            # Get "sample".
            image = images[sample_number].cpu().data.numpy()
            label = labels[sample_number]
            answer = answers[sample_number]

            # Reshape image.
            if image.shape[0] == 1:
                # This is a single channel image - get rid of this dimension
                image = np.squeeze(image, axis=0)
            else:
                # More channels - move channels to axis2, according to matplotilb documentation.
                # (X : array_like, shape (n, m) or (n, m, 3) or (n, m, 4))
                image = image.transpose(1, 2, 0)

            # Show data.
            plt.title('Sample: {} (index: {})\nPrediction: {}  | Target: {}'.format(sample_number, indices[sample_number], answer, label))
            plt.imshow(image, interpolation='nearest', aspect='auto')

            # Plot!
            plt.show()


