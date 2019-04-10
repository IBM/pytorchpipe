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

__author__ = "Tomasz Kornuta & Vincent Marois"


import torch

from ptp.components.models.model import Model
from ptp.data_types.data_definition import DataDefinition


class VariationalFlowLeNet5(Model):
    """
    A proof of concept of variational flow with LeNet-5 model for MNIST digits classification. 

    Uses masks (depending on targets here) to control the where each particular sample "flows through during backpropagation".

    For that purpose it has two flows, first for first subset of classes (0-2) and second for the remainder (3-9).

    """ 
    def __init__(self, name, config):
        """
        Initializes the model, retrieves key mappings, creates two flows.

        :param name: Name of the model (taken from the configuration file).

        :param config: Parameters read from configuration file.
        :type config: ``ptp.configuration.ConfigInterface``

        """
        super(VariationalFlowLeNet5, self).__init__(name, VariationalFlowLeNet5, config)

        # Get key mappings.
        self.key_inputs = self.stream_keys["inputs"]
        self.key_targets = self.stream_keys["targets"]

        self.key_flow1_predictions = self.stream_keys["flow1_predictions"]
        self.key_flow1_masks = self.stream_keys["flow1_masks"]
        self.key_flow2_predictions = self.stream_keys["flow2_predictions"]
        self.key_flow2_masks = self.stream_keys["floww_masks"]

        # Retrieve prediction sizes from globals.
        self.flow1_prediction_size = self.globals["flow1_prediction_size"]
        self.flow2_prediction_size = self.globals["flow2_prediction_size"]
        
        # Retrieve word mappings from globals.
        self.flow1_word_mappings = self.globals["flow1_word_mappings"]
        self.flow2_word_mappings = self.globals["flow2_word_mappings"]


        # Create flow 1.
        self.flow1_image_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 6, kernel_size=(5, 5)),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            torch.nn.Conv2d(6, 16, kernel_size=(5, 5)),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            torch.nn.Conv2d(16, 120, kernel_size=(5, 5)),
            torch.nn.ReLU(inplace=True)
            )

        self.flow1_classifier = torch.nn.Sequential(
            torch.nn.Linear(120, 84),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(84, 10), # FOR NOW
            torch.nn.LogSoftmax(dim=1)
            )

        # Create flow 2.
        self.flow2_image_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 6, kernel_size=(5, 5)),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            torch.nn.Conv2d(6, 16, kernel_size=(5, 5)),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            torch.nn.Conv2d(16, 120, kernel_size=(5, 5)),
            torch.nn.ReLU(inplace=True)
            )

        self.flow2_classifier = torch.nn.Sequential(
            torch.nn.Linear(120, 84),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(84, self.flow2_prediction_size),
            torch.nn.LogSoftmax(dim=1)
            )


    def input_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of input data that are required by the component.

        :return: dictionary containing input data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {
            self.key_inputs: DataDefinition([-1, 1, 32, 32], [torch.Tensor], "Batch of images [BATCH_SIZE x IMAGE_DEPTH x IMAGE_HEIGHT x IMAGE WIDTH]"),
            self.key_targets: DataDefinition([-1], [torch.Tensor], "Batch of targets [BATCH_SIZE]"),
            }


    def output_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of output data produced the component.

        :return: dictionary containing output data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return { # 10 for now!
            self.key_flow1_predictions: DataDefinition([-1, 10], [torch.Tensor], "Batch of flow 1predictions, each represented as probability distribution over classes [BATCH_SIZE x FLOW1_PREDICTION_SIZE]"),
            self.key_flow1_masks: DataDefinition([-1], [torch.Tensor], "Batch of masks for flow 1 [BATCH_SIZE]"),
            self.key_flow2_predictions: DataDefinition([-1, self.flow2_prediction_size], [torch.Tensor], "Batch of flow 2 predictions, each represented as probability distribution over classes [BATCH_SIZE x FLOW2_PREDICTION_SIZE]"),
            self.key_flow2_masks: DataDefinition([-1], [torch.Tensor], "Batch of masks for flow 2 [BATCH_SIZE]"),
            }


    def forward(self, data_dict):
        """
        Main forward pass of the model.
        In fact performs two passes, using masks generated on the fly using targets.

        :param data_dict: DataDict({'images',**}), where:

            - images: [batch_size, num_channels, width, height]

        :type data_dict: ``miprometheus.utils.DataDict``

        :return: Predictions [batch_size, num_classes]

        """
        # Produce masks.
        # TODO.

        # Get images.
        img = data_dict[self.key_inputs]

        # Pass inputs through flow 1.
        x1 = self.flow1_image_encoder(img)
        x1 = x1.view(-1, 120)
        x1 = self.flow1_classifier(x1)

        # Pass inputs through flow 2.
        x2 = self.flow2_image_encoder(img)
        x2 = x2.view(-1, 120)
        x2 = self.flow2_classifier(x2)


        # Add predictions to datadict.
        data_dict.extend({
            self.key_flow1_predictions: x1,
            self.key_flow2_predictions: x2,
            })
