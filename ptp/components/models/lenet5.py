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
import torch.nn.functional as F

from ptp.components.models.model import Model
from ptp.data_types.data_definition import DataDefinition


class LeNet5(Model):
    """
    A classical LeNet-5 model for MNIST digits classification. 
    """ 
    def __init__(self, name, params):
        """
        Initializes the ``LeNet5`` model, creates the required layers.

        :param name: Name of the model (taken from the configuration file).

        :param params: Parameters read from configuration file.
        :type params: ``miprometheus.utils.ParamInterface``

        """
        super(LeNet5, self).__init__(name, LeNet5, params)

        # Set key mappings.
        self.key_inputs = self.mapkey("inputs")
        self.key_predictions = self.mapkey("predictions")

        # Retrieve prediction size from globals.
        self.key_prediction_size = self.mapkey("prediction_size")
        self.prediction_size = self.app_state[self.key_prediction_size]

        # Create the LeNet-5 layers.
        self.conv1 = torch.nn.Conv2d(1, 6, kernel_size=(5, 5))
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = torch.nn.Conv2d(6, 16, kernel_size=(5, 5))
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = torch.nn.Conv2d(16, 120, kernel_size=(5, 5))
        self.linear1 = torch.nn.Linear(120, 84)
        self.linear2 = torch.nn.Linear(84, self.prediction_size)

    def input_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of input data that are required by the component.

        :return: dictionary containing input data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {
            self.key_inputs: DataDefinition([-1, 1, 32, 32], [torch.Tensor], "Batch of images [BATCH_SIZE x IMAGE_DEPTH x IMAGE_HEIGHT x IMAGE WIDTH]"),
            }


    def output_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of output data produced the component.

        :return: dictionary containing output data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {
            self.key_predictions: DataDefinition([-1, self.prediction_size], [torch.Tensor], "Batch of predictions, each represented as probability distribution over classes [BATCH_SIZE x PREDICTION_SIZE]")
            }

    def forward(self, data_dict):
        """
        Main forward pass of the ``LeNet5`` model.

        :param data_dict: DataDict({'images',**}), where:

            - images: [batch_size, num_channels, width, height]

        :type data_dict: ``miprometheus.utils.DataDict``

        :return: Predictions [batch_size, num_classes]

        """
        # Add noise to weights
        #for _, param in self.named_parameters():
        #    if param.requires_grad:
        #        #print (name, param.data)
        #        #noise = -torch.randn(param.data.shape)*0.3
        #        noise = 0.3
        #        param.data = param.data * (1 + noise)
        #        #print (name, param.data)


        # Unpack DataDict.
        img = data_dict[self.key_inputs]

        # Pass inputs through layers.
        x = self.conv1(img)
        x = torch.nn.functional.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = torch.nn.functional.relu(x)
        x = x.view(-1, 120)
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.linear2(x)
        # Log softmax.
        predictions = F.log_softmax(x, dim=1)
        # Add predictions to datadict.
        data_dict.extend({self.key_predictions: predictions})
