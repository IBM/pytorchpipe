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
import torchvision.models as models

from ptp.components.models.model import Model
from ptp.data_types.data_definition import DataDefinition


class TorchVisionWrapper(Model):
    """
    Class
    """ 
    def __init__(self, name, config):
        """
        Initializes the ``LeNet5`` model, creates the required layers.

        :param name: Name of the model (taken from the configuration file).

        :param config: Parameters read from configuration file.
        :type config: ``ptp.configuration.ConfigInterface``

        """
        super(TorchVisionWrapper, self).__init__(name, TorchVisionWrapper, config)

        # Get key mappings.
        self.key_inputs = self.stream_keys["inputs"]
        self.key_predictions = self.stream_keys["predictions"]

        # Retrieve prediction size from globals.
        self.prediction_size = self.globals["prediction_size"]

        # Get VGG16
        self.model = models.vgg16(pretrained=True)
        # "Replace" last layer.
        self.model.classifier._modules['6'] = torch.nn.Linear(4096, self.prediction_size)


    def input_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of input data that are required by the component.

        :return: dictionary containing input data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {
            self.key_inputs: DataDefinition([-1, 3, 224, 224], [torch.Tensor], "Batch of images [BATCH_SIZE x IMAGE_DEPTH x IMAGE_HEIGHT x IMAGE WIDTH]"),
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
        Main forward pass of the model.

        :param data_dict: DataDict({'inputs', ....}), where:

            - inputs: expected stream containing images [BATCH_SIZE x IMAGE_DEPTH x IMAGE_HEIGHT x IMAGE WIDTH]
            - outpus: added stream containing predictions [BATCH_SIZE x PREDICTION_SIZE]

        :type data_dict: ``ptp.data_types.DataDict``

        """

        # Unpack DataDict.
        img = data_dict[self.key_inputs]

        predictions = self.model(img)

        # Add predictions to datadict.
        data_dict.extend({self.key_predictions: predictions})
