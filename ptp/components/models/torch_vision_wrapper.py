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


import torch
import torchvision.models as models

from ptp.configuration.config_parsing import get_value_from_dictionary
from ptp.configuration.configuration_error import ConfigurationError
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
        self.key_outputs = self.stream_keys["outputs"]

        # Get operation modes.
        self.return_feature_maps = self.config["return_feature_maps"]
        pretrained = self.config["pretrained"]

        # Get model type from configuration.
        self.model_type = get_value_from_dictionary("model_type", self.config, "vgg16 | sensenet121 | resnet152 | resnet50".split(" | "))

        if(self.model_type == 'vgg16'):
            # Get VGG16
            self.model = models.vgg16(pretrained=pretrained)

            if self.return_feature_maps:
                # Use only the "feature encoder".
                self.model = self.model.features

                # Height of the returned features tensor (SET)
                self.feature_maps_height = 7
                self.globals["feature_maps_height"] = self.feature_maps_height
                # Width of the returned features tensor (SET)
                self.feature_maps_width = 7
                self.globals["feature_maps_width"] = self.feature_maps_width
                # Depth of the returned features tensor (SET)
                self.feature_maps_depth = 512
                self.globals["feature_maps_depth"] = self.feature_maps_depth

            else:
                # Use the whole model, but cut/reshape only the last layer.
                self.output_size = self.globals["output_size"]
                # "Replace" the last layer.
                self.model.classifier._modules['6'] = torch.nn.Linear(4096, self.output_size)

        elif(self.model_type == 'densenet121'):
            # Get densenet121
            self.model = models.densenet121(pretrained=pretrained)

            if self.return_feature_maps:
                raise ConfigurationError("'densenet121' doesn't support 'return_feature_maps' mode (yet)")

            # Use the whole model, but cut/reshape only the last layer.
            self.output_size = self.globals["output_size"]
            self.model.classifier = torch.nn.Linear(1024, self.output_size)


        elif(self.model_type == 'resnet152'):
            # Get resnet152
            self.model = models.resnet152(pretrained=pretrained)

            if self.return_feature_maps:
                # Get all modules exluding last (avgpool) and (fc)
                modules=list(self.model.children())[:-2]
                self.model=torch.nn.Sequential(*modules)                

                # Height of the returned features tensor (SET)
                self.feature_maps_height = 7
                self.globals["feature_maps_height"] = self.feature_maps_height
                # Width of the returned features tensor (SET)
                self.feature_maps_width = 7
                self.globals["feature_maps_width"] = self.feature_maps_width
                # Depth of the returned features tensor (SET)
                self.feature_maps_depth = 2048
                self.globals["feature_maps_depth"] = self.feature_maps_depth

            else:
                # Use the whole model, but cut/reshape only the last layer.
                self.output_size = self.globals["output_size"]
                self.model.fc = torch.nn.Linear(2048, self.output_size)

        elif(self.model_type == 'resnet50'):
            # Get resnet50
            self.model = models.resnet50(pretrained=pretrained)

            if self.return_feature_maps:
                # Get all modules exluding last (avgpool) and (fc)
                modules=list(self.model.children())[:-2]
                self.model=torch.nn.Sequential(*modules)                

                # Height of the returned features tensor (SET)
                self.feature_maps_height = 7
                self.globals["feature_maps_height"] = self.feature_maps_height
                # Width of the returned features tensor (SET)
                self.feature_maps_width = 7
                self.globals["feature_maps_width"] = self.feature_maps_width
                # Depth of the returned features tensor (SET)
                self.feature_maps_depth = 2048
                self.globals["feature_maps_depth"] = self.feature_maps_depth

            else:
                # Use the whole model, but cut/reshape only the last layer.
                self.output_size = self.globals["output_size"]
                self.model.fc = torch.nn.Linear(2048, self.output_size)


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
        if self.return_feature_maps:
            return {
                self.key_outputs: DataDefinition([-1, self.feature_maps_depth, self.feature_maps_height, self.feature_maps_width], [torch.Tensor], "Batch of feature maps [BATCH_SIZE x FEAT_DEPTH x FEAT_HEIGHT x FEAT_WIDTH]")
                }
        else:
            return {
                self.key_outputs: DataDefinition([-1, self.output_size], [torch.Tensor], "Batch of outputs, each represented as probability distribution over classes [BATCH_SIZE x PREDICTION_SIZE]")
                }

    def forward(self, data_dict):
        """
        Main forward pass of the model.

        :param data_dict: DataDict({'inputs', ....}), where:

            - inputs: expected stream containing images [BATCH_SIZE x IMAGE_DEPTH x IMAGE_HEIGHT x IMAGE WIDTH]
            - outpus: added stream containing outputs [BATCH_SIZE x PREDICTION_SIZE]

        :type data_dict: ``ptp.data_types.DataDict``

        """
        # Unpack DataDict.
        img = data_dict[self.key_inputs]

        print("{}: input shape: {}, device: {}\n".format(self.name, img.shape, img.device))

        outputs = self.model(img)

        # Add outputs to datadict.
        data_dict.extend({self.key_outputs: outputs})
