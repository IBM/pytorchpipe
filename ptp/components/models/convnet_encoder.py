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

__author__ = "Younes Bouhadjar, Vincent Marois, Tomasz Kornuta"

import torch
import numpy as np
import torch.nn as nn

from ptp.components.models.model import Model
from ptp.data_types.data_definition import DataDefinition


class ConvNetEncoder(Model):
    """
    A simple image encoder consisting of 3 consecutive convolutional layers. \
    The parameters of input image (width, height and depth) are not hardcoded so the encoder can be adjusted for given application.
    """

    def __init__(self, name, params):
        """
        Constructor of the ``SimpleConvNet``. \

        The overall structure of this CNN is as follows:

            (Conv1 -> MaxPool1 -> ReLu) -> (Conv2 -> MaxPool2 -> ReLu) -> (Conv3 -> MaxPool3 -> ReLu)

        The parameters that the user can change are:

         - For Conv1, Conv2 & Conv3: number of output channels, kernel size, stride and padding.
         - For MaxPool1, MaxPool2 & MaxPool3: Kernel size


        .. note::

            We are using the default values of ``dilatation``, ``groups`` & ``bias`` for ``nn.Conv2D``.

            Similarly for the ``stride``, ``padding``, ``dilatation``, ``return_indices`` & ``ceil_mode`` of \
            ``nn.MaxPool2D``.


        :param name: Name of the model (tken from the configuration file).

        :param params: dict of parameters (read from configuration ``.yaml`` file).
        :type params: ``ptp.configuration.ParamInterface``

        """
        # Call base constructor.
        super(ConvNetEncoder, self).__init__(name, ConvNetEncoder, params)

        # Set key mappings.
        self.key_inputs = self.get_stream_key("inputs")
        self.key_feature_maps = self.get_stream_key("feature_maps")

        # Get input image information from the global parameters.
        self.input_width = self.global_value["input_width"]
        self.input_height = self.global_value["input_height"]        
        self.input_depth = self.global_value["input_depth"]

        # Retrieve the Conv1 parameters.
        self.out_channels_conv1 = params['conv1']['out_channels']
        self.kernel_size_conv1 = params['conv1']['kernel_size']
        self.stride_conv1 = params['conv1']['stride']
        self.padding_conv1 = params['conv1']['padding']

        # Retrieve the MaxPool1 parameter.
        self.kernel_size_maxpool1 = params['maxpool1']['kernel_size']

        # Retrieve the Conv2 parameters.
        self.out_channels_conv2 = params['conv2']['out_channels']
        self.kernel_size_conv2 = params['conv2']['kernel_size']
        self.stride_conv2 = params['conv2']['stride']
        self.padding_conv2 = params['conv2']['padding']

        # Retrieve the MaxPool2 parameter.
        self.kernel_size_maxpool2 = params['maxpool2']['kernel_size']

        # Retrieve the Conv3 parameters.
        self.out_channels_conv3 = params['conv3']['out_channels']
        self.kernel_size_conv3 = params['conv3']['kernel_size']
        self.stride_conv3 = params['conv3']['stride']
        self.padding_conv3 = params['conv3']['padding']

        # Retrieve the MaxPool3 parameter.
        self.kernel_size_maxpool3 = params['maxpool3']['kernel_size']

        # We can compute the spatial size of the output volume as a function of the input volume size (W),
        # the receptive field size of the Conv Layer neurons (F), the stride with which they are applied (S),
        # and the amount of zero padding used (P) on the border.
        # The corresponding equation is conv_size = ((W−F+2P)/S)+1.

        # doc for nn.Conv2D: https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d
        # doc for nn.MaxPool2D: https://pytorch.org/docs/stable/nn.html#torch.nn.MaxPool2d

        # ----------------------------------------------------
        # Conv1
        self.conv1 = nn.Conv2d(in_channels=self.input_depth,
                               out_channels=self.out_channels_conv1,
                               kernel_size=self.kernel_size_conv1,
                               stride=self.stride_conv1,
                               padding=self.padding_conv1,
                               dilation=1,
                               groups=1,
                               bias=True)

        self.width_features_conv1 = np.floor(
            ((self.input_width - self.kernel_size_conv1 + 2*self.padding_conv1) / self.stride_conv1) + 1)
        self.height_features_conv1 = np.floor(
            ((self.input_height - self.kernel_size_conv1 + 2*self.padding_conv1) / self.stride_conv1) + 1)

        # ----------------------------------------------------
        # MaxPool1
        self.maxpool1 = nn.MaxPool2d(kernel_size=self.kernel_size_maxpool1)

        self.width_features_maxpool1 = np.floor(
            ((self.width_features_conv1 - self.maxpool1.kernel_size + 2 * self.maxpool1.padding) / self.maxpool1.stride) + 1)

        self.height_features_maxpool1 = np.floor(
            ((self.height_features_conv1 - self.maxpool1.kernel_size + 2 * self.maxpool1.padding) / self.maxpool1.stride) + 1)

        # ----------------------------------------------------
        # Conv2
        self.conv2 = nn.Conv2d(in_channels=self.out_channels_conv1,
                               out_channels=self.out_channels_conv2,
                               kernel_size=self.kernel_size_conv2,
                               stride=self.stride_conv2,
                               padding=self.padding_conv2,
                               dilation=1,
                               groups=1,
                               bias=True)

        self.width_features_conv2 = np.floor(
            ((self.width_features_maxpool1 - self.kernel_size_conv2 + 2*self.padding_conv2) / self.stride_conv2) + 1)
        self.height_features_conv2 = np.floor(
            ((self.height_features_maxpool1 - self.kernel_size_conv2 + 2*self.padding_conv2) / self.stride_conv2) + 1)

        # ----------------------------------------------------
        # MaxPool2
        self.maxpool2 = nn.MaxPool2d(kernel_size=self.kernel_size_maxpool2)

        self.width_features_maxpool2 = np.floor(
            ((self.width_features_conv2 - self.maxpool2.kernel_size + 2 * self.maxpool2.padding) / self.maxpool2.stride) + 1)
        self.height_features_maxpool2 = np.floor(
            ((self.height_features_conv2 - self.maxpool2.kernel_size + 2 * self.maxpool2.padding) / self.maxpool2.stride) + 1)

        # ----------------------------------------------------
        # Conv3
        self.conv3 = nn.Conv2d(in_channels=self.out_channels_conv2,
                               out_channels=self.out_channels_conv3,
                               kernel_size=self.kernel_size_conv3,
                               stride=self.stride_conv3,
                               padding=self.padding_conv3,
                               dilation=1,
                               groups=1,
                               bias=True)

        self.width_features_conv3 = np.floor(
            ((self.width_features_maxpool2 - self.kernel_size_conv3 + 2*self.padding_conv3) / self.stride_conv3) + 1)
        self.height_features_conv3 = np.floor(
            ((self.height_features_maxpool2 - self.kernel_size_conv3 + 2*self.padding_conv3) / self.stride_conv3) + 1)

        # ----------------------------------------------------
        # MaxPool3
        self.maxpool3 = nn.MaxPool2d(kernel_size=self.kernel_size_maxpool3)

        self.width_features_maxpool3 = np.floor(
            ((self.width_features_conv3 - self.maxpool3.kernel_size + 2 * self.maxpool3.padding) / self.maxpool3.stride) + 1)

        self.height_features_maxpool3 = np.floor(
            ((self.height_features_conv3 - self.maxpool1.kernel_size + 2 * self.maxpool3.padding) / self.maxpool3.stride) + 1)

        # Set global variables: output dims
        self.global_value["feature_map_height"] = self.height_features_maxpool3
        self.global_value["feature_map_width"] = self.width_features_maxpool3
        self.global_value["feature_map_depth"] = self.out_channels_conv3
        
        # log some info.
        self.logger.info('Input: [-1, {}, {}, {}]'.format(self.input_depth, self.input_width, self.input_height))
        self.logger.info('Computed output shape of each layer:')
        self.logger.info('Conv1: [-1, {}, {}, {}]'.format(self.out_channels_conv1, self.width_features_conv1,
                                                      self.height_features_conv1))
        self.logger.info('MaxPool1: [-1, {}, {}, {}]'.format(self.out_channels_conv1, self.width_features_maxpool1,
                                                      self.height_features_maxpool1))
        self.logger.info('Conv2: [-1, {}, {}, {}]'.format(self.out_channels_conv2, self.width_features_conv2,
                                                      self.height_features_conv2))
        self.logger.info('MaxPool2: [-1, {}, {}, {}]'.format(self.out_channels_conv2, self.width_features_maxpool2,
                                                         self.height_features_maxpool2))
        self.logger.info('Conv3: [-1, {}, {}, {}]'.format(self.out_channels_conv3, self.width_features_conv3,
                                                      self.height_features_conv3))
        self.logger.info('MaxPool3: [-1, {}, {}, {}]'.format(self.out_channels_conv3, self.width_features_maxpool3,
                                                         self.height_features_maxpool3))



    def input_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of input data that are required by the component.

        :return: dictionary containing input data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {
            self.key_inputs: DataDefinition([-1, self.input_depth, self.input_height, self.input_width], [torch.Tensor], "Batch of images [BATCH_SIZE x IMAGE_DEPTH x IMAGE_HEIGHT x IMAGE WIDTH]"),
            }


    def output_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of output data produced the component.

        :return: dictionary containing output data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {
            self.key_feature_maps: DataDefinition([-1, self.out_channels_conv3, self.height_features_maxpool3, self.width_features_maxpool3], [torch.Tensor], "Batch of filter maps [BATCH_SIZE x DEPTH x HEIGHT x WIDTH]")
            }

    def forward(self, data_dict):
        """
        forward pass of the ``SimpleConvNet`` model.

        :param data_dict: DataDict({'inputs','outputs'}), where:

            - inputs: [batch_size, in_depth, in_height, in_width],
            - feature_maps: batch of feature maps [batch_size, out_depth, out_height, out_width]

        """
        # get images
        images = data_dict[self.key_inputs]

        # apply Convolutional layer 1
        out_conv1 = self.conv1(images)

        # apply max_pooling and relu
        out_maxpool1 = torch.nn.functional.relu(self.maxpool1(out_conv1))

        # apply Convolutional layer 2
        out_conv2 = self.conv2(out_maxpool1)

        # apply max_pooling and relu
        out_maxpool2 = torch.nn.functional.relu(self.maxpool2(out_conv2))

        # apply Convolutional layer 3
        out_conv3 = self.conv3(out_maxpool2)

        # apply max_pooling and relu
        out_maxpool3 = torch.nn.functional.relu(self.maxpool3(out_conv3))

        # Add output to datadict.
        data_dict.extend({self.key_feature_maps: out_maxpool3})
