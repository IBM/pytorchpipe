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

__author__ = "Deepta Rajan"


import torch

from ptp.components.models.model import Model
from ptp.data_types.data_definition import DataDefinition
import torch.nn.functional as F

class MultimodalFactorizedBilinearPooling(Model):
    """
    Element of one of the classical baselines for Visual Question Answering.
    The multi-modal data is fused via sum-pooling of the element-wise multiplied high-dimensional representations and returned (for subsequent classification, done in a separate component e.g. ffn).

    On the basis of: Zhou Yu, Jun Yu. "Beyond Bilinear: Generalized Multi-modal Factorized High-order Pooling for Visual Question Answering" (2015).
    Code: https://github.com/Cadene/block.bootstrap.pytorch/blob/master/block/models/networks/fusions/fusions.py
    """
    def __init__(self, name, config):
        """
        Initializes the model, creates the required layers.

        :param name: Name of the model (taken from the configuration file).

        :param config: Parameters read from configuration file.
        :type config: ``ptp.configuration.ConfigInterface``

        """
        super(MultimodalFactorizedBilinearPooling, self).__init__(name, MultimodalFactorizedBilinearPooling, config)

        # Get key mappings.
        self.key_image_encodings = self.stream_keys["image_encodings"]
        self.key_question_encodings = self.stream_keys["question_encodings"]
        self.key_outputs = self.stream_keys["outputs"]

        # Retrieve input/output sizes from globals.
        self.image_encoding_size = self.globals["image_encoding_size"]
        self.question_encoding_size = self.globals["question_encoding_size"]

        # Get size of latent space and number of heads from config.
        self.latent_size = self.config["latent_size"]
        self.factor = self.config["pool_factor"]
        # Output feature size
        self.output_size = self.latent_size

        # Export to globals.
        self.globals["output_size"] = self.output_size

        # Map image and question encodings to a common latent space of dimension 'latent_size'.
        self.image_encodings_ff = torch.nn.Linear(self.image_encoding_size, self.latent_size*self.factor)
        self.question_encodings_ff = torch.nn.Linear(self.question_encoding_size, self.latent_size*self.factor)

        # Create activation layer.
        self.activation = torch.nn.ReLU()

        # Retrieve dropout rate value - if set, will put dropout between every layer.
        dropout_rate = self.config["dropout_rate"]

        # Create dropout layer.
        self.dropout = torch.nn.Dropout(dropout_rate)


    def input_data_definitions(self):
        """
        Function returns a dictionary with definitions of input data that are required by the component.

        :return: dictionary containing input data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {
            self.key_image_encodings: DataDefinition([-1, self.image_encoding_size], [torch.Tensor], "Batch of encoded images [BATCH_SIZE x IMAGE_ENCODING_SIZE]"),
            self.key_question_encodings: DataDefinition([-1, self.question_encoding_size], [torch.Tensor], "Batch of encoded questions [BATCH_SIZE x QUESTION_ENCODING_SIZE]"),
            }


    def output_data_definitions(self):
        """
        Function returns a dictionary with definitions of output data produced the component.

        :return: dictionary containing output data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {
            self.key_outputs: DataDefinition([-1, self.output_size], [torch.Tensor], "Batch of outputs [BATCH_SIZE x OUTPUT_SIZE]")
            }

    def forward(self, data_dict):
        """
        Main forward pass of the model.

        :param data_dict: DataDict({'images',**})
        :type data_dict: ``ptp.dadatypes.DataDict``
        """

        # Unpack DataDict.
        enc_img = data_dict[self.key_image_encodings] #[48, 2048]
        enc_q = data_dict[self.key_question_encodings] #[48, 100]

        # Map image and question encodings to high-dimensional space using FF
        latent_img = self.dropout(self.image_encodings_ff(enc_img)) # [48, 512]
        latent_q =  self.dropout(self.question_encodings_ff(enc_q)) # [48, 512]

        # Element-wise mutliplication of image and question encodings
        enc_z = latent_img * latent_q # [48, 512]
        # Dropout regularization
        enc_z = self.dropout(enc_z)
        enc_z = enc_z.view(enc_z.size(0), self.latent_size, self.factor) # [48, 256, 2]
        # Sum pooling
        enc_z = enc_z.sum(2) # [48, 256]
        # Power and L2 normalization
        enc_z = torch.sqrt(self.activation(enc_z)) - torch.sqrt(self.activation(-enc_z))
        outputs = F.normalize(enc_z, p=2, dim=1) # [48, 256]

        # Add predictions to datadict.
        data_dict.extend({self.key_outputs: outputs})
