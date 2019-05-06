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


class VQA_Attention(Model):
    """
    Element of one of the classical baselines for Visual Question Answering.
    Attention-weighted image maps are computed based on the question.
    The multi-modal data (question and attention-weighted image maps) are fused via concatenation and returned (for subsequent classification, done in a separate component e.g. ffn).

    On the basis of: Vahid Kazemi Ali Elqursh. "Show, Ask, Attend, and Answer: A Strong Baseline For Visual Question Answering" (2017).
    Code: https://github.com/Cyanogenoid/pytorch-vqa/blob/master/model.py
    """
    def __init__(self, name, config):
        """
        Initializes the model, creates the required layers.

        :param name: Name of the model (taken from the configuration file).

        :param config: Parameters read from configuration file.
        :type config: ``ptp.configuration.ConfigInterface``

        """
        super(VQA_Attention, self).__init__(name, VQA_Attention, config)

        # Get key mappings.
        self.key_feature_maps = self.stream_keys["feature_maps"]
        self.key_question_encodings = self.stream_keys["question_encodings"]
        self.key_outputs = self.stream_keys["outputs"]

        # Retrieve input/output sizes from globals.
        self.feature_maps_height = self.globals["feature_maps_height"]
        self.feature_maps_width = self.globals["feature_maps_width"]
        self.feature_maps_depth = self.globals["feature_maps_depth"]
        self.question_encoding_size = self.globals["question_encoding_size"]

        # Get size of latent space and number of heads from config.
        self.latent_size = self.config["latent_size"]
        self.num_attention_heads = self.config["num_attention_heads"]

        # Output new attention weighted image encoding only, or both image and question image_encodings
        self.output_mode = self.config["output_mode"]

        # Output feature size
        if(self.output_mode == 'Image'):
            self.output_size = self.feature_maps_depth*self.num_attention_heads
        elif(self.output_mode == 'None'):
            self.output_size = self.feature_maps_depth*self.num_attention_heads + self.question_encoding_size

        # Export to globals.
        self.globals["output_size"] = self.output_size

        # Map image and question encodings to a common latent space of dimension 'latent_size'.
        self.image_encodings_conv = torch.nn.Conv2d(self.feature_maps_depth, self.latent_size, 1, bias=False)
        self.question_encodings_ff = torch.nn.Linear(self.question_encoding_size, self.latent_size)

        # Scalar-dot product attention function is implemented as a Conv operation
        self.attention_conv = torch.nn.Conv2d(self.latent_size, self.num_attention_heads, 1)

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
            self.key_feature_maps: DataDefinition([-1, self.feature_maps_depth, self.feature_maps_height, self.feature_maps_width], [torch.Tensor], "Batch of feature maps [BATCH_SIZE x FEAT_DEPTH x FEAT_HEIGHT x FEAT_WIDTH]"),
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
        enc_img = data_dict[self.key_feature_maps] #[48, 2048, 7, 7]
        enc_q = data_dict[self.key_question_encodings] #[48, 100]
        # print("im_enc", enc_img.shape)
        # print("enc_q", enc_q.shape)

        # L2 norm of image encoding
        enc_img = enc_img / (enc_img.norm(p=2, dim=1, keepdim=True).expand_as(enc_img) + 1e-8)

        # Compute attention maps for image using questions
        latent_img = self.image_encodings_conv(self.dropout(enc_img)) # [48, 100, 7, 7]
        # print("latent_im", latent_img.shape)
        latent_q =  self.question_encodings_ff(self.dropout(enc_q)) # [48, 100]
        # print("latent_q", latent_q.shape)
        latent_q_tile = tile_2d_over_nd(latent_q, latent_img) # [48, 100, 7, 7]
        # print("latent_q_tile", latent_q_tile.shape)
        attention = self.activation(latent_img + latent_q_tile) #
        # print("attention", attention.shape)
        attention = self.attention_conv(self.dropout(attention)) # [48, 2, 7, 7]
        # print("attention", attention.shape)

        # Apply attention to image encoding
        attention_enc_img = apply_attention(enc_img, attention) # [48, 2048, 7, 7], [48, 2, 7, 7]
        # print("attention im", attention_enc_img.shape)

        if(self.output_mode == 'Image'):
        # Output attention-weighted image encodings
            outputs = attention_enc_img
        elif(self.output_mode == 'None'):
            # Fusion -- Concatenate attention-weighted image encodings and question encodings.
            outputs = torch.cat([attention_enc_img, latent_q], dim=1)
        # print("outputs", outputs.shape)
        # Add predictions to datadict.
        data_dict.extend({self.key_outputs: outputs})


def tile_2d_over_nd(feature_vector, feature_map):
    """ Repeat the same feature vector over all spatial positions of a given feature map.
        The feature vector should have the same batch size and number of features as the feature map.
    """
    n, c = feature_vector.size()
    spatial_size = feature_map.dim() - 2
    tiled = feature_vector.view(n, c, *([1] * spatial_size)).expand_as(feature_map)
    return tiled


def apply_attention(input, attention):
    """ Apply any number of attention maps over the input. """
    n, c = input.size()[:2]
    glimpses = attention.size(1) # glimpses is equivalent to multiple heads in attention

    # flatten the spatial dims into the third dim, since we don't need to care about how they are arranged
    input = input.view(n, 1, c, -1) # [n, 1, c, s] [batch, 1, channels, height*width] [48, 1, 2048, 7*7]
    attention = attention.view(n, glimpses, -1) # [48, 2, 7*7]
    attention = torch.nn.functional.softmax(attention, dim=-1).unsqueeze(2) # [n, g, 1, s] [batch, multi_head, 1, height*width] [48, 2, 1, 7*7]
    weighted = attention * input # [n, g, c, s] [48, 2, 2048, 7*7]
    weighted_mean = weighted.sum(dim=-1) # [n, g, c] [48, 2, 2048]
    return weighted_mean.view(n, -1) # [48, 4096]
