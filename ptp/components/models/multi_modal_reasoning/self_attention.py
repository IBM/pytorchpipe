#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) IBM Corporation 2019
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


class SelfAttention(Model):
    """
    Element of one of the classical baselines for Visual Question Answering.
    Attention within an image or text is computed.
    The attention weighted data (question or image) is returned (for subsequent classification, done in a separate component e.g. ffn).
    Currently only supports self-attention on text data

    On the basis of: Vaswani et. al Attention is all you need (2017)

    """
    def __init__(self, name, config):
        """
        Initializes the model, creates the required layers.

        :param name: Name of the model (taken from the configuration file).

        :param config: Parameters read from configuration file.
        :type config: ``ptp.configuration.ConfigInterface``

        """
        super(SelfAttention, self).__init__(name, SelfAttention, config)

        # Get key mappings.
        self.key_question_encodings = self.stream_keys["question_encodings"]
        self.key_outputs = self.stream_keys["outputs"]

        # Retrieve input/output sizes from globals.
        self.question_encoding_size = self.globals["question_encoding_size"]

        # Get size of latent space and number of heads from config.
        self.latent_size = self.config["latent_size"]
        self.num_attention_heads = self.config["num_attention_heads"]

        # Output feature size
        self.output_size = self.question_encoding_size*self.num_attention_heads
        # Create activation layer.
        self.activation = torch.nn.ReLU()

        # Create FF layers
        self.W1 = torch.nn.Linear(self.question_encoding_size, self.latent_size)
        self.W2 = torch.nn.Linear(self.latent_size, self.num_attention_heads)


    def input_data_definitions(self):
        """
        Function returns a dictionary with definitions of input data that are required by the component.

        :return: dictionary containing input data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {
            self.key_question_encodings: DataDefinition([-1, -1, self.question_encoding_size], [torch.Tensor], "Batch of encoded questions [BATCH_SIZE x SEQ_LEN x QUESTION_ENCODING_SIZE]"),
            }


    def output_data_definitions(self):
        """
        Function returns a dictionary with definitions of output data produced the component.

        :return: dictionary containing output data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {
            self.key_outputs: DataDefinition([-1, self.output_size], [torch.Tensor], "Batch of outputs [BATCH_SIZE x OUTPUT_SIZE]")
            }

    def forward(self, data_streams):
        """
        Main forward pass of the model.

        :param data_streams: DataStreams({'images',**})
        :type data_streams: ``ptp.dadatypes.DataStreams``
        """

        # Unpack DataStreams.
        input_enc = data_streams[self.key_question_encodings] # [batch, num_words, embed_dim] # Dense prediction from RNN
        batch_size = input_enc.size()[0] # [48, 8, 100]

        # Attention computed as two FF layers with ReLU activation and softmax for probabilities ==> softmax(FF(ReLU(FF(input))))
        self.Attention = torch.softmax(self.W2(self.activation(self.W1(input_enc))), dim = 1) # [48, 8, 4] [batch, num_words, num_heads]

        # Multiply attention weights with question encoding
        input_enc_weighted = torch.matmul(self.Attention.transpose(1,2),input_enc) # [48, 4, 100] [batch, num_heads, embed_dim]

        # Concatenate features from multi-head attention
        outputs = input_enc_weighted.view(batch_size, -1) # [48, 400] [batch, num_heads*embed_dim]
        # # Alternatively: combine multi-head attention using a mean or sum operation
        # outputs = torch.sum(input_enc_weighted,1)/self.num_attention_heads

        # Add predictions to datadict.
        data_streams.publish({self.key_outputs: outputs})
