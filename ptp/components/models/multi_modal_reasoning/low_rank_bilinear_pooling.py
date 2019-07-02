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

from ptp.components.models.model import Model
from ptp.data_types.data_definition import DataDefinition


class LowRankBilinearPooling(Model):
    """
    Element of one of classical baselines for Visual Question Answering.
    The model inputs (question and image encodings) are fused via element-wise multiplication and returned (for subsequent classification, done in a separate component e.g. ffn).

    On the basis of: Jiasen Lu and Xiao Lin and Dhruv Batra and Devi Parikh. "Deeper LSTM and normalized CNN visual question answering model" (2015).
    """ 
    def __init__(self, name, config):
        """
        Initializes the model, creates the required layers.

        :param name: Name of the model (taken from the configuration file).

        :param config: Parameters read from configuration file.
        :type config: ``ptp.configuration.ConfigInterface``

        """
        super(LowRankBilinearPooling, self).__init__(name, LowRankBilinearPooling, config)

        # Get key mappings.
        self.key_image_encodings = self.stream_keys["image_encodings"]
        self.key_question_encodings = self.stream_keys["question_encodings"]
        self.key_outputs = self.stream_keys["outputs"]

        # Retrieve input/output sizes from globals.
        self.image_encoding_size = self.globals["image_encoding_size"]
        self.question_encoding_size = self.globals["question_encoding_size"]
        self.output_size = self.globals["output_size"]

        # Create the model.
        self.image_encodings_ff = torch.nn.Linear(self.image_encoding_size, self.output_size)
        self.question_encodings_ff = torch.nn.Linear(self.question_encoding_size, self.output_size)

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

    def forward(self, data_streams):
        """
        Main forward pass of the model.

        :param data_streams: DataStreams({'images',**})
        :type data_streams: ``ptp.dadatypes.DataStreams``
        """

        # Unpack DataStreams.
        enc_img = data_streams[self.key_image_encodings]
        enc_q = data_streams[self.key_question_encodings]

        # Apply nonlinearities and dropout on images.
        enc_img = self.activation(enc_img)
        enc_img = self.dropout(enc_img)

        # Apply nonlinearities and dropout on questions.
        enc_q = self.activation(enc_q)
        enc_q = self.dropout(enc_q)

        # Pass inputs layers mapping them to the same "latent space".
        latent_img = self.image_encodings_ff(enc_img)
        latent_q = self.question_encodings_ff(enc_q)
        
        # Element wise multiplication.
        outputs = latent_img * latent_q

        # Add predictions to datadict.
        data_streams.publish({self.key_outputs: outputs})
