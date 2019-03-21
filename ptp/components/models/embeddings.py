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


class Embeddings(Model):
    """
    A simple model consisting of a single word embeddings layer.
    """ 
    def __init__(self, name, params):
        """
        Initializes the ``Embeddings`` layer.

        :param name: Name of the model (taken from the configuration file).

        :param params: Parameters read from configuration file.
        :type params: ``miprometheus.utils.ParamInterface``
        """
        super(Embeddings, self).__init__(name, params)

        # Set key mappings.
        self.key_inputs = self.mapkey("inputs")
        self.key_outputs = self.mapkey("outputs")

        # Retrieve vocabulary size from globals.
        self.key_vocab_size = self.mapkey("vocab_size")
        vocab_size = self.app_state[self.key_vocab_size]

        # Retrieve embeddings size from configuration and export it to globals.
        self.embeddings_size = params['embeddings_size']
        self.key_embeddings_size = self.mapkey("embeddings_size")
        self.app_state[self.key_embeddings_size] = self.embeddings_size

        self.logger.info("Initializing embeddings layer with vocabulary size = {} and embeddings size = {}".format(vocab_size, self.embeddings_size))

        # Finally: create the embeddings layer.
        self.embeddings = torch.nn.Embedding(vocab_size, self.embeddings_size)


    def input_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of input data that are required by the component.

        :return: dictionary containing input data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {
            self.key_inputs: DataDefinition([-1, -1], [torch.Tensor], "Batch of of indexed sentences [BATCH_SIZE x SENTENCE_LENGTH]"),
            }


    def output_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of output data produced the component.

        :return: dictionary containing output data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {
            self.key_outputs: DataDefinition([-1, -1, self.embeddings_size], [torch.Tensor], "Batch of embedded sentences [BATCH_SIZE x SENTENCE_LENGTH x EMBEDDING_SIZE]")
            }


    def forward(self, data_dict):
        """
        Forward pass  - performs embedding.

        :param data_dict: DataDict({'images',**}), where:

            - inputs: expected indexed sentences [BATCH_SIZE x SENTENCE_LENGTH]
            - outputs: added embedded sentences [BATCH_SIZE x SENTENCE_LENGTH x EMBEDDING_SIZE]

        :type data_dict: ``miprometheus.utils.DataDict``
        """

        # Unpack DataDict.
        inputs = data_dict[self.key_inputs]

        # Embedd inputs.
        embeds = self.embeddings(inputs)

        # Add embeddings to datadict.
        data_dict.extend({self.key_outputs: embeds})
