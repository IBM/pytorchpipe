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


class RelationalNetwork(Model):
    """
    Model implements relational network.
    Model expects image (CNN) features and encoded question.

    
    """ 
    def __init__(self, name, config):
        """
        Initializes the model, creates the required layers.

        :param name: Name of the model (taken from the configuration file).

        :param config: Parameters read from configuration file.
        :type config: ``ptp.configuration.ConfigInterface``

        """
        super(RelationalNetwork, self).__init__(name, RelationalNetwork, config)

        # Get key mappings.
        self.key_feature_maps = self.stream_keys["feature_maps"]
        self.key_question_encodings = self.stream_keys["question_encodings"]
        self.key_outputs = self.stream_keys["outputs"]

        # Retrieve input sizes from globals.
        self.feature_maps_height = self.globals["feature_maps_height"]
        self.feature_maps_width = self.globals["feature_maps_width"]
        self.feature_maps_depth = self.globals["feature_maps_depth"]
        self.question_encoding_size = self.globals["question_encoding_size"]
        

        # Create "object" coordinates.
        self.obj_coords = []
        for h in range(self.feature_maps_height):
            for w in range(self.feature_maps_width):
                self.obj_coords.append((h,w))

        # Get output_size from config and send it to globals.
        self.output_size = self.config["output_size"]
        self.globals["output_size"] = self.output_size

        # Calculate input size to the g_theta: two "objects" + question (+ optionally: image size)
        input_size = 2 * self.feature_maps_depth + self.question_encoding_size

        # Retrieve dropout rate value - if set, will put dropout between every layer.
        dropout_rate = self.config["dropout_rate"]

        # Create the model, i.e. the "relational" g_theta MLP.
        self.g_theta = torch.nn.Sequential(
            torch.nn.Linear(input_size, self.output_size),
            # Create activation layer.
            torch.nn.ReLU(),
            # Create dropout layer.
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(self.output_size, self.output_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(self.output_size, self.output_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(self.output_size, self.output_size)
            )

        

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
        feat_m = data_dict[self.key_feature_maps]
        enc_q = data_dict[self.key_question_encodings]

        summed_relations = None
        # Iterate through all pairs of "objects".
        for (h1,w1) in self.obj_coords:
            for (h2,w2) in self.obj_coords:
                # Get feature maps.
                fm1 = feat_m[:, :, h1,w1].view(-1, self.feature_maps_depth)
                fm2 = feat_m[:, :, h2,w2].view(-1, self.feature_maps_depth)
                # Concatenate with question.
                concat = torch.cat([fm1, fm2, enc_q], dim=1)
                
                # Pass it through g_theta.
                rel = self.g_theta(concat)

                # Add to relations.
                if summed_relations is None:
                    summed_relations = rel
                else:
                    # Element wise sum.
                    summed_relations += rel

        # Add outputs to datadict.
        data_dict.extend({self.key_outputs: summed_relations})
