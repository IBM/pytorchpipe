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

from ptp.configuration.configuration_error import ConfigurationError
from ptp.components.models.model import Model
from ptp.data_types.data_definition import DataDefinition


class RelationalNetwork(Model):
    """
    Model implements relational network.
    Model expects image (CNN) features and encoded question.

    
    Santoro, A., Raposo, D., Barrett, D. G., Malinowski, M., Pascanu, R., Battaglia, P., & Lillicrap, T. (2017). A simple neural network module for relational reasoning. In Advances in neural information processing systems (pp. 4967-4976).
    Reference paper: https://arxiv.org/abs/1706.01427.
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

        # Calculate input size to the g_theta: two "objects" + question (+ optionally: image size)
        input_size = 2 * self.feature_maps_depth + self.question_encoding_size

        # Create the module list.
        modules = []

        # Retrieve dropout rate value - if set, will put dropout between every layer.
        dropout_rate = self.config["dropout_rate"]

        # Create the model, i.e. the "relational" g_theta network.
        g_theta_sizes = self.config["g_theta_sizes"]
        if type(g_theta_sizes) == list and len(g_theta_sizes) > 1:
            # First input dim.
            input_dim = input_size
            for hidden_dim in g_theta_sizes:
                # Add linear layer.
                modules.append( torch.nn.Linear(input_dim, hidden_dim) )
                # Add activation and dropout.
                modules.append( torch.nn.ReLU() )
                if (dropout_rate > 0):
                    modules.append( torch.nn.Dropout(dropout_rate) )
                # Remember input dim of next layer.
                input_dim = hidden_dim

            # Add output layer.
            modules.append( torch.nn.Linear(input_dim, hidden_dim) )

            self.logger.info("Created g_theta network with {} layers".format(len(g_theta_sizes)+1))

        else:
            raise ConfigurationError("'g_theta_sizes' must contain a list with numbers of neurons in g_theta layers (currently {})".format(self.hidden_sizes))

        # Export output_size  to globals.
        self.output_size = g_theta_sizes[-1]
        self.globals["output_size"] = self.output_size

        # Finally create the sequential model out of those modules.
        self.g_theta = torch.nn.Sequential(*modules)


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

    def forward(self, data_streams):
        """
        Main forward pass of the model.

        :param data_streams: DataStreams({'images',**})
        :type data_streams: ``ptp.dadatypes.DataStreams``
        """
        # Unpack DataStreams.
        feat_m = data_streams[self.key_feature_maps]
        enc_q = data_streams[self.key_question_encodings]

        # List [FEAT_WIDTH x FEAT_HEIGHT] of tensors [BATCH SIZE x (2 * FEAT_DEPTH + QUESTION_SIZE)]
        relational_inputs = []
        # Iterate through all pairs of "objects".
        for (h1,w1) in self.obj_coords:
            for (h2,w2) in self.obj_coords:
                # Get feature maps.
                fm1 = feat_m[:, :, h1,w1].view(-1, self.feature_maps_depth)
                fm2 = feat_m[:, :, h2,w2].view(-1, self.feature_maps_depth)
                # Concatenate with question [BATCH SIZE x (2 * FEAT_DEPTH + QUESTION_SIZE)]
                concat = torch.cat([fm1, fm2, enc_q], dim=1)
                relational_inputs.append(concat)

        # Stack tensors along with the batch dimension
        # [BATCH SIZE x (FEAT_WIDTH x FEAT_HEIGHT)^2  x (2 * FEAT_DEPTH + QUESTION_SIZE)]
        # i.e. [BATCH SIZE x NUM_RELATIONS  x (2 * FEAT_DEPTH + QESTION_SIZE)]
        stacked_inputs = torch.stack(relational_inputs, dim=1)

        # Get shape [BATCH SIZE x (2 * FEAT_DEPTH + QESTION_SIZE)]
        shape = stacked_inputs.shape

        # Reshape such that we do a broadcast over the last dimension.
        stacked_inputs = stacked_inputs.contiguous().view(-1, shape[-1])

        # Pass it through g_theta.
        stacked_relations = self.g_theta(stacked_inputs)

        # Reshape to [BATCH_SIZE x NUM_RELATIONS x OUTPUT_SIZE]
        stacked_relations = stacked_relations.view(*shape[0:-1], self.output_size)

        # Element wise sum along relations [BATCH_SIZE x OUTPUT_SIZE]
        summed_relations = torch.sum(stacked_relations, dim=1)

        # Add outputs to datadict.
        data_streams.publish({self.key_outputs: summed_relations})
