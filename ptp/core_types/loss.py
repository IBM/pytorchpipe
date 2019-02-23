# -*- coding: utf-8 -*-
#
# Copyright (C) tkornuta, IBM Corporation 2019
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

from ptp.core_types.component import Component
from ptp.core_types.data_definition import DataDefinition


class Loss(Component):
    """
    Class representing base class for all losses.

    Implements features & attributes used by all subclasses.

    """

    def __init__(self, name, params):
        """
        Initializes loss object.

        :param name: Loss name.
        :type name: str

        :param params: Dictionary of parameters (read from the configuration ``.yaml`` file).
        :type params: :py:class:`ptp.utils.ParamInterface`

        """
        # Call constructors of parent classes.
        Component.__init__(self, name, params)

        # Set key mappings.
        self.key_targets = self.mapkey("targets")
        self.key_predictions = self.mapkey("predictions")
        self.key_loss = self.mapkey("loss")

    def input_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of input data that are required by the component.

        :return: dictionary containing input data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {
            self.key_targets: DataDefinition([-1, 1], [list, int], "Batch of targets, each represented as index [BATCH_SIZE] x [int]"),
            self.key_predictions: DataDefinition([-1, -1], [torch.Tensor], "Batch of predictions, represented as tensor with probability distribution over classes [BATCH_SIZE x NUM_CLASSES]")
            }


    def output_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of output data produced the component.

        :return: dictionary containing output data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {
            self.key_loss: DataDefinition([1], [torch.Tensor], "Loss value (scalar, i.e. 1D tensor)")
            }

    def loss_keys(self):
        """ 
        Function returns a list containing keys used to return losses in DataDict.
        Those keys will be used to find objects that will be roots for backpropagation of gradients.

        :return: list of keys associated with losses returned by the component.
        """
        return [ self.key_loss ]
