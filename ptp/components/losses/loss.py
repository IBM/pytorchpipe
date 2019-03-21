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

from ptp.components.component import Component
from ptp.data_types.data_definition import DataDefinition


class Loss(Component):
    """
    Class representing base class for all losses.

    Implements features & attributes used by all subclasses.

    """

    def __init__(self, name, class_type, params):
        """
        Initializes loss object.

        :param name: Loss name.
        :type name: str

        :param class_type: Class type of the component.

        :param params: Dictionary of parameters (read from the configuration ``.yaml`` file).
        :type params: :py:class:`ptp.utils.ParamInterface`

        """
        # Call constructors of parent classes.
        Component.__init__(self, name, class_type, params)

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
            self.key_targets: DataDefinition([-1], [torch.Tensor], "Batch of targets, each being a single index [BATCH_SIZE]"),
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


    def add_statistics(self, stat_col):
        """
        Adds most elementary shared statistics to ``StatisticsCollector``: episode and loss.

        :param stat_col: ``StatisticsCollector``.

        """
        # Add default statistics with formatting.
        stat_col.add_statistic(self.key_loss, '{:12.10f}')

    def collect_statistics(self, stat_col, data_dict):
        """
        Collects statistics (loss) for given episode.

        :param stat_col: ``StatisticsCollector``.

        """
        stat_col[self.key_loss] = data_dict[self.key_loss].item()

    def add_aggregators(self, stat_agg):
        """
        Adds basic loss-related statistical aggregators to ``StatisticsAggregator``.

        :param stat_agg: ``StatisticsAggregator``.

        """
        # Add default statistical aggregators for the loss (indicating a formatting).
        # Represents the average loss, but stying with loss for TensorBoard "variable compatibility".
        stat_agg.add_aggregator(self.key_loss, '{:12.10f}')  
        stat_agg.add_aggregator(self.key_loss+'_min', '{:12.10f}')
        stat_agg.add_aggregator(self.key_loss+'_max', '{:12.10f}')
        stat_agg.add_aggregator(self.key_loss+'_std', '{:12.10f}')

    def aggregate_statistics(self, stat_col, stat_agg):
        """
        Aggregates the statistics collected by the ``StatisticsCollector``.

        .. note::
            Computes min, max, mean, std of the loss as these are basic statistical aggregator by default.

            Given that the ``StatisticsAggregator`` uses the statistics collected by the ``StatisticsCollector``, \
            It should be ensured that these statistics are correctly collected (i.e. use of ``self.add_statistics()`` \
            and ``collect_statistics()``).

        :param stat_col: ``StatisticsCollector``

        :param stat_agg: ``StatisticsAggregator``

        """
        # Get loss values.
        loss_values = stat_col[self.key_loss]

        # Calculate default aggregates.
        stat_agg.aggregators[self.key_loss] = torch.mean(torch.tensor(loss_values))
        stat_agg.aggregators[self.key_loss+'_min'] = min(loss_values)
        stat_agg.aggregators[self.key_loss+'_max'] = max(loss_values)
        stat_agg.aggregators[self.key_loss+'_std'] = 0.0 if len(loss_values) <= 1 else torch.std(torch.tensor(loss_values))
