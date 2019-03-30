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


class Accuracy(Component):
    """
    Class collecting statistics: batch size.

    """

    def __init__(self, name, params):
        """
        Initializes object.

        :param name: Loss name.
        :type name: str

        :param params: Dictionary of parameters (read from the configuration ``.yaml`` file).
        :type params: :py:class:`ptp.utils.ParamInterface`

        """
        # Call constructors of parent classes.
        Component.__init__(self, name, Accuracy, params)

        # Set key mappings.
        self.key_targets = self.stream_keys["targets"]
        self.key_predictions = self.stream_keys["predictions"]

        self.key_accuracies = self.statistics_keys["accuracies"]


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
        Function returns a empty dictionary with definitions of output data produced the component.

        :return: Empty dictionary.
        """
        return {}


    def __call__(self, data_dict):
        """
        Call method - empty for all statistics.
        """
        pass


    def calculate_accuracy(self, data_dict):
        """
        Calculates accuracy equal to mean number of correct classification in a given batch.

        :param data_dict: DataDict containing the targets.
        :type data_dict: DataDict

        :return: Accuracy.

        """
        # Get indices of the max log-probability.
        #pred = data_dict[self.key_predictions].max(1, keepdim=True)[1]
        preds = data_dict[self.key_predictions].max(1)[1]

        # Calculate the number of correct predictinos.
        correct = preds.eq(data_dict[self.key_targets]).sum().item()
        #print ("TARGETS = ",data_dict[self.key_targets])
        #print ("PREDICTIONS = ",data_dict[self.key_predictions])
        #print ("MAX PREDICTIONS = ", preds)
        #print("CORRECTS = ", correct)

        # Normalize.
        batch_size = data_dict[self.key_predictions].shape[0]       
        accuracy = correct / batch_size
        #print("ACCURACY = ", accuracy)

        return accuracy


    def add_statistics(self, stat_col):
        """
        Adds 'accuracy' statistics to ``StatisticsCollector``.

        :param stat_col: ``StatisticsCollector``.

        """
        stat_col.add_statistic(self.key_accuracies, '{:12.10f}')

    def collect_statistics(self, stat_col, data_dict):
        """
        Collects statistics (batch_size) for given episode.

        :param stat_col: ``StatisticsCollector``.

        """
        stat_col[self.key_accuracies] = self.calculate_accuracy(data_dict)

    def add_aggregators(self, stat_agg):
        """
        Adds aggregator summing samples from all collected batches.

        :param stat_agg: ``StatisticsAggregator``.

        """
        stat_agg.add_aggregator(self.key_accuracies, '{:12.10f}')  # represents the average accuracy
        stat_agg.add_aggregator(self.key_accuracies+'_min', '{:12.10f}')
        stat_agg.add_aggregator(self.key_accuracies+'_max', '{:12.10f}')
        stat_agg.add_aggregator(self.key_accuracies+'_std', '{:12.10f}')


    def aggregate_statistics(self, stat_col, stat_agg):
        """
        Aggregates samples from all collected batches.

        :param stat_col: ``StatisticsCollector``

        :param stat_agg: ``StatisticsAggregator``

        """
        acc = stat_col[self.key_accuracies]
        # TODO: instead of mean use weighted sum + mean.
        stat_agg[self.key_accuracies] = torch.mean(torch.tensor(acc))
        stat_agg[self.key_accuracies+'_min'] = min(acc)
        stat_agg[self.key_accuracies+'_max'] = max(acc)
        stat_agg[self.key_accuracies+'_std'] = 0.0 if len(acc) <= 1 else torch.std(torch.tensor(acc))
