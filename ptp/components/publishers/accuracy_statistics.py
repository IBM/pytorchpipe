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
import math
import numpy as np

from ptp.components.component import Component
from ptp.data_types.data_definition import DataDefinition


class AccuracyStatistics(Component):
    """
    Class collecting statistics: batch size.

    """

    def __init__(self, name, config):
        """
        Initializes object.

        :param name: Loss name.
        :type name: str

        :param config: Dictionary of parameters (read from the configuration ``.yaml`` file).
        :type config: :py:class:`ptp.configuration.ConfigInterface`

        """
        # Call constructors of parent classes.
        Component.__init__(self, name, AccuracyStatistics, config)

        # Get stream key mappings.
        self.key_targets = self.stream_keys["targets"]
        self.key_predictions = self.stream_keys["predictions"]
        self.key_masks = self.stream_keys["masks"]

        # Get prediction distributions/indices flag.
        self.use_prediction_distributions = self.config["use_prediction_distributions"]

        # Get masking flag.
        self.use_masking = self.config["use_masking"]

        # Get statistics key mappings.
        self.key_accuracy = self.statistics_keys["accuracy"]


    def input_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of input data that are required by the component.

        :return: dictionary containing input data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        input_defs = {
            self.key_targets: DataDefinition([-1], [torch.Tensor], "Batch of targets, each being a single index [BATCH_SIZE]")
            }

        if self.use_prediction_distributions:
            input_defs[self.key_predictions] = DataDefinition([-1, -1], [torch.Tensor], "Batch of predictions, represented as tensor with probability distribution over classes [BATCH_SIZE x NUM_CLASSES]")
        else: 
            input_defs[self.key_predictions] = DataDefinition([-1], [torch.Tensor], "Batch of predictions, represented as tensor with indices of predicted answers [BATCH_SIZE]")
        
        if self.use_masking:
            input_defs[self.key_masks] = DataDefinition([-1], [torch.Tensor], "Batch of masks [BATCH_SIZE]")
        return input_defs


    def output_data_definitions(self):
        """ 
        Function returns a empty dictionary with definitions of output data produced the component.

        :return: Empty dictionary.
        """
        return {}


    def __call__(self, data_streams):
        """
        Call method - empty for all statistics.
        """
        pass


    def calculate_accuracy(self, data_streams):
        """
        Calculates accuracy equal to mean number of correct classification in a given batch.

        :param data_streams: DataStreams containing the targets.
        :type data_streams: DataStreams

        :return: Accuracy.

        """
        # Get targets.
        targets = data_streams[self.key_targets].data.cpu().numpy()

        if self.use_prediction_distributions:
            # Get indices of the max log-probability.
            preds = data_streams[self.key_predictions].max(1)[1].data.cpu().numpy()
        else: 
            preds = data_streams[self.key_predictions].data.cpu().numpy()

        # Calculate the correct predictinos.
        correct = np.equal(preds, targets)

        #print(" Target: {}\n Prediction: {}\n Correct: {}\n".format(targets, preds, correct))

        if self.use_masking:
            # Get masks from inputs.
            masks = data_streams[self.key_masks].data.cpu().numpy()
            correct = correct * masks
            batch_size = masks.sum()       
        else:
            batch_size = preds.shape[0]       
        
        #print(" Mask: {}\n Masked Correct: {}\n".format(masks, correct))

        # Simply sum the correct values.
        num_correct = correct.sum()

        #print(" num_correct: {}\n batch_size: {}\n".format(num_correct, batch_size))

        # Normalize by batch size.
        if batch_size > 0:
            accuracy = num_correct / batch_size
        else:
            accuracy = 0

        return accuracy, batch_size


    def add_statistics(self, stat_col):
        """
        Adds 'accuracy' statistics to ``StatisticsCollector``.

        :param stat_col: ``StatisticsCollector``.

        """
        stat_col.add_statistics(self.key_accuracy, '{:6.4f}')
        stat_col.add_statistics(self.key_accuracy+'_support', None)

    def collect_statistics(self, stat_col, data_streams):
        """
        Collects statistics (accuracy and support set size) for given episode.

        :param stat_col: ``StatisticsCollector``.

        """
        acc, batch_size = self.calculate_accuracy(data_streams)
        stat_col[self.key_accuracy] = acc
        stat_col[self.key_accuracy+'_support'] = batch_size
        

    def add_aggregators(self, stat_agg):
        """
        Adds aggregator summing samples from all collected batches.

        :param stat_agg: ``StatisticsAggregator``.

        """
        stat_agg.add_aggregator(self.key_accuracy, '{:7.5f}')  # represents the average accuracy
        #stat_agg.add_aggregator(self.key_accuracy+'_min', '{:7.5f}')
        #stat_agg.add_aggregator(self.key_accuracy+'_max', '{:7.5f}')
        stat_agg.add_aggregator(self.key_accuracy+'_std', '{:7.5f}')


    def aggregate_statistics(self, stat_col, stat_agg):
        """
        Aggregates samples from all collected batches.

        :param stat_col: ``StatisticsCollector``

        :param stat_agg: ``StatisticsAggregator``

        """
        accuracies = stat_col[self.key_accuracy]
        supports = stat_col[self.key_accuracy+'_support']

        # Special case - no samples!
        if sum(supports) == 0:
            stat_agg[self.key_accuracy] = 0
            stat_agg[self.key_accuracy+'_std'] = 0

        else: 
            # Calculate weighted precision.
            accuracies_avg = np.average(accuracies, weights=supports)
            accuracies_var = np.average((accuracies-accuracies_avg)**2, weights=supports)

            stat_agg[self.key_accuracy] = accuracies_avg
            #stat_agg[self.key_accuracy+'_min'] = np.min(accuracies)
            #stat_agg[self.key_accuracy+'_max'] = np.max(accuracies)
            stat_agg[self.key_accuracy+'_std'] = math.sqrt(accuracies_var)
