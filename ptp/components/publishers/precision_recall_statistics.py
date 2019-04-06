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

import numpy as np
import torch

from ptp.components.component import Component
from ptp.data_types.data_definition import DataDefinition


class PrecisionRecallStatistics(Component):
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
        Component.__init__(self, name, PrecisionRecallStatistics, config)

        # Set key mappings.
        self.key_targets = self.stream_keys["targets"]
        self.key_predictions = self.stream_keys["predictions"]

        # Get statistic key mappings.
        self.key_precision = self.statistics_keys["precision"]
        self.key_recall = self.statistics_keys["recall"]
        self.key_f1score = self.statistics_keys["f1score"]

        # Get the number of possible outputs.
        self.num_classes = self.globals["num_classes"]
        
        # Get (or create) vocabulary.
        if self.config["use_word_mappings"]:
            # Get labels from word mappings.
            self.labels = []
            # Assume they are ordered, starting from 0.
            for (_,value) in self.globals["word_mappings"]:
                self.labels.append(value)
        else:
            self.labels = list(range(self.num_classes))

        # Check display options.
        self.show_confusion_matrix = self.config["show_confusion_matrix"]
        self.show_class_scores = self.config["show_class_scores"]

        # Internal counter, as we do not have access to episode number.
        self.episode = 0

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
        Calculates prediction recall statistics.

        :param data_dict: DataDict containing the targets.
        :type data_dict: DataDict

        """
        # Use worker interval.
        if self.episode % self.app_state.args.logging_interval == 0:

            targets = data_dict[self.key_targets]
            #print("Targets :", targets)

            # Get indices of the max log-probability.
            preds = data_dict[self.key_predictions].max(1)[1]
            #print("Predictions :", preds)

            # Create the confusion matrix, use SciKit learn order:
            # Column - predicted class
            # Row - target (actual) class
            confunsion_matrix = np.zeros([self.num_classes, self.num_classes], dtype=int)
            for (target, pred) in zip(targets, preds):
                confunsion_matrix[target][pred] += 1

            if self.show_confusion_matrix:
                self.logger.info("Confusion matrix:\n{}".format(confunsion_matrix))

            # Calculate true positive (TP), eqv. with hit.
            tp = np.zeros([self.num_classes], dtype=int)
            for i in range(self.num_classes):
                tp[i] = confunsion_matrix[i][i]
            #print("TP = ",tp)        

            # Calculate false positive (FP) eqv. with false alarm, Type I error
            # Predictions that incorrectly labelled as belonging to a given class.
            # Sum wrong predictions along the column.
            fp = np.sum(confunsion_matrix, axis=0) - tp
            #print("FP = ",fp)

            # Calculate false negative (FN), eqv. with miss, Type II error
            # The target belonged to a given class, but it wasn't correctly labeled.
            # Sum wrong predictions along the row.
            fn = np.sum(confunsion_matrix, axis=1) - tp
            #print("FN = ",fn)        

            # Precision is the fraction of events where we correctly declared i
            # out of all instances where the algorithm declared i.
            precision = [float(tpi) / float(tpi+fpi) if (tpi+fpi) > 0 else 0.0 for (tpi,fpi) in zip(tp,fp)]

            # Recall is the fraction of events where we correctly declared i 
            # out of all of the cases where the true of state of the world is i.
            recall = [float(tpi) / float(tpi+fni) if (tpi+fni) > 0 else 0.0 for (tpi,fni) in zip(tp,fn)]

            # Calcualte f1-score.
            f1score = [ 2 * pi * ri / (pi+ri) if (pi+ri) > 0 else 0.0 for (pi,ri) in zip(precision,recall)]

            # Get support.
            support = np.sum(confunsion_matrix, axis=1)

            # Calculate weighted averages.
            support_sum = sum(support)
            precision_avg = sum([pi*si / support_sum for (pi,si) in zip(precision,support)])
            recall_avg = sum([ri*si / support_sum for (ri,si) in zip(recall,support)])
            f1score_avg = sum([fi*si / support_sum for (fi,si) in zip(f1score,support)])

            # Remember those values so can be used in statistics.
            self.precision = precision_avg
            self.recall = recall_avg
            self.f1score = f1score_avg

            #print('precision: {}'.format(precision))
            #print('recall: {}'.format(recall))
            #print('f1score: {}'.format(f1score))
            #print('support: {}'.format(support))

            # Log class scores.
            if self.show_class_scores:
                log_str = "\n| Precision | Recall | F1Score | Support | Label\n"
                log_str+= "|-----------|--------|---------|---------|-------\n"
                for i in range(self.num_classes):
                    log_str += "|     {:05.3f} |  {:05.3f} |   {:05.3f} |   {:5d} | {}\n".format(
                        precision[i], recall[i], f1score[i], support[i], self.labels[i])
                log_str+= "|-----------|--------|---------|---------|-------\n"
                log_str += "|     {:05.3f} |  {:05.3f} |   {:05.3f} |   {:5d} | weighted avg\n".format(
                        precision_avg, recall_avg, f1score_avg, support_sum)
                self.logger.info(log_str)

        # Increment episode.
        self.episode += 1


    def add_statistics(self, stat_col):
        """
        Adds 'accuracy' statistics to ``StatisticsCollector``.

        :param stat_col: ``StatisticsCollector``.

        """
        stat_col.add_statistics(self.key_precision, '{:12.10f}')
        stat_col.add_statistics(self.key_recall, '{:12.10f}')
        stat_col.add_statistics(self.key_f1score, '{:12.10f}')

    def collect_statistics(self, stat_col, data_dict):
        """
        Collects statistics (batch_size) for given episode.

        :param stat_col: ``StatisticsCollector``.

        """
        stat_col[self.key_precision] = self.precision
        stat_col[self.key_recall] = self.recall
        stat_col[self.key_f1score] = self.f1score

    def add_aggregators(self, stat_agg):
        """
        Adds aggregator summing samples from all collected batches.

        :param stat_agg: ``StatisticsAggregator``.

        """
        stat_agg.add_aggregator(self.key_precision, '{:12.10f}') 
        stat_agg.add_aggregator(self.key_precision+'_std', '{:12.10f}')
        stat_agg.add_aggregator(self.key_recall, '{:12.10f}') 
        stat_agg.add_aggregator(self.key_recall+'_std', '{:12.10f}')
        stat_agg.add_aggregator(self.key_f1score, '{:12.10f}') 
        stat_agg.add_aggregator(self.key_f1score+'_std', '{:12.10f}')


    def aggregate_statistics(self, stat_col, stat_agg):
        """
        Aggregates samples from all collected batches.

        :param stat_col: ``StatisticsCollector``

        :param stat_agg: ``StatisticsAggregator``

        """
        precisions = stat_col[self.key_precision]
        recalls = stat_col[self.key_recall]
        f1scores = stat_col[self.key_f1score]
        
        # TODO: instead of mean use weighted sum + mean.
        stat_agg[self.key_precision] = torch.mean(torch.tensor(precisions))
        stat_agg[self.key_precision+'_std'] = 0.0 if len(precisions) <= 1 else torch.std(torch.tensor(precisions))

        stat_agg[self.key_recall] = torch.mean(torch.tensor(recalls))
        stat_agg[self.key_recall+'_std'] = 0.0 if len(recalls) <= 1 else torch.std(torch.tensor(recalls))

        stat_agg[self.key_f1score] = torch.mean(torch.tensor(f1scores))
        stat_agg[self.key_f1score+'_std'] = 0.0 if len(f1s) <= 1 else torch.std(torch.tensor(f1scores))
