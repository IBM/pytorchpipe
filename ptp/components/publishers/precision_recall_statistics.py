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
import numpy as np
import math

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

        # Get stream key mappings.
        self.key_targets = self.stream_keys["targets"]
        self.key_predictions = self.stream_keys["predictions"]
        self.key_masks = self.stream_keys["masks"]

        # Get prediction distributions/indices flag.
        self.use_prediction_distributions = self.config["use_prediction_distributions"]

        # Get masking flag.
        self.use_masking = self.config["use_masking"]

        # Get statistics key mappings.
        self.key_precision = self.statistics_keys["precision"]
        self.key_recall = self.statistics_keys["recall"]
        self.key_f1score = self.statistics_keys["f1score"]

        
        # Get (or create) vocabulary.
        if self.config["use_word_mappings"]:
            # Get labels from word mappings.
            self.labels = []
            self.index_mappings = {}
            # Assume they are ordered, starting from 0.
            for i,(word,index) in enumerate(self.globals["word_mappings"].items()):
                self.labels.append(word)
                self.index_mappings[index] = i
            # Set number of classes by looking at labels.
            self.num_classes = len(self.labels)
        else:
            # Get the number of possible outputs.
            self.num_classes = self.globals["num_classes"]
            self.labels = list(range(self.num_classes))
            self.index_mappings = {i: i for i in range(self.num_classes)}

        # Check display options.
        self.show_confusion_matrix = self.config["show_confusion_matrix"]
        self.show_class_scores = self.config["show_class_scores"]

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
        Calculates precission recall statistics.

        :param data_streams: DataStreams containing the targets.
        :type data_streams: DataStreams

        """
        # Use worker interval.
        if self.app_state.episode % self.app_state.args.logging_interval == 0:

            # Calculate all four statistics.
            confusion_matrix, precisions, recalls, f1scores, supports = self.calculate_statistics(data_streams)

            if self.show_confusion_matrix:
                self.logger.info("Confusion matrix:\n{}".format(confusion_matrix))

            # Calculate weighted averages.
            support_sum = sum(supports)
            if support_sum > 0:
                precision_avg = sum([pi*si for (pi,si) in zip(precisions,supports)]) / support_sum 
                recall_avg = sum([ri*si for (ri,si) in zip(recalls,supports)]) / support_sum
                f1score_avg = sum([fi*si for (fi,si) in zip(f1scores,supports)]) / support_sum
            else:
                precision_avg = 0
                recall_avg = 0
                f1score_avg = 0

            # Log class scores.
            if self.show_class_scores:
                log_str = "\n| Precision | Recall | F1Score | Support | Label\n"
                log_str+= "|-----------|--------|---------|---------|-------\n"
                for i in range(self.num_classes):
                    log_str += "|    {:05.4f} | {:05.4f} |  {:05.4f} |   {:5d} | {}\n".format(
                        precisions[i], recalls[i], f1scores[i], supports[i], self.labels[i])
                log_str+= "|-----------|--------|---------|---------|-------\n"
                log_str += "|    {:05.4f} | {:05.4f} |  {:05.4f} |   {:5d} | Weighted Avg\n".format(
                        precision_avg, recall_avg, f1score_avg, support_sum)
                self.logger.info(log_str)


    def calculate_statistics(self, data_streams):
        """
        Calculates confusion_matrix, precission, recall, f1score and support statistics.

        :param data_streams: DataStreams containing the targets.
        :type data_streams: DataStreams

        :return: Calculated statistics.
        """
        targets = data_streams[self.key_targets].data.cpu().numpy()
        #print("Targets :", targets)

        if self.use_prediction_distributions:
            # Get indices of the max log-probability.
            preds = data_streams[self.key_predictions].max(1)[1].data.cpu().numpy()
        else: 
            preds = data_streams[self.key_predictions].data.cpu().numpy()
        #print("Predictions :", preds)

        if self.use_masking:
            # Get masks from inputs.
            masks = data_streams[self.key_masks].data.cpu().numpy()
        else:
            # Create vector full of ones.
            masks = np.ones(targets.shape[0])

        # Create the confusion matrix, use SciKit learn order:
        # Column - predicted class
        #print(self.index_mappings)
        # Row - target (actual) class
        confusion_matrix = np.zeros([self.num_classes, self.num_classes], dtype=int)
        for i, (target, pred) in enumerate(zip(targets, preds)):
            #print("T: ",target)
            #print("P: ",pred)
            # If both indices are ok.
            if target in self.index_mappings.keys() and pred in self.index_mappings.keys():
                #print(self.index_mappings[target])
                #print(self.index_mappings[pred])
                confusion_matrix[self.index_mappings[target]][self.index_mappings[pred]] += 1 * masks[i]

        # Calculate true positive (TP), eqv. with hit.
        tp = np.zeros([self.num_classes], dtype=int)
        for i in range(self.num_classes):
            tp[i] = confusion_matrix[i][i]
        #print("TP = ",tp)        

        # Calculate false positive (FP) eqv. with false alarm, Type I error
        # Predictions that incorrectly labelled as belonging to a given class.
        # Sum wrong predictions along the column.
        fp = np.sum(confusion_matrix, axis=0) - tp
        #print("FP = ",fp)

        # Calculate false negative (FN), eqv. with miss, Type II error
        # The target belonged to a given class, but it wasn't correctly labeled.
        # Sum wrong predictions along the row.
        fn = np.sum(confusion_matrix, axis=1) - tp
        #print("FN = ",fn)        

        # Precision is the fraction of events where we correctly declared i
        # out of all instances where the algorithm declared i.
        precisions = [float(tpi) / float(tpi+fpi) if (tpi+fpi) > 0 else 0.0 for (tpi,fpi) in zip(tp,fp)]

        # Recall is the fraction of events where we correctly declared i 
        # out of all of the cases where the true of state of the world is i.
        recalls = [float(tpi) / float(tpi+fni) if (tpi+fni) > 0 else 0.0 for (tpi,fni) in zip(tp,fn)]

        # Calcualte f1-scores.
        f1scores = [ 2 * pi * ri / float(pi+ri) if (pi+ri) > 0 else 0.0 for (pi,ri) in zip(precisions,recalls)]

        # Get support.
        supports = np.sum(confusion_matrix, axis=1)

        #print('precision: {}'.format(precision))
        #print('recall: {}'.format(recall))
        #print('f1score: {}'.format(f1score))
        #print('support: {}'.format(support))

        return confusion_matrix, precisions, recalls, f1scores, supports


    def add_statistics(self, stat_col):
        """
        Adds 'accuracy' statistics to ``StatisticsCollector``.

        :param stat_col: ``StatisticsCollector``.

        """
        # Those will be displayed.
        stat_col.add_statistics(self.key_precision, '{:05.4f}')
        stat_col.add_statistics(self.key_recall, '{:05.4f}')
        stat_col.add_statistics(self.key_f1score, '{:05.4f}')
        # That one will be collected and used by aggregator.
        stat_col.add_statistics(self.key_f1score+'_support', None)


    def collect_statistics(self, stat_col, data_streams):
        """
        Collects statistics (batch_size) for given episode.

        :param stat_col: ``StatisticsCollector``.

        """
        # Calculate all four statistics.
        _, precisions, recalls, f1scores, supports = self.calculate_statistics(data_streams)

        # Calculate weighted averages.
        precision_sum = sum([pi*si for (pi,si) in zip(precisions,supports)])
        recall_sum = sum([ri*si for (ri,si) in zip(recalls,supports)])
        f1score_sum = sum([fi*si for (fi,si) in zip(f1scores,supports)])
        support_sum = sum(supports)

        if support_sum > 0:
            precision_avg = precision_sum / support_sum 
            recall_avg = recall_sum / support_sum
            f1score_avg = f1score_sum / support_sum
        else:
            precision_avg = 0
            recall_avg = 0
            f1score_avg = 0

        # Export averages to statistics.
        stat_col[self.key_precision] = precision_avg
        stat_col[self.key_recall] = recall_avg
        stat_col[self.key_f1score] = f1score_avg

        # Export support to statistics.
        stat_col[self.key_f1score+'_support'] = support_sum



    def add_aggregators(self, stat_agg):
        """
        Adds aggregator summing samples from all collected batches.

        :param stat_agg: ``StatisticsAggregator``.

        """
        stat_agg.add_aggregator(self.key_precision, '{:05.4f}') 
        stat_agg.add_aggregator(self.key_precision+'_std', '{:05.4f}')
        stat_agg.add_aggregator(self.key_recall, '{:05.4f}') 
        stat_agg.add_aggregator(self.key_recall+'_std', '{:05.4f}')
        stat_agg.add_aggregator(self.key_f1score, '{:05.4f}') 
        stat_agg.add_aggregator(self.key_f1score+'_std', '{:05.4f}')


    def aggregate_statistics(self, stat_col, stat_agg):
        """
        Aggregates samples from all collected batches.

        :param stat_col: ``StatisticsCollector``

        :param stat_agg: ``StatisticsAggregator``

        """
        precision_sums = stat_col[self.key_precision]
        recall_sums = stat_col[self.key_recall]
        f1score_sums = stat_col[self.key_f1score]
        supports = stat_col[self.key_f1score+'_support']

        # Special case - no samples!
        if sum(supports) == 0:
            stat_agg[self.key_precision] = 0
            stat_agg[self.key_precision+'_std'] = 0
            stat_agg[self.key_recall] = 0
            stat_agg[self.key_recall+'_std'] = 0
            stat_agg[self.key_f1score] = 0
            stat_agg[self.key_f1score+'_std'] = 0

        else: 
            # Else: calculate weighted precision.
            precisions_avg = np.average(precision_sums, weights=supports)
            precisions_var = np.average((precision_sums-precisions_avg)**2, weights=supports)
            
            stat_agg[self.key_precision] = precisions_avg
            stat_agg[self.key_precision+'_std'] = math.sqrt(precisions_var)

            # Calculate weighted recall.
            recalls_avg = np.average(recall_sums, weights=supports)
            recalls_var = np.average((recall_sums-recalls_avg)**2, weights=supports)

            stat_agg[self.key_recall] = recalls_avg
            stat_agg[self.key_recall+'_std'] = math.sqrt(recalls_var)

            # Calculate weighted f1 score.
            f1scores_avg = np.average(f1score_sums, weights=supports)
            f1scores_var = np.average((f1score_sums-f1scores_avg)**2, weights=supports)

            stat_agg[self.key_f1score] = f1scores_avg
            stat_agg[self.key_f1score+'_std'] = math.sqrt(f1scores_var)
