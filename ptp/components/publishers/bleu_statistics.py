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
from nltk.translate.bleu_score import sentence_bleu

from ptp.components.component import Component
from ptp.data_types.data_definition import DataDefinition


class BLEUStatistics(Component):
    """
    Class collecting statistics: BLEU (Bilingual Evaluation Understudy Score).

    It accepts targets and predictions represented as indices of words and uses the provided word mappings to change those into words used finally for calculation of BLEU similarity.

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
        Component.__init__(self, name, BLEUStatistics, config)

        # Get stream key mappings.
        self.key_targets = self.stream_keys["targets"]
        self.key_predictions = self.stream_keys["predictions"]
        self.key_masks = self.stream_keys["masks"]

        # Get prediction distributions/indices flag.
        self.use_prediction_distributions = self.config["use_prediction_distributions"]

        # Get masking flag.
        #self.use_masking = self.config["use_masking"]

        # Retrieve word mappings from globals.
        word_to_ix = self.globals["word_mappings"]
        # Construct reverse mapping for faster processing.
        self.ix_to_word = dict((v,k) for k,v in word_to_ix.items())

        # Get masking flag.
        self.weights = self.config["weights"]


        # Get statistics key mappings.
        self.key_bleu = self.statistics_keys["bleu"]


    def input_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of input data that are required by the component.

        :return: dictionary containing input data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        # Add targets.
        input_defs = {
            self.key_targets: DataDefinition([-1, -1], [torch.Tensor], "Batch of sentences represented as a single tensor of indices of particular words  [BATCH_SIZE x SEQ_LENGTH]"),
            }
        # Add predictions.
        if self.use_prediction_distributions:
            input_defs[self.key_predictions] = DataDefinition([-1, -1, -1], [torch.Tensor], "Batch of predictions, represented as tensor with sequences of probability distributions over classes [BATCH_SIZE x SEQ_LENGTH x NUM_CLASSES]")
        else: 
            input_defs[self.key_predictions] = DataDefinition([-1, -1], [torch.Tensor], "Batch of predictions, represented as tensor with sequences of indices of predicted answers [BATCH_SIZE x SEQ_LENGTH]")
        # Add masks.
        #if self.use_masking:
        #    input_defs[self.key_masks] = DataDefinition([-1, -1], [torch.Tensor], "Batch of masks (separate mask for each sequence in the batch) [BATCH_SIZE x SEQ_LENGTH]")
        return input_defs


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


    def calculate_BLEU(self, data_dict):
        """
        Calculates BLEU for predictions of a given batch.

        :param data_dict: DataDict containing the targets and predictions (and optionally masks).
        :type data_dict: DataDict

        :return: Accuracy.

        """
        # Get targets.
        targets = data_dict[self.key_targets].data.cpu().numpy().tolist()

        if self.use_prediction_distributions:
            # Get indices of the max log-probability.
            preds = data_dict[self.key_predictions].max(-1)[1].data.cpu().numpy().tolist()
        else: 
            preds = data_dict[self.key_predictions].data.cpu().numpy().tolist()

        #if self.use_masking:
        #    # Get masks from inputs.
        #    masks = data_dict[self.key_masks].data.cpu().numpy().tolist()
        #else:
        #    batch_size = preds.shape[0]       
        
        # Calculate the correct predictinos.
        scores = []

        #print("targets ({}): {}\n".format(len(targets), targets[0]))
        #print("preds ({}): {}\n".format(len(preds), preds[0]))

        for target_indices, pred_indices in zip(targets, preds):
            # Change target indices to words.
            target_words = []
            for t_ind in target_indices:
                if t_ind in self.ix_to_word.keys():
                    target_words.append(self.ix_to_word[t_ind])
            # Change prediction indices to words.
            pred_words = []
            for p_ind in pred_indices:
                if p_ind in self.ix_to_word.keys():
                    pred_words.append(self.ix_to_word[p_ind])
            # Calculate BLEU.
            scores.append(sentence_bleu([target_words], pred_words, self.weights))
            #print("TARGET: {}\n".format(target_words))
            #print("PREDICTION: {}\n".format(pred_words))
            #print("BLEU: {}\n".format(scores[-1]))

            
        # Get batch size.
        batch_size = len(targets)

        # Normalize by batch size.
        if batch_size > 0:
            score = sum(scores) / batch_size
        else:
            score = 0

        return score


    def add_statistics(self, stat_col):
        """
        Adds 'accuracy' statistics to ``StatisticsCollector``.

        :param stat_col: ``StatisticsCollector``.

        """
        stat_col.add_statistics(self.key_bleu, '{:6.4f}')

    def collect_statistics(self, stat_col, data_dict):
        """
        Collects statistics (batch_size) for given episode.

        :param stat_col: ``StatisticsCollector``.

        """
        stat_col[self.key_bleu] = self.calculate_BLEU(data_dict)

    def add_aggregators(self, stat_agg):
        """
        Adds aggregator summing samples from all collected batches.

        :param stat_agg: ``StatisticsAggregator``.

        """
        stat_agg.add_aggregator(self.key_bleu, '{:7.5f}')  # represents the average accuracy
        #stat_agg.add_aggregator(self.key_bleu+'_min', '{:7.5f}')
        #stat_agg.add_aggregator(self.key_bleu+'_max', '{:7.5f}')
        stat_agg.add_aggregator(self.key_bleu+'_std', '{:7.5f}')


    def aggregate_statistics(self, stat_col, stat_agg):
        """
        Aggregates samples from all collected batches.

        :param stat_col: ``StatisticsCollector``

        :param stat_agg: ``StatisticsAggregator``

        """
        scores = stat_col[self.key_bleu]

        # Check if batch size was collected.
        if "batch_size" in stat_col.keys():
            batch_sizes = stat_col['batch_size']

            # Calculate weighted precision.
            scores_avg = np.average(scores, weights=batch_sizes)
            scores_var = np.average((scores-scores_avg)**2, weights=batch_sizes)

            stat_agg[self.key_bleu] = scores_avg
            #stat_agg[self.key_bleu+'_min'] = np.min(scores)
            #stat_agg[self.key_bleu+'_max'] = np.max(scores)
            stat_agg[self.key_bleu+'_std'] = math.sqrt(scores_var)
        else:
            # Else: use simple mean.
            stat_agg[self.key_bleu] = np.mean(scores)
            #stat_agg[self.key_bleu+'_min'] = np.min(scores)
            #stat_agg[self.key_bleu+'_max'] = np.max(scores)
            stat_agg[self.key_bleu+'_std'] = np.std(scores)
            # But inform user about that!
            self.logger.warning("Aggregated statistics might contain errors due to the lack of information about sizes of aggregated batches")
