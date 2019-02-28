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

__author__ = "Tomasz Kornuta, Younes Bouhadjar, Vincent Marois"
import torch
import numpy as np
import torch.nn as nn

from ptp.components.problems.problem import Problem

class ImageToClassProblem(Problem):
    """
    Abstract base class for image classification problems.

    Problem classes like MNIST & CIFAR10 inherits from it.

    Provides some basic features useful in all problems of such type.

    """

    def __init__(self, name, params):
        """
        Initializes problem.

        :param name: Problem name.
        :type name: str

        :param params: Dictionary of parameters (read from the configuration ``.yaml`` file).
        :type params: :py:class:`ptp.utils.ParamInterface`
        """
        # Call base class constructors.
        super(ImageToClassProblem, self).__init__(name, params)

        # Set default key mappings.
        self.key_inputs = self.mapkey("inputs")
        self.key_targets = self.mapkey("targets")



    def calculate_accuracy(self, data_dict):
        """
        Calculates accuracy equal to mean number of correct classification in a given batch.

        :param data_dict: DataDict containing the targets.
        :type data_dict: DataDict

        :return: Accuracy.

        """

        # Get the index of the max log-probability.
        #pred = logits.max(1, keepdim=True)[1]
        #correct = pred.eq(data_dict['targets'].view_as(pred)).sum().item()

        # Calculate the accuracy.
        #batch_size = logits.size(0)
        #accuracy = correct / batch_size

        #return accuracy
        pass

    def add_statistics(self, stat_col):
        """
        Add accuracy statistic to ``StatisticsCollector``.

        :param stat_col: ``StatisticsCollector``.

        """
        #stat_col.add_statistic('acc', '{:12.10f}')
        pass

    def collect_statistics(self, stat_col, data_dict):
        """
        Collects accuracy.

        :param stat_col: ``StatisticsCollector``.

        :param data_dict: DataDict containing the targets and the mask.
        :type data_dict: DataDict

        """
        #stat_col['acc'] = self.calculate_accuracy(data_dict)
        pass

    def add_aggregators(self, stat_agg):
        """
        Adds problem-dependent statistical aggregators to ``StatisticsAggregator``.

        :param stat_agg: ``StatisticsAggregator``.

        """
        #stat_agg.add_aggregator('acc', '{:12.10f}')  # represents the average accuracy
        #stat_agg.add_aggregator('acc_min', '{:12.10f}')
        #stat_agg.add_aggregator('acc_max', '{:12.10f}')
        #stat_agg.add_aggregator('acc_std', '{:12.10f}')
        pass

    def aggregate_statistics(self, stat_col, stat_agg):
        """
        Aggregates the statistics collected by ``StatisticsCollector'' and adds the results to ``StatisticsAggregator``.

        :param stat_col: ``StatisticsCollector``.

        :param stat_agg: ``StatisticsAggregator``.

        """
        #stat_agg['acc_min'] = min(stat_col['acc'])
        #stat_agg['acc_max'] = max(stat_col['acc'])
        #stat_agg['acc'] = torch.mean(torch.tensor(stat_col['acc']))
        #stat_agg['acc_std'] = 0.0 if len(stat_col['acc']) <= 1 else torch.std(torch.tensor(stat_col['acc']))
        pass

    def show_sample(self, data_dict, sample_number=0):
        """
        Shows a sample from the batch.

        :param data_dict: ``DataDict`` containing inputs and targets.
        :type data_dict: DataDict

        :param sample_number: Number of sample in batch (default: 0)
        :type sample_number: int

        """
        import matplotlib.pyplot as plt

        # Unpack dict.
        images, targets, labels = data_dict.values()

        # Get sample.
        image = images[sample_number].cpu().detach().numpy()
        target = targets[sample_number].cpu().detach().numpy()
        label = labels[sample_number]

        # Reshape image.
        if image.shape[0] == 1:
            # This is a single channel image - get rid of this dimension
            image = np.squeeze(image, axis=0)
        else:
            # More channels - move channels to axis2, according to matplotilb documentation.
            # (X : array_like, shape (n, m) or (n, m, 3) or (n, m, 4))
            image = image.transpose(1, 2, 0)

        # Show data.
        plt.title('Target class: {} ({})'.format(label, target))
        plt.imshow(image, interpolation='nearest', aspect='auto')

        # Plot!
        plt.show()
