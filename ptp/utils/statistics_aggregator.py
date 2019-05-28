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

__author__ = "Vincent Marois, Tomasz Kornuta"

import numpy as np
from ptp.utils.statistics_collector import StatisticsCollector


class StatisticsAggregator(StatisticsCollector):
    """
    Specialized class used for the computation of several statistical aggregators.

    Inherits from :py:class:`ptp.utils.StatisticsCollector` as it extends its capabilities: it relies \
    on :py:class:`ptp.utils.StatisticsCollector` to collect the statistics over an epoch (training set) \
    or a validation (over the validation set).

    Once the statistics have been collected, the :py:class:`ptp.utils.StatisticsAggregator` allows \
    to compute several statistical aggregators to summarize the last epoch or validation phase.

    E.g. With the list of loss values from the last epoch, we can compute the average loss, the min & max, \
    and the standard deviation.


    """

    def __init__(self):
        """
        Constructor for the :py:class:`ptp.utils.StatisticsAggregator`. Defines empty aggregators dict.

        Other statistical aggregators can be added via :py:func:`StatisticsAggregator.add_aggregator()`.

        """
        # call base constructor
        super(StatisticsAggregator, self).__init__()

        self.aggregators = dict()

    def add_aggregator(self, key, formatting):
        """
        Add a statistical aggregator.
        The value associated to the specified key is initiated as -1.

        :param key: Statistical aggregator to add. Such aggregator (e.g. min, max, mean, std...)\
         should be based on an existing statistics collected by the :py:class:`ptp.utils.StatisticsCollector` \
         (e.g. added by :py:func:`StatisticsCollector.add_statistics()` and collected by \
         :py:func:`ptp.models.Model.collect_statistics()` or \
         :py:func:`ptp.models.Problem.collect_statistics()`.
        :type key: str

        :param formatting: Formatting that will be used when logging and exporting to CSV.
        :type formatting: str

        """
        self.formatting[key] = formatting

        # instantiate associated value as list.
        self.aggregators[key] = -1

    def __getitem__(self, key):
        """
        Get the values list of the specified statistical aggregator.

        :param key: Name of the statistical aggregator to get the values list of.
        :type key: str

        :return: Values list associated with the specified statistical aggregator.

        """
        return self.aggregators[key]

    def __setitem__(self, key, value):
        """
        Set the value of the specified statistical aggregator, thus overwriting the existing one.

        :param key: Name of the statistical aggregator to set the value of.
        :type key: str

        :param value: Value to set for the given key.
        :type value: int, float

        """
        self.aggregators[key] = value

    def __delitem__(self, key):
        """
        Delete the specified statistical aggregator.

        :param key: Key to be deleted.
        :type key: str

        """
        del self.aggregators[key]

    def __len__(self):
        """
        Returns the number of tracked statistical aggregators.
        """
        return self.aggregators.__len__()

    def __eq__(self, other):
        """
        Check whether two aggregators are equal (just for the purpose of compatibility with the base Mapping class).
        """
        if isinstance(other, self.__class__):
            # Check statistics, formatting and aggregators.
            return self.statistics == other.statistics and self.formatting == other.formatting and self.aggregators == other.aggregators
        else:
            return False

    def __iter__(self):
        """
        Return an iterator on the currently tracked statistical aggregators.
        """
        return self.aggregators.__iter__()


    def initialize_csv_file(self, log_dir, filename):
        """
        This method creates a new `csv` file and initializes it with a header produced \
        on the base of the statistical aggregators names.

        :param log_dir: Path to file.
        :type log_dir: str

        :param filename: Filename to be created.
        :type filename: str

        :return: File stream opened for writing.

        """
        return self.__initialize_csv_file(log_dir, filename, self.aggregators.keys())


    def export_to_csv(self, csv_file=None):
        """
        This method writes the current statistical aggregators values to the `csv_file` using the associated formatting.

        :param csv_file: File stream opened for writing, optional.

        """
        # Try to use the remembered one.    
        if csv_file is None:
            csv_file = self.csv_file
        # If it is still None - well, we cannot do anything more.
        if csv_file is None:
            return

        values_str = ''

        # Iterate through values and concatenate them.
        for key, value in self.aggregators.items():
            # If formatting is set to None - ignore this key.
            if self.formatting.get(key) is not None:
                # Get formatting - using '{}' as default.
                format_str = self.formatting.get(key, '{}')

                # Add value to string using formatting.
                values_str += format_str.format(value) + ","

        # Remove last coma.
        if len(values_str) > 1:
            values_str = values_str[:-1]
        # Add last \n.
        values_str = values_str + '\n'

        csv_file.write(values_str)

    def export_to_checkpoint(self):
        """
        This method exports the aggregated data into a dictionary using the associated formatting.

        """
        chkpt = {}

        # Iterate through key, values and format them.
        for key, value in self.aggregators.items():
            # If formatting is set to None - ignore this key.
            if self.formatting.get(key) is not None:
                # Get formatting - using '{}' as default.
                format_str = self.formatting.get(key, '{}')

                # Add to dict.
                chkpt[key] = format_str.format(value)

        return chkpt        

    def export_to_string(self, additional_tag=''):
        """
        This method returns the current statistical aggregators values in the form of a string using the \
        associated formatting.

        :param additional_tag: An additional tag to append at the end of the created string.
        :type additional_tag: str


        :return: String being the concatenation of the statistical aggregators names & values.

        """
        stat_str = ''

        # Iterate through keys and values and concatenate them.
        for key, value in self.aggregators.items():
            # If formatting is set to None - ignore this key.
            if self.formatting.get(key) is not None:
                stat_str += key + ' '
                # Get formatting - using '{}' as default.
                format_str = self.formatting.get(key, '{}')
                # Add value to string using formatting.
                stat_str += format_str.format(value) + "; "

        # Remove last two elements.
        if len(stat_str) > 2:
            stat_str = stat_str[:-2]
        
        # Add addtional tag.
        stat_str = stat_str + " " + additional_tag

        return stat_str

    def export_to_tensorboard(self, tb_writer = None):
        """
        Method exports current statistical aggregators values to TensorBoard.

        :param tb_writer: TensorBoard writer, optional
        :type tb_writer: :py:class:`tensorboardX.SummaryWriter`

        """
        # Get episode number.
        episode = self.aggregators['episode']

        if tb_writer is None:
            tb_writer = self.tb_writer
        # If it is still None - well, we cannot do anything more.
        if tb_writer is None:
            return

        # Iterate through keys and values and concatenate them.
        for key, value in self.aggregators.items():
            # Skip episode.
            if key == 'episode':
                continue
            # If formatting is set to None - ignore this key.
            if self.formatting.get(key) is not None:
                tb_writer.add_scalar(key, value, episode)


if __name__ == "__main__":

    stat_col = StatisticsCollector()
    stat_agg = StatisticsAggregator()


    # Add default statistics with formatting.
    stat_col.add_statistics('loss', '{:12.10f}')
    stat_col.add_statistics('episode', '{:06d}')
    stat_col.add_statistics('batch_size', None)

    import random
    # create some random values
    loss_values = random.sample(range(100), 100)
    # "Collect" basic statistics.
    for episode, loss in enumerate(loss_values):
        stat_col['episode'] = episode
        stat_col['loss'] = loss
        stat_col['batch_size'] = 1
        # print(stat_col.export_statistics_to_string())

    print(stat_agg.export_to_string())

    # Add new aggregator (a simulation of "additional statistics collected by model")
    # Add default statistical aggregators for the loss (indicating a formatting).
    #stat_agg.add_aggregator('loss', '{:12.10f}')  
    # add 'aggregators' for the episode.
    #stat_agg.add_aggregator('episode', '{:06d}')
    # Number of aggregated episodes.
    #stat_agg.add_aggregator('episodes_aggregated', '{:06d}')
    stat_agg.add_aggregator('acc_mean', '{:2.5f}')
    collected_loss_values  = stat_col['loss']
    batch_sizes = stat_col['batch_size']
    stat_agg['acc_mean'] = np.mean(collected_loss_values) / np.sum(batch_sizes)

    print(stat_agg.export_to_string('[Epoch 1]'))
