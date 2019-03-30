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

from ptp.components.component import Component
from ptp.data_types.data_definition import DataDefinition


class BatchSize(Component):
    """
    Class collecting statistics: batch size.

    """

    def __init__(self, name, config):
        """
        Initializes object.

        :param name: Batch size name.
        :type name: str

        :param config: Dictionary of parameters (read from the configuration ``.yaml`` file).
        :type config: :py:class:`ptp.configuration.ConfigInterface`

        """
        # Call constructors of parent classes.
        Component.__init__(self, name, BatchSize, config)

        # Set key mappings.
        self.key_indices = self.stream_keys["indices"]

    def input_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of input data that are required by the component.

        :return: dictionary containing input data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {
            self.key_indices: DataDefinition([-1, 1], [list, int], "Batch of indices [BATCH_SIZE] x [1]")
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

    def add_statistics(self, stat_col):
        """
        Adds 'batch_size' statistics to ``StatisticsCollector``.

        :param stat_col: ``StatisticsCollector``.

        """
        stat_col.add_statistic('batch_size', '{:06d}')

    def collect_statistics(self, stat_col, data_dict):
        """
        Collects statistics (batch_size) for given episode.

        :param stat_col: ``StatisticsCollector``.

        """
        stat_col['batch_size'] = data_dict[self.key_indices].shape[0] # Batch major.

    def add_aggregators(self, stat_agg):
        """
        Adds aggregator summing number of samples from all collected batches.

        :param stat_agg: ``StatisticsAggregator``.

        """
        stat_agg.add_aggregator('samples_aggregated', '{:006d}')


    def aggregate_statistics(self, stat_col, stat_agg):
        """
        Aggregates number of samples from all collected batches.

        :param stat_col: ``StatisticsCollector``

        :param stat_agg: ``StatisticsAggregator``

        """
        stat_agg['samples_aggregated'] = sum(stat_col['batch_size'])
