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

import unittest
import random
import numpy as np

from ptp.utils.statistics_collector import StatisticsCollector
from ptp.utils.statistics_aggregator import StatisticsAggregator

class TestStatistics(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestStatistics, self).__init__(*args, **kwargs)

    def test_collector_string(self):
        """ Tests whether the collector is collecting and producing the right string. """

        stat_col = StatisticsCollector()
        stat_col.add_statistics('loss', '{:12.10f}')
        stat_col.add_statistics('episode', '{:06d}')
        stat_col.add_statistics('acc', '{:2.3f}')
        stat_col.add_statistics('acc_help', None)

        # Episode 0.
        stat_col['episode'] = 0
        stat_col['loss'] = 0.7
        stat_col['acc'] = 100
        stat_col['acc_help'] = 121

        # Export.
        #csv_file = stat_col.initialize_csv_file('./', 'collector_test.csv')
        #stat_col.export_to_csv(csv_file)
        self.assertEqual(stat_col.export_to_string(), "loss 0.7000000000; episode 000000; acc 100.000 ")

        # Episode 1.
        stat_col['episode'] = 1
        stat_col['loss'] = 0.7
        stat_col['acc'] = 99.3

        stat_col.add_statistics('seq_length', '{:2.0f}')
        stat_col['seq_length'] = 5

        # Export.
        #stat_col.export_to_csv(csv_file)
        self.assertEqual(stat_col.export_to_string('[Validation]'), "loss 0.7000000000; episode 000001; acc 99.300; seq_length  5 [Validation]")

        # Empty.
        stat_col.empty()
        self.assertEqual(stat_col.export_to_string(), "loss ; episode ; acc ; seq_length  ")

    def test_aggregator_string(self):
        """ Tests whether the collector is aggregating and producing the right string. """

        stat_col = StatisticsCollector()
        stat_agg = StatisticsAggregator()

        # Add default statistics with formatting.
        stat_col.add_statistics('loss', '{:12.10f}')
        stat_col.add_statistics('episode', '{:06d}')
        stat_col.add_statistics('batch_size', None)

        # create some random values
        loss_values = random.sample(range(100), 100)
        # "Collect" basic statistics.
        for episode, loss in enumerate(loss_values):
            stat_col['episode'] = episode
            stat_col['loss'] = loss
            stat_col['batch_size'] = 1
            # print(stat_col.export_statistics_to_string())

        # Empty before aggregation.
        self.assertEqual(stat_agg.export_to_string(), " ")

        # Number of aggregated episodes.
        stat_agg.add_aggregator('acc_mean', '{:2.5f}')
        collected_loss_values  = stat_col['loss']
        batch_sizes = stat_col['batch_size']
        stat_agg['acc_mean'] = np.mean(collected_loss_values) / np.sum(batch_sizes)

        # Aggregated result.
        self.assertEqual(stat_agg.export_to_string('[Epoch 1]'), "acc_mean 0.49500 [Epoch 1]")


#if __name__ == "__main__":
#    unittest.main()