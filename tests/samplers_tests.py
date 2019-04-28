#!/usr/bin/env python3
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
import yaml
import numpy as np

from ptp.configuration.config_interface import ConfigInterface
from ptp.utils.samplers import kFoldRandomSampler, kFoldWeightedRandomSampler

class TestkFoldRandomSampler(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestkFoldRandomSampler, self).__init__(*args, **kwargs)

    def test_kfold_random_sampler_current_fold(self):
        """ Tests the k-fold sampler current_fold mode. """

        # Create the sampler.
        sampler = kFoldRandomSampler(20, 3, all_but_current_fold=False)

        # Test zero-th fold.
        indices = list(iter(sampler))
        # Check number of samples.
        self.assertEqual(len(indices), 7)
        # Check presence of all indices.
        for ix in range(0,7):
            self.assertIn(ix, indices)

        # Test first fold.
        indices = list(iter(sampler))
        # Check number of samples.
        self.assertEqual(len(indices), 7)
        # Check presence of all indices.
        for ix in range(7,14):
            self.assertIn(ix, indices)

        # Test second fold.
        indices = list(iter(sampler))
        # Check number of samples.
        self.assertEqual(len(indices), 6)
        # Check presence of all indices.
        for ix in range(14,20):
            self.assertIn(ix, indices)

        # Test third (i.e. zero-th for the second time) fold.
        indices = list(iter(sampler))
        # Check number of samples.
        self.assertEqual(len(indices), 7)
        # Check presence of all indices.
        for ix in range(0,7):
            self.assertIn(ix, indices)


    def test_kfold_random_sampler_all_but_current_fold(self):
        """ Tests the k-fold sampler all_but_current_fold mode. """

        # Create the sampler.
        sampler = kFoldRandomSampler(20, 3, all_but_current_fold=True)

        # Test zero-th fold.
        indices = list(iter(sampler))
        # Check number of samples.
        self.assertEqual(len(indices), 13)
        self.assertEqual(len(sampler), 13)
        
        # Check presence of all indices.
        for ix in range(7,20):
            self.assertIn(ix, indices)

        # Test first fold.
        indices = list(iter(sampler))
        # Check number of samples.
        self.assertEqual(len(indices), 13)
        # Check presence of all indices.
        for ix in [*range(0,7), *range(14,20)]:
            self.assertIn(ix, indices)

        # Test second fold.
        indices = list(iter(sampler))
        # Check number of samples.
        self.assertEqual(len(indices), 14)
        # Check presence of all indices.
        for ix in range(0,14):
            self.assertIn(ix, indices)

        # Test third (i.e. zero-th for the second time) fold.
        indices = list(iter(sampler))
        # Check number of samples.
        self.assertEqual(len(indices), 13)
        # Check presence of all indices.
        for ix in range(7,20):
            self.assertIn(ix, indices)


class TestkFoldWeightedRandomSampler(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestkFoldWeightedRandomSampler, self).__init__(*args, **kwargs)

    def test_kfold_weighed_random_sampler_current_fold(self):
        """ Tests the k-fold sampler current_fold mode. """
        # Non-uniform weights.
        weights = [0.5]*2 + [0] * 2 + [0.5] + [0] * 2 +[0.5]

        # Create the sampler.
        sampler = kFoldWeightedRandomSampler(weights, 8, 2, all_but_current_fold=False)

        # Test zero-th fold.
        indices = list(iter(sampler))
        #print(indices)

        # Check that the rights indices are there.
        for ix in indices:
            self.assertIn(ix, [0,1])

        # Test first fold.
        indices = list(iter(sampler))
        #print(indices)

        # Check that the rights indices are there.
        for ix in indices:
            self.assertIn(ix, [4,7])


if __name__ == "__main__":
    unittest.main()