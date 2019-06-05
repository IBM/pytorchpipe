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
from ptp.application.sampler_factory import SamplerFactory

# Problem.
class TestProblemMockup(object):
    def __len__(self):
        return 50

class TestSamplerFactory(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestSamplerFactory, self).__init__(*args, **kwargs)

    def test_create_subset_random_sampler_range(self):
        """ Tests whther SubsetRandomSampler accepts 'indices' with the option 1: range. """

        indices = range(20)
        config = ConfigInterface()
        config.add_default_params({'type': 'SubsetRandomSampler',
                                'indices': indices})
        # Create the sampler.
        sampler = SamplerFactory.build(TestProblemMockup(), config, "training")

        # Check number of samples.
        self.assertEqual(len(sampler), 20)

    def test_create_subset_random_sampler_range_str(self):
        """ Tests whther SubsetRandomSampler accepts 'indices' with the option 2: range as str. """

        range_str = '0, 20'
        config = ConfigInterface()
        config.add_default_params({'type': 'SubsetRandomSampler',
                                'indices': range_str})
        # Create the sampler.
        sampler = SamplerFactory.build(TestProblemMockup(), config, "training")

        # Check number of samples.
        self.assertEqual(len(sampler), 20)
        

    def test_create_subset_random_sampler_list_of_indices(self):
        """ Tests whther SubsetRandomSampler accepts 'indices' with the option 3: list of indices. """

        yaml_list = yaml.safe_load('[0, 2, 5, 10]')
        config = ConfigInterface()
        config.add_default_params({'type': 'SubsetRandomSampler',
                                'indices': yaml_list})
        # Create the sampler.
        sampler = SamplerFactory.build(TestProblemMockup(), config, "training")

        # Check number of samples.
        self.assertEqual(len(sampler), 4)


    def test_create_subset_random_sampler_file(self):
        """ Tests whther SubsetRandomSampler accepts 'indices' with the option 4: name of the file containing indices. """

        filename = "/tmp/tmp_indices.txt"
        # Store indices to file.
        indices = np.asarray([1,2,3,4,5],dtype=int)
        # Write array to file, separate elements with commas.
        indices.tofile(filename, sep=',', format="%s")

        config = ConfigInterface()
        config.add_default_params({'type': 'SubsetRandomSampler',
                                'indices': filename})
        # Create the sampler.
        sampler = SamplerFactory.build(TestProblemMockup(), config, "training")

        # Check number of samples.
        self.assertEqual(len(sampler), 5)

#if __name__ == "__main__":
#    unittest.main()