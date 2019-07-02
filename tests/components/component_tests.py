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

from ptp.components.component import Component
from ptp.components.problems.problem import Problem
from ptp.data_types.data_definition import DataDefinition
from ptp.configuration.config_interface import ConfigInterface

class MockupProblem (Problem):
    """
    Mockup problem class.
    """
    def __init__(self, name, config):
        Problem.__init__(self, name, None, config)

    def output_data_definitions(self):
        return {
            "inputs": DataDefinition([-1, 1], [list, int], "inputs"),
        }


class MockupComponent (Component):
    """
    Mockup component class.
    """
    def __init__(self, name, config):
        Component.__init__(self, name, None, config)


class TestComponent(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestComponent, self).__init__(*args, **kwargs)

        # Overwrite abc abstract methods.
        MockupComponent.__abstractmethods__=set()
        MockupProblem.__abstractmethods__=set()

        # Create mocked-up component.
        config = ConfigInterface()
        self.problem = MockupProblem("test_problem", config)
        self.component = MockupComponent("test_component", config)

    def test_create_data_streams_key_present(self):
        """ Tests whether the created data dict contains required keys. """
        data_streams = self.problem.create_data_streams(1)
        # Check presence of index.
        self.assertEqual(data_streams['indices'], 1) # Even if we didn't explicitly indicated that in definitions!
        self.assertEqual(data_streams['inputs'], None)
        # And targets is not present (yet)...
        with self.assertRaises(KeyError):
            data_streams['targets']

    def test_extend_data_streams_key_present(self):
        """ Tests whether the created data dict contains required keys. """
        data_streams = self.problem.create_data_streams(1)
        # Extend data_streams.
        data_streams.publish({"targets": 3})

        # Check presence of all "streams".
        self.assertEqual(data_streams['indices'], 1) # Even if we didn't explicitly indicated that in definitions!
        self.assertEqual(data_streams['inputs'], None)
        self.assertEqual(data_streams['targets'], 3)

    def test_global_set_get(self):
        """ Tests setting and getting global value. """
        # Set global value.
        self.component.globals["embeddings_size"] = 10
        # Get global value.
        self.assertEqual(self.component.globals["embeddings_size"], 10)

    def test_global_overwrite(self):
        """ Tests global value overwrite """
        # Set global value.
        self.component.globals["value"] = "ala"
        # Overwrite with the same value - ok.
        self.component.globals["value"] = "ala"
        # Overwrite with the same value - error.
        with self.assertRaises(KeyError):
            self.component.globals["value"] = "ola"

