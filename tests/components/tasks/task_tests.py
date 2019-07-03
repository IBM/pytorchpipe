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

from ptp.components.tasks.task import Task
from ptp.data_types.data_definition import DataDefinition
from ptp.configuration.config_interface import ConfigInterface


class MockupTask (Task):
    """
    Mockup task class.
    """
    def __init__(self, name, config):
        Task.__init__(self, name, None, config)

    def output_data_definitions(self):
        return {
            "inputs": DataDefinition([-1, 1], [list, int], "inputs"),
            "targets": DataDefinition([-1, -1, -1], [list, list, str], "targets")
            }


class TestTask(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestTask, self).__init__(*args, **kwargs)

        # Overwrite abc abstract methods.
        MockupTask.__abstractmethods__=set()
        # Create mocked-up task.
        config = ConfigInterface()
        self.task = MockupTask("test", config)

    def test_crete_data_streams_key_present(self):
        """ Tests whether the created data dict contains required keys. """
        data_streams = self.task.create_data_streams(1)
        # Check presence of index.
        self.assertEqual(data_streams['indices'], 1) # Even if we didn't explicitly indicated that in definitions!
        self.assertEqual(data_streams['inputs'], None)
        self.assertEqual(data_streams['targets'], None)

#if __name__ == "__main__":
#    unittest.main()