# -*- coding: utf-8 -*-
#
# Copyright (C) IBM Corporation, tkornuta 2019
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
from ptp.utils.app_state import AppState

class TestAppState(unittest.TestCase):


    def test_01keys_present(self):
        """ Tests whether the original keys are present and can be retrieved/modified. """
        # Initialize object.
        app_state = AppState()
        # Add global.
        app_state["global1"] = 1 
        # Check its value.
        self.assertEqual(app_state['global1'], 1)

    def test_02keys_present_singleton(self):
        """ Tests whether the original keys are still present in new AppState "instance". """
        # Initialize object.
        app_state = AppState()
        # Check its value.
        self.assertEqual(app_state['global1'], 1)

    def test_03keys_absent(self):
        """ Tests whether absent keys are really absent. """
        with self.assertRaises(KeyError):
            a = AppState()["global2"]

    def test_04keys_overwrite(self):
        """ Tests whether you can overwrite existing key. """
        with self.assertRaises(KeyError):
            AppState()["global1"] = 2
