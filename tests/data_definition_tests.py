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
from ptp.data_types.data_definition import DataDefinition

class TestDataDefinition(unittest.TestCase):

    def test_values(self):
        """ Tests whether the values are set. """
        dd = DataDefinition([1], [int], "Value")

        self.assertEqual(dd.dimensions, [1])
        self.assertEqual(dd.types,[int])
        self.assertEqual(dd.description, "Value")


    def test_override(self):
        """ Tests whether values cannot be overriden. """
        dd = DataDefinition([1], [int], "Value")

        with self.assertRaises(AttributeError):
            dd.dimensions = [1,2]
        with self.assertRaises(AttributeError):
            dd.types = [str,list]
        with self.assertRaises(AttributeError):
            dd.description = "New Description"

