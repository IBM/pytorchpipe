#!/usr/bin/env python3
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
from unittest.mock import patch

from ptp.core_types.component import Component
from ptp.core_types.data_definition import DataDefinition
from ptp.utils.param_interface import ParamInterface


class MockupComponent (Component):
    """
    Mockup component class.
    """
    def __init__(self):
        Component.__init__(self, "MockupComponent", ParamInterface())

    def input_data_definitions(self):
        return {
            "input1": DataDefinition([-1, 1], [list, int], "comment1"),
            "input2": DataDefinition([-1, -1, -1], [list, list, str], "comment2"),
            "input3": DataDefinition([-1, -1], [float], "comment3")
            }

    def output_data_definitions(self):
        return {
            "output": DataDefinition([1], [int], "Value (scalar)")
            }


class TestHandshaking(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestHandshaking, self).__init__(*args, **kwargs)

        # Overwrite abc abstract methods.
        MockupComponent.__abstractmethods__=set()
        # Create mockup component
        self.component = MockupComponent()


    def test_handshake_input_definitions_keys(self):
        """ Tests handskaking of input definition keys. """
        # Both inputs missing.
        all_defs = {}
        self.assertEqual(self.component.handshake_input_definitions( all_defs, log_errors=False ), 3)

        # One input missing.
        all_defs["input1"] = DataDefinition([-1, 1], [list, int], "comment")
        self.assertEqual(self.component.handshake_input_definitions( all_defs, log_errors=False ), 2)

        # All inputs ok.
        all_defs["input2"] = DataDefinition([-1, -1, -1], [list, list, str], "comment2")
        all_defs["input3"] = DataDefinition([-1, -1], [float], "comment3")
        self.assertEqual(self.component.handshake_input_definitions( all_defs, log_errors=False ), 0)


    def test_handshake_input_definitions_dimensions(self):
        """ Tests handskaking of input definition keys. """
        all_defs = {}
        all_defs["input2"] = DataDefinition([-1, -1, -1], [list, list, str], "comment2")
        all_defs["input3"] = DataDefinition([-1, -1], [float], "comment3")

        # One input with wrong number of dimensions.
        all_defs["input1"] = DataDefinition([-1, 10], [list, int], "comment")
        self.assertEqual(self.component.handshake_input_definitions( all_defs, log_errors=False ), 1)

        # One input with wrong number of dimensions, but with dynamic size provided while we expect fixed.
        all_defs["input1"] = DataDefinition([-1, -1], [list, int], "comment")
        self.assertEqual(self.component.handshake_input_definitions( all_defs, log_errors=False ), 1)

        # One input with wrong number of dimensions, but it is the "agnostic" one.
        all_defs["input1"] = DataDefinition([1, 1], [list, int], "comment")
        self.assertEqual(self.component.handshake_input_definitions( all_defs, log_errors=False ), 0)


    def test_handshake_input_definitions_types(self):
        """ Tests handskaking of input definition keys. """
        all_defs = {}
        all_defs["input2"] = DataDefinition([-1, -1, -1], [list, list, str], "comment2")
        all_defs["input3"] = DataDefinition([-1, -1], [float], "comment3")

        # One input with wrong number of types.
        all_defs["input1"] = DataDefinition([-1, 1], [list], "comment")
        self.assertEqual(self.component.handshake_input_definitions( all_defs, log_errors=False ), 1)

        # One input with one wrong type.
        all_defs["input1"] = DataDefinition([-1, 1], [list, str], "comment")
        self.assertEqual(self.component.handshake_input_definitions( all_defs, log_errors=False ), 1)


    def test_extension_definitions(self):
        """ Tests extension of output definition keys. """
        all_defs = {} 

        # Key not existing in output definitions - ADD.
        all_defs["output2"] = DataDefinition([-1, -1, -1], [list, list, str], "comment")
        self.assertEqual(self.component.export_output_definitions( all_defs, log_errors=False ), 0)

        # Key already existing in output definitions.
        all_defs["output"] = DataDefinition([-1, -1, -1], [list, list, str], "comment")
        self.assertEqual(self.component.export_output_definitions( all_defs, log_errors=False ), 1)


