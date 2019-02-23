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

import ptp

from ptp.utils.param_interface import ParamInterface
from ptp.utils.app_state import AppState
from ptp.utils.pipeline import Pipeline

class TestPipeline(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestPipeline, self).__init__(*args, **kwargs)
        # Set required globals.
        app_state = AppState()
        app_state.__setitem__("input_size", 10, override=True)
 
    def test_create_component_full_type(self):
        """ Tests whether component can be created when using full module name with 'path'. """
        # Instantiate.
        params = ParamInterface()
        params.add_default_params({
            'bow_encoders' : 
                {
                    'type': 'ptp.text.bow_encoder.BOWEncoder'
                }
            })
        # Build object.
        pipe = Pipeline(params)
        pipe.build()

        # Assert type.
        self.assertEqual(type(pipe.components[0]).__name__, "BOWEncoder")


    def test_create_component_type(self):
        """ Tests whether component can be created when using only module name. """
        # Instantiate.
        params = ParamInterface()
        params.add_default_params({
            'bow_encoders' : 
                {
                    'type': 'BOWEncoder'
                }
            })
        # Build object.
        pipe = Pipeline(params)
        pipe.build()

        # Assert type.
        self.assertEqual(type(pipe.components[0]).__name__, "BOWEncoder")


if __name__ == "__main__":
    unittest.main()