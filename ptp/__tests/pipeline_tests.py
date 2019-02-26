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

from ptp.configuration.param_interface import ParamInterface
from ptp.configuration.param_registry import ParamRegistry
from ptp.configuration.app_state import AppState
from ptp.configuration.pipeline_manager import PipelineManager

class TestPipeline(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestPipeline, self).__init__(*args, **kwargs)
        # Set required globals.
        app_state = AppState()
        app_state.__setitem__("input_size", 10, override=True)
        
 
    def test_create_component_full_type(self):
        """ Tests whether component can be created when using full module name with 'path'. """
        # Instantiate.
        ParamRegistry()._clear_registry()
        params = ParamInterface()
        params.add_default_params({
            'bow_encoder' : 
                {
                    'type': 'ptp.text.bow_encoder.BOWEncoder',
                    'priority': 1.1
                }
            })
        # Build object.
        pipe = PipelineManager('testpm', params)
        pipe.build(False)

        # Assert type.
        self.assertEqual(type(pipe[0]).__name__, "BOWEncoder")


    def test_create_component_type(self):
        """ Tests whether component can be created when using only module name. """
        # Instantiate.
        ParamRegistry()._clear_registry()
        params = ParamInterface()
        params.add_default_params({
            'bow_encoder' : 
                {
                    'type': 'BOWEncoder',
                    'priority': 1.2
                }
            })
        # Build object.
        pipe = PipelineManager('testpm', params)
        pipe.build(False)

        # Assert type.
        self.assertEqual(type(pipe[0]).__name__, "BOWEncoder")


    def test_skip_section(self):
        """ Tests whether skipping works properly. """
        # Set param registry.
        ParamRegistry()._clear_registry()
        params = ParamInterface()
        params.add_default_params({
            'skip': 'bow_encoder',
            'bow_encoder' : 
                {
                    'type': 'BOWEncoder',
                    'priority': 1
                }
            })
        # Build object.
        pipe = PipelineManager('testpm', params)
        pipe.build(False)

        # Assert no components were created.
        self.assertEqual(len(pipe), 0)


    def test_priorities(self):
        """ Tests component priorities. """
        # Instantiate.
        ParamRegistry()._clear_registry()
        params = ParamInterface()
        params.add_default_params({
            'bow_encoder2' : 
                {
                    'type': 'BOWEncoder',
                    'priority': 2.1
                },
            'bow_encoder1' : 
                {
                    'type': 'BOWEncoder',
                    'priority': 0.1
                }
            })
        pipe = PipelineManager('testpm', params)
        pipe.build(False)

        # Assert the right order of components.
        self.assertEqual(len(pipe), 2)
        self.assertEqual(pipe[0].name, 'bow_encoder1')
        self.assertEqual(pipe[1].name, 'bow_encoder2')


#if __name__ == "__main__":
#    unittest.main()