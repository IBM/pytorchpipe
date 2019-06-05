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
import os

from ptp.utils.app_state import AppState
from ptp.configuration.config_interface import ConfigInterface
from ptp.configuration.config_registry import ConfigRegistry
from ptp.application.pipeline_manager import PipelineManager

class TestPipeline(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestPipeline, self).__init__(*args, **kwargs)
        # Set required globals.
        app_state = AppState()
        app_state.__setitem__("bow_size", 10, override=True)
 
    def test_create_component_full_type(self):
        """ Tests whether component can be created when using full module name with 'path'. """
        # Instantiate.
        ConfigRegistry()._clear_registry()
        config = ConfigInterface()
        config.add_default_params({
            'bow_encoder' : 
                {
                    'type': 'ptp.components.text.bow_encoder.BOWEncoder',
                    'priority': 1.1
                }
            })
        # Build object.
        pipe = PipelineManager('testpm', config)
        pipe.build(False)

        # Assert type.
        self.assertEqual(type(pipe[0]).__name__, "BOWEncoder")


    def test_create_component_type(self):
        """ Tests whether component can be created when using only module name. """
        # Instantiate.
        ConfigRegistry()._clear_registry()
        config = ConfigInterface()
        config.add_default_params({
            'bow_encoder' : 
                {
                    'type': 'BOWEncoder',
                    'priority': 1.2
                }
            })
        # Build object.
        pipe = PipelineManager('testpm', config)
        pipe.build(False)

        # Assert type.
        self.assertEqual(type(pipe[0]).__name__, "BOWEncoder")


    def test_disable_component(self):
        """ Tests whether skipping (disable) works properly. """
        # Set param registry.
        ConfigRegistry()._clear_registry()
        config = ConfigInterface()
        config.add_default_params({
            'disable': 'bow_encoder',
            'bow_encoder' : 
                {
                    'type': 'BOWEncoder',
                    'priority': 1
                }
            })
        # Build object.
        pipe = PipelineManager('testpm', config)
        pipe.build(False)

        # Assert no components were created.
        self.assertEqual(len(pipe), 0)


    def test_priorities(self):
        """ Tests component priorities. """
        # Instantiate.
        ConfigRegistry()._clear_registry()
        config = ConfigInterface()
        config.add_default_params({
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
        pipe = PipelineManager('testpm', config)
        pipe.build(False)

        # Assert the right order of components.
        self.assertEqual(len(pipe), 2)
        self.assertEqual(pipe[0].name, 'bow_encoder1')
        self.assertEqual(pipe[1].name, 'bow_encoder2')


#if __name__ == "__main__":
#    unittest.main()