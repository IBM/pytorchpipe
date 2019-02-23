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
from ptp.utils.pipeline import Pipeline
from ptp.utils.param_interface import ParamInterface

class TestPipeline(unittest.TestCase):

    def test_create_component_full_type(self):
        """ Tests whether component can be created when using full module 'path' """

        params = ParamInterface()
        params.add_default_params({
            'bow_encoders' : 
                {
                    'type': 'ptp.text.bow_encoder.BOWEncoder'
                }
            })

        pipe = Pipeline(params)
        pipe.build()
        print(pipe.components[0])

    def test_create_component_type(self):
        import ptp
        #c_name = "ptp.text.bow_encoder.BOWEncoder"
        c_name = "BOWEncoder"

        if c_name.find("ptp.") != -1:
            # Try to evaluate it directly.
            class_obj = eval(c_name)
        else:
            # Try to find it in the main "ptp" namespace.
            class_obj = getattr(ptp, c_name)

        print(class_obj)
