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

from ptp.configuration.config_registry import ConfigRegistry

class TestConfigRegistry(unittest.TestCase):

    def test_values(self):

    config = ConfigRegistry()

    config.add_default_params({'default_0': {'default_1': 'str'}})
    config.add_config_params({'config_0': {'config_1': 'int'}})

    print(dict(params))
    config.del_config_params(['config_0', 'config_1'])
    print(dict(params))

    config.del_default_params(['default_0', 'default_1'])
    print(dict(params))
