# -*- coding: utf-8 -*-
#
# Copyright (C) IBM Corporation 2019
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

from ptp.utils.app_state import AppState

class GlobalsFacade(object):
    """
    Simple facility for accessing global variables using provided mappings using list-like read-write access.
    """
    def __init__(self, key_mappings):
        """
        Constructor. Initializes app state and stores key mappings.

        :param key_mappings: Dictionary of global key mappings of the parent object.
        """
        # Remember parent object global keys mappings.
        self.key_mappings = key_mappings
        self.app_state = AppState()

    def __setitem__(self, key, value):
        """
        Sets global value using parent object key mapping.

        :param key: Global key name (that will be mapped).

        :param value: Value that will be set.
        """
        # Retrieve key using parent object global key mappings.
        mapped_key = self.key_mappings.get(key, key)
        # Set global balue.
        self.app_state[mapped_key] = value


    def __getitem__(self, key):
        """
        Global value getter function.
        Uses parent object key mapping for accesing the value.

        :param key: Global key name (that will be mapped).

        :return: Associated Value.
        """
        # Retrieve key using parent object global key mappings.
        mapped_key = self.key_mappings.get(key, key)
        # Retrieve the value.
        return self.app_state[mapped_key]