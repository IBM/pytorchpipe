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

class KeyMappingsFacade(object):
    """
    Simple facility for accessing key names using provided mappings using list-like (read-only) access.
    """
    def __init__(self, key_mappings):
        """
        Constructor. Stores key mappings.

        :param key_mappings: Dictionary of key mappings of the parent object.
        """
        # Remember parent object global keys.
        self.keys_mappings = key_mappings

    def __getitem__(self, key):
        """
        Global value getter function.
        Uses parent object key mapping for accesing the value.

        :param key: Global key name (that will be mapped).

        :return: Associated Value.
        """
        # Retrieve key using parent object global key mappings.
        return self.keys_mappings.get(key, key)
