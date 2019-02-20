#!/usr/bin/env python3
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


__author__ = "Alexis Asseman, Tomasz Kornuta"

import torch

from ptp.utils.singleton import SingletonMetaClass
from ptp.utils.data_dict import DataDict


class AppState(metaclass=SingletonMetaClass):
    """
    Represents the application state. For now, really naive, only visualization.
    """

    def __init__(self):
        """
        Constructor:

            - Disable visualization by default,
        """
        # Disable visualization by default.
        self.visualize = False
        # Field storing global variables.
        self.__globals = dict()


    def __setitem__(self, key, value):
        """
        Adds global variable. 

        :param key: Dict Key.

        :param value: Associated value.


        .. warning::
            Once global variable is set, its value cannot be changed (it becomes immutable).
        """
        if key in self.__globals.keys():
            msg = 'Cannot add or modify key "{}" as it is already present in global variables'.format(key)
            raise KeyError(msg)
        else:
            self.__globals[key] = value


    def __getitem__(self, key):
        """
        Value getter function.

        :param key: Dict Key.

        :return: Associated Value.
        """
        if key not in self.__globals.keys():
            msg = 'Key "{}" not present in global variables'.format(key)
            raise KeyError(msg)
        else:
            return self.__globals[key]


if __name__ == '__main__':

    app_state = AppState()

    app_state["global1"] = 1 
    app_state["global2"] = 2

    print(app_state["global1"])

    #print(repr(app_state.__globals))