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

import os.path
import logging
import inspect

import ptp

from ptp.configuration.configuration_error import ConfigurationError


class ComponentFactory(object):
    """
    Class instantiating the components using the passed params.
    """

    @staticmethod
    def check_inheritance(class_obj, parent_class_name):
        """
        Checks whether given class inherits (even indirectly) from parent class.
        """
        # Check if class is derived (even indirectly) from Component.
        for c in inspect.getmro(class_obj):
            if c.__name__ == parent_class_name:
                return True
        return False



    @staticmethod
    def build(name, params):
        """
        Method creates a single component on the basis of configuration section.
        Raises ConfigurationError exception when encountered issues.

        :param name: Name of the section/component.

        :param params: Parameters used to instantiate all components.
        :type params: ``utils.param_interface.ParamInterface``

        :return: tuple (component, component class).
        """

        # Check presence of type.
        if 'type' not in params:
            raise ConfigurationError("Section {} does not contain the key 'type' defining the component type".format(name))

        # Get the class type.
        c_type = params["type"]

        # Get class object.
        if c_type.find("ptp.") != -1:
            # Try to evaluate it directly.
            class_obj = eval(c_type)
        else:
            # Try to find it in the main "ptp" namespace.
            class_obj = getattr(ptp, c_type)

        # Check if class is derived (even indirectly) from Component.
        if not ComponentFactory.check_inheritance(class_obj, ptp.Component.__name__):
            raise ConfigurationError("Class '{}' is not derived from the Component class".format(c_type))

        # Instantiate component.
        component = class_obj(name, params)

        return component, class_obj
