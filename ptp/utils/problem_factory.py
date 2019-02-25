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

from ptp.utils.component_factory import ComponentFactory
from ptp.utils.configuration_error import ConfigurationError



class ProblemFactory(object):
    """
    Class instantiating problems using the passed params.
    """

    @staticmethod
    def build(name, params, log_errors=True):
        """
        Method creates a problem on the basis of configuration section.

        :param name: Name of the section/component.

        :param params: Parameters used to instantiate the problem class.
        :type params: ``utils.param_interface.ParamInterface``

        :param log_errors: Logs the detected errors (DEFAULT: TRUE)

        :return: problem object (or None when faced errors)
        """
        try: 
            # Create component.
            component, class_obj = ComponentFactory.build(name, params)

            # Check if class is derived (even indirectly) from Problem.
            if not ComponentFactory.check_inheritance(class_obj, ptp.Problem.__name__):
                raise ConfigurationError("Class '{}' is not derived from the Problem class!".format(class_obj.__name__))            

            # Return component.
            return component

        except ConfigurationError as e:
            if log_errors:
                self.logger.error(e)
            # Return None, i.e. error.
            return None
