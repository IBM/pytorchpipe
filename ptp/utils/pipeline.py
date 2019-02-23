#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) IBM tkornuta, Corporation 2019
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
from ptp.utils.io_utils import get_project_root


class Pipeline(object):
    """
    Class responsible for instantiating the pipeline consisting of several components.
    """


    def __init__(self, params):
        """
        Initializes problem object.

        :param params: Parameters used to instantiate all components.
        :type params: ``utils.param_interface.ParamInterface``
        """
        self.params = params
        # Initialize the logger.
        self.logger = logging.getLogger("Pipeline")

        # Set initial values of all pipeline elements.
        # Single problem.
        self.problem = None
        # Empty list of all components, sorted by their priorities.
        self.__components = {}
        # Empty list of all models - it will contain only "references" to objects stored in the components list.
        self.models = []
        # Empty list of all losses - it will contain only "references" to objects stored in the components list.
        self.losses = []
        # Empty list of all loss keys. Those keys will be used to find objects that will be roots for backpropagation of gradients.
        #self.loss_keys = []


    def build(self, log_errors=True):
        """
        Method creating the pipeline, consisting of:
            - a list components ordered by the priority (dictionary).
            - problem (as a separate "link" to object in the list of components, instance of a class derrived from Problem class)
            - models (separate list with link to objects in components dict)
            - losses (selarate list with links to objects in components dict)

        :param log_errors: Logs the detected errors (DEFAULT: TRUE)

        :return: number of detected errors.
        """
        errors = 0

        # Check "skip" section.
        sections_to_skip = "skip optimizer gradient_clipping terminal_conditions seed_torch seed_numpy".split()
        if "skip" in self.params:
            # Expand.
            sections_to_skip = [*sections_to_skip, *self.params["skip"].split(",")]

        for c_key, c_params in self.params.items():
            # The section "key" will be used as "component" name.

            # Skip "special" sections.
            if c_key in sections_to_skip:
                self.logger.info("Skipping section {}".format(c_key))
                continue
    
            # Check presence of type.
            if 'type' not in c_params:
                if log_errors:
                    self.logger.error("Section {} does not contain the key 'type' defining the component type".format(c_key))
                errors += 1
                continue

            # Get the class type.
            c_type = c_params["type"]

            # Get class object.
            if c_type.find("ptp.") != -1:
                # Try to evaluate it directly.
                class_obj = eval(c_type)
            else:
                # Try to find it in the main "ptp" namespace.
                class_obj = getattr(ptp, c_type)

            # Check if class is derived (even indirectly) from Component.
            c_inherits = False
            for c in inspect.getmro(class_obj):
                if c.__name__ == ptp.Component.__name__:
                    c_inherits = True
                    break
            if not c_inherits:
                if log_errors:
                    self.logger.warning("The specified class '{}' is not derived from the Component class".format(c_type))
                errors += 1
                continue

            # Instantiate component.
            component = class_obj(c_key, c_params)

            # Check if class is derived (even indirectly) from Problem.
            p_inherits = False
            for c in inspect.getmro(class_obj):
                if c.__name__ == ptp.Problem.__name__:
                    p_inherits = True
                    break
            if p_inherits:
                if self.problem == None:
                    # Perfect!
                    self.problem = component
                    # Do not add it to list of components!
                    continue
                else:
                    # Oo, two problems?
                    if log_errors:
                        self.logger.error("Pipeline definition contains more than one problem! (here: {}, {})".format(
                            self.problem.name, c_key))
                    errors += 1
                    continue

            # Check presence of priority.
            if 'priority' not in c_params:
                if log_errors:
                    self.logger.error("Section {} does not contain the key 'priority' defining the pipeline order".format(c_key))
                errors += 1
                continue

            # Get the priority.
            try:
                c_priority = float(c_params["priority"])
            except ValueError:
                if log_errors:
                    self.logger.error("Priority {} in section {} is not a floating point number".format(c_params["priority"], c_key))
                errors += 1
                continue

            # Check uniqueness of the priority.
            if c_priority in self.__components.keys():
                if log_errors:
                    self.logger.error("Found more than one component with the same priority ({})".format(c_priority))
                errors += 1
                continue

            # Add it to dict.
            self.__components[c_priority] = component

            # Check if class is derived (even indirectly) from Model.
            m_inherits = False
            for c in inspect.getmro(class_obj):
                if c.__name__ == ptp.Model.__name__:
                    m_inherits = True
                    break
            if m_inherits:
                # Add to list.
                self.models.append(component)

            # Check if class is derived (even indirectly) from Loss.
            l_inherits = False
            for c in inspect.getmro(class_obj):
                if c.__name__ == ptp.Loss.__name__:
                    l_inherits = True
                    break
            if l_inherits:
                # Add to list.
                self.losses.append(component)

        # List of priorities.
        self.__priorities=sorted(self.__components.keys())        

        # Return detected errors.
        return errors


    def __getitem__(self, number):
        """
        Returns the component, using the enumeration resulting from priorities.

        :param number: Number of the component in the pipeline.
        :type key: str

        :return: object of type :py:class:`Component`.

        """
        return self.__components[self.__priorities[number]]


    def __len__(self):
        """
        Returns the number of objects in the pipeline (all components + 1 for problem)
        :return: Length of the :py:class:`Pipeline`.

        """
        length = len(self.__priorities) 
        if self.problem is not None:
            length += 1
        return length
