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
        # Empty list of all components.
        self.components = []
        # Empty list of all models - it will contain only "references" to objects stored in the components list.
        self.models = []
        # Empty list of all losses - it will contain only "references" to objects stored in the components list.
        self.losses = []


    def build(self):
        """
        Static method creating pipeline:
            - list components ordered by the priority.
            - problem (as a separate "link" to object in the list of components, instance of a class derrived from Problem class)
            - models (separate list )

        """
        # Check "skip" section.
        sections_to_skip = "optimizer gradient_clipping terminal_conditions seed_torch seed_numpy".split()
        if "skip" in self.params:
            # Expand.
            sections_to_skip = [*sections_to_skip, *self.params["skip"].split(",")]

        for c_key, c_params in self.params.items():
            # The section "key" will be used as "component" name.

            # Skip "special" sections.
            if c_key in sections_to_skip:
                self.logger.info("Skipping section {}".format(c_key))
                continue
    
            # Check presence of the type.
            if 'type' not in c_params:
                self.logger.error("Section {} does not contain the key 'type' defining the component type".format(c_key))
                exit(-1)

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
                self.logger.warning("The specified class '{}' is not derived from the Component class".format(c_type))
                continue

            # Instantiate component.
            component = class_obj(c_key, c_params)
            self.components.append(component)

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
                else:
                    # Oo, two problems?
                    self.logger.error("Pipeline cannot contain more than one problem !(here: {}, {})".format(
                        type(self.problem).__name__, c_type))
                    exit(1)

            # Check if class is derived (even indirectly) from Model.
            m_inherits = False
            for c in inspect.getmro(class_obj):
                if c.__name__ == ptp.Model.__name__:
                    m_inherits = True
                    break
            if m_inherits:
                # Add to list.
                self.models.append(component)

        # Returns problem!
        return self.problem
