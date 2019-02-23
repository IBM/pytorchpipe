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

        for c_key, c_params in self.params.iteritems():
            # Section "key" will be used as "component" name.

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

            # Instantiate component.
            component = class_obj(c_params)
            #print( isinstance(c_type, types.ClassType) )

            self.components.append(component)

        # Returns problem!
        return None

#class BOWEncoder(object):
#    pass

if __name__ == "__main__":
    #from miprometheus.utils.param_interface import ParamInterface
    #params = ParamInterface()
    #params.add_default_params({
    #    'encoders' : [ # LIST!
    #        {
    #            'name': 'Encoder1'
    #        },{
    #            'name': 'Encoder2'
    #        }
    #        ]
    #    })
    #encoders = EncoderFactory.build(params)
    #for encoder in encoders:
    #    print(type(encoder))

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
