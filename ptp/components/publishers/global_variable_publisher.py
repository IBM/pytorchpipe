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

from ptp.components.component import Component
from ptp.configuration.configuration_error import ConfigurationError

class GlobalVariablePublisher(Component):
    """
    Component responsible for publishing variables set in configuration file as globals.

    """

    def __init__(self, name, config):
        """
        Initializes object. Loads keys and values of variables and adds them to globals.

        :param name: Loss name.
        :type name: str

        :param config: Dictionary of parameters (read from the configuration ``.yaml`` file).
        :type config: :py:class:`ptp.configuration.ConfigInterface`

        """
        # Call constructors of parent classes.
        Component.__init__(self, name, GlobalVariablePublisher, config)


        # Get list of keys of global variables - can be both list of strings or a single string with comma-separated values.
        keys = self.config["keys"]
        if type(keys) is str:
            keys = keys.replace(" ","").split(",")
        # Get list of values - must be a single value or a list.
        values = self.config["values"]

        
        if type(values) is list:
            # Make sure that both are lists.
            if type(keys) is not list or len(keys) != len(values):
                raise ConfigurationError("Number of parameters indicated by provided 'keys' must be equal to number of provided 'values'")

            # Publish globals one by one.
            for (key, value) in zip(keys, values):
                self.globals[key] = value
        elif keys != '':
            # Publish single global.
            self.globals[keys[0]] = values



    def input_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of input data that are required by the component.

        :return: dictionary containing input data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return { }

    def output_data_definitions(self):
        """ 
        Function returns a empty dictionary with definitions of output data produced the component.

        :return: Empty dictionary.
        """
        return { }


    def __call__(self, data_streams):
        """
        Empty method.

        :param data_streams: :py:class:`ptp.utils.DataStreams` object.

        """
        pass