#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) IBM Corporation 2018
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

import os
import yaml

from ptp.configuration.app_state import AppState


def load_default_configuration_file(class_type):
    """
    Function loads default configuration from the default config file associated with the given class type and adds it to parameter registry.

    :param class_type: Class type of a given object.

    :raturn: Loaded default configuration.
    """
    
    # Extract path to default config.
    module = class_type.__module__.replace(".","/")
    rel_path = module[module.find("ptp")+4:]
    # Build the abs path to the default config file of a given component.
    abs_default_config = AppState().absolute_config_path + "default/" + rel_path + ".yml"

    # Check if file exists.
    if not os.path.isfile(abs_default_config):
        print("ERROR: The default configuration file '{}' for '{}' component does not exist".format(abs_default_config, class_type.__module__))
        exit(-1)

    try:
        # Open file and get parameter dictionary.
        with open(abs_default_config, 'r') as stream:
            param_dict = yaml.safe_load(stream)

        # Return default parameters so they can be added to the global registry.
        if param_dict is None:
                print("WARNING: The default configuration file '{}' is empty!".format(abs_default_config))
                return {}
        else:
            return param_dict

    except yaml.YAMLError as e:
        print("ERROR: Couldn't properly parse the {} default configuration file. YAML error:\n  {}".format(abs_default_config, e))
        exit(-2)
