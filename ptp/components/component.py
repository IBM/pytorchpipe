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

import abc

import ptp.utils.logger as logging

from ptp.utils.app_state import AppState
from ptp.utils.globals_facade import GlobalsFacade
from ptp.utils.key_mappings_facade import KeyMappingsFacade

from ptp.configuration.config_parsing import load_class_default_config_file


class Component(abc.ABC):
    def __init__(self, name, class_type, config):
        """
        Initializes the component.

        This constructor:

        - stores a pointer to ``config``:

            >>> self.config = config

        - sets a problem name:

            >>> self.name = name

        - initializes the logger.

            >>> self.logger = logging.getLogger(self.name)        

        - sets the access to ``AppState``: for dtype, visualization flag etc.

            >>> self.app_state = AppState()

        :param name: Name of the component.

        :param class_type: Class type of the component.

        :param config: Dictionary of parameters (read from configuration ``.yaml`` file).
        :type config: :py:class:`ptp.configuration.ConfigInterface`

        """
        self.name = name
        self.config = config

        # Get access to AppState: for command line args, globals etc.
        self.app_state = AppState()

        # Initialize logger.
        self.logger = logging.initialize_logger(self.name)        

        # Load default configuration.
        if class_type is not None:
            self.config.add_default_params(load_class_default_config_file(class_type))

        # Initialize the "streams mapping facility".
        if "streams" not in config or config["streams"] is None:
            self.__stream_keys = {}
        else:
            self.__stream_keys = config["streams"]
        self.stream_keys = KeyMappingsFacade(self.__stream_keys)

        # Initialize the "globals mapping facility".
        if "globals" not in config or config["globals"] is None:
            self.__global_keys = {}
        else:
            self.__global_keys = config["globals"]
        self.global_keys = KeyMappingsFacade(self.__global_keys)

        # Initialize the "statistics mapping facility".
        if "statistics" not in config or config["statistics"] is None:
            self.__statistics_keys = {}
        else:
            self.__statistics_keys = config["statistics"]
        self.statistics_keys = KeyMappingsFacade(self.__statistics_keys)

        # Facade for accessing global parameters (stored still in AppState).
        self.globals = GlobalsFacade(self.__global_keys)


    def summarize_io(self, priority = -1):
        """
        Summarizes the component by showing its name, type and input/output definitions.

        :param priority: Component priority (DEFAULT: -1)

        :return: Summary as a str.

        """
        summary_str = "  + {} ({}) [{}] \n".format(self.name, type(self).__name__, priority)
        # Get inputs
        summary_str += '      Inputs:\n' 
        for key,value in self.input_data_definitions().items():
            summary_str += '        {}: {}, {}, {}\n'.format(key, value.dimensions, value.types, value. description)
        # Get outputs.
        summary_str += '      Outputs:\n' 
        for key,value in self.output_data_definitions().items():
            summary_str += '        {}: {}, {}, {}\n'.format(key, value.dimensions, value.types, value. description)
        # Return string.
        return summary_str


    @abc.abstractmethod
    def input_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of input data that are required by the component.
        Abstract, must be implemented by all derived classes.

        :return: dictionary containing input data definitions (each of type :py:class:`ptp.configuration.DataDefinition`).
        """
        pass

    @abc.abstractmethod
    def output_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of output data produced the component.
        Abstract, must be implemented by all derived classes.

        :return: dictionary containing output data definitions (each of type :py:class:`ptp.configuration.DataDefinition`).
        """
        pass

    def handshake_input_definitions(self, all_definitions, log_errors=True):
        """ 
        Checks whether all_definitions contain fields required by the given component.

        :param all_definitions: dictionary containing output data definitions (each of type :py:class:`ptp.configuration.DataDefinition`).

        :param log_errors: Logs the detected errors (DEFAULT: TRUE)

        :return: number of detected errors.
        """
        errors = 0
        for (key,id) in self.input_data_definitions().items():
            # Check presence of key.
            if key not in all_definitions.keys():
                if log_errors:
                    self.logger.error("Input definition: expected field '{}' not found in DataDict keys ({})".format(key, all_definitions.keys()))
                errors += 1
                continue
            # Check number of dimensions.
            dd = all_definitions[key]
            if len(id.dimensions) != len (dd.dimensions):
                if log_errors:
                    self.logger.error("Input definition: field '{}' in DataDict has different dimensions from expected (expected {} while received {})".format(key, id.dimensions, dd.dimensions))
                errors += 1
            else: 
                # Check dimensions one by one.
                for index, (did, ddd) in enumerate(zip(id.dimensions, dd.dimensions)):
                    # -1 means that it can handle different values here.
                    if did != -1 and did != ddd:
                        if log_errors:
                            self.logger.error("Input definition: field '{}' in DataDict has dimension {} different from expected (expected {} while received {})".format(key,index, id.dimensions, dd.dimensions))
                        errors += 1
            # Check number of types.
            if len(id.types) != len (dd.types):
                if log_errors:
                    self.logger.error("Input definition: field '{}' in DataDict has number of types different from expected (expected {} while received {})".format(key, id.types, dd.types))
                errors += 1
            else: 
                # Check types one by one.
                for index, (tid, tdd) in enumerate(zip(id.types, dd.types)):
                    # -1 means that it can handle different values here.
                    if tid != tdd:
                        if log_errors:
                            self.logger.error("Input definition: field '{}' in DataDict has type {} different from expected (expected {} while received {})".format(key,index, id.types, dd.types))
                        errors += 1

        return errors
    
    def export_output_definitions(self, all_definitions, log_errors=True):
        """ 
        Exports output definitions to all_definitions, checking errors (e.g. if output field is already present in all_definitions).

        :param all_definitions: dictionary containing output data definitions (each of type :py:class:`ptp.configuration.DataDefinition`).

        :param log_errors: Logs the detected errors (DEFAULT: TRUE)

        :return: number of detected errors.
        """
        errors = 0
        for (key,od) in self.output_data_definitions().items():
            # Check presence of key.
            if key in all_definitions.keys():
                if log_errors:
                    self.logger.error("Output definition error: field '{}' cannot be added to DataDict, as it is already present in its keys ({})".format(key, all_definitions.keys()))
                errors += 1
            else:
                # Add field to definitions.
                all_definitions[key] = od

        return  errors


    @abc.abstractmethod
    def __call__(self, data_dict):
        """
        Method responsible for processing the data dict.
        Abstract, must be implemented by all derived classes.

        :param data_dict: :py:class:`ptp.core_types.DataDict` object containing both input data to be processed and that will be extended by the results.
        """
        pass


    def add_statistics(self, stat_col):
        """
        Adds statistics to :py:class:`ptp.configuration.StatisticsCollector`.

        .. note::

            Empty - To be redefined in inheriting classes.

        :param stat_col: :py:class:`ptp.configuration.StatisticsCollector`.

        """
        pass


    def collect_statistics(self, stat_col, data_dict):
        """
        Base statistics collection.

         .. note::

            Empty - To be redefined in inheriting classes. The user has to ensure that the corresponding entry \
            in the :py:class:`ptp.configuration.StatisticsCollector` has been created with \
            :py:func:`add_statistics` beforehand.

        :param stat_col: :py:class:`ptp.configuration.StatisticsCollector`.

        :param data_dict: ``DataDict`` containing inputs, targets etc.
        :type data_dict: :py:class:`ptp.core_types.DataDict`

        """
        pass


    def add_aggregators(self, stat_agg):
        """
        Adds statistical aggregators to :py:class:`ptp.configuration.StatisticsAggregator`.

        .. note::

            Empty - To be redefined in inheriting classes.

        :param stat_agg: :py:class:`ptp.configuration.StatisticsAggregator`.

        """
        pass


    def aggregate_statistics(self, stat_col, stat_agg):
        """
        Aggregates the statistics collected by :py:class:`ptp.configuration.StatisticsCollector` and adds the \
        results to :py:class:`ptp.configuration.StatisticsAggregator`.

         .. note::

            Empty - To be redefined in inheriting classes.
            The user can override this function in subclasses but should call \
            :py:func:`aggregate_statistics` to collect basic statistical aggregators (if set).

        :param stat_col: :py:class:`ptp.configuration.StatisticsCollector`.

        :param stat_agg: :py:class:`ptp.configuration.StatisticsAggregator`.

        """
        pass
