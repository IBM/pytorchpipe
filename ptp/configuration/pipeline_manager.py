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


import logging
import inspect

import ptp

from ptp.configuration.configuration_error import ConfigurationError
from ptp.configuration.component_factory import ComponentFactory
from ptp.configuration.app_state import AppState

class PipelineManager(object):
    """
    Class responsible for instantiating the pipeline consisting of several components.
    """

    def __init__(self, name, params):
        """
        Initializes the pipeline object.

        :param params: Parameters used to instantiate all required components.
        :type params: ``utils.param_interface.ParamInterface``

        """
        # Initialize the logger.
        self.name = name
        self.params = params
        self.app_state = AppState()
        self.logger = logging.getLogger(name)

        # Set initial values of all pipeline elements.
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

        :param log_errors: Logs the detected errors (DEFAULT: True)

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
            try:
                # Skip "special" sections.
                if c_key in sections_to_skip:
                    self.logger.info("Skipping section '{}'".format(c_key))
                    continue
        
                # Create component.
                component, class_obj = ComponentFactory.build(c_key, c_params)

                # Check if class is derived (even indirectly) from Problem.
                if ComponentFactory.check_inheritance(class_obj, ptp.Problem.__name__):
                    raise ConfigurationError("Object '{}' cannot be instantiated as part of pipeline, \
                        as its class type '{}' is derived from Problem class!".format(c_key, class_obj.__name__))

                # Check presence of priority.
                if 'priority' not in c_params:
                    raise ConfigurationError("Section '{}' does not contain the key 'priority' defining the pipeline order".format(c_key))

                # Get the priority.
                try:
                    c_priority = float(c_params["priority"])
                except ValueError:
                    raise ConfigurationError("Priority '{}' in section '{}' is not a floating point number".format(c_params["priority"], c_key))

                # Check uniqueness of the priority.
                if c_priority in self.__components.keys():
                    raise ConfigurationError("Found more than one component with the same priority ('{}')".format(c_priority))

                # Add it to dict.
                self.__components[c_priority] = component

                # Check if class is derived (even indirectly) from Model.
                if ComponentFactory.check_inheritance(class_obj, ptp.Model.__name__):
                    # Add to list.
                    self.models.append(component)

                # Check if class is derived (even indirectly) from Loss.
                if ComponentFactory.check_inheritance(class_obj, ptp.Loss.__name__):
                    # Add to list.
                    self.losses.append(component)

            except ConfigurationError as e:
                if log_errors:
                    self.logger.error(e)
                errors += 1
                continue
                # end try/else
            # end for
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
        Returns the number of objects in the pipeline (excluding problems)
        :return: Length of the :py:class:`Pipeline`.

        """
        length = len(self.__priorities) 
        return length


    def summarize_io_header(self):
        """
        Creates the summary header.

        :return: Summary header as a str.
        """
        summary_str = '\n' + '='*80 + '\n'
        summary_str += 'Pipeline\n'
        summary_str += '  + Component name (type) [priority]\n'
        summary_str += '      Inputs:\n' 
        summary_str += '        key: dims, types, description\n'
        summary_str += '      Outputs:\n' 
        summary_str += '        key: dims, types, description\n'
        summary_str += '=' * 80 + '\n'
        return summary_str


    def summarize_io(self):
        """
        Summarizes the pipeline by showing all its components (excluding problem).

        :return: Summary as a str.
        """
        summary_str = '' 
        for prio in self.__priorities:
            # Get component
            comp = self.__components[prio]
            print(comp)
            summary_str += comp.summarize_io(prio)
        summary_str += '=' * 80 + '\n'
        return summary_str


    def handshake(self, data_dict, log=True):
        """
        Performs handshaking of inputs and outputs definitions of all components in the pipeline.

        :param data_dict: Initial datadict returned by the problem.

        :param log: Logs the detected errors and info (DEFAULT: True)

        :return: Number of detected errors.
        """
        errors = 0

        for prio in self.__priorities:
            # Get component
            comp = self.__components[prio]
            # Handshake inputs and outputs.
            errors += comp.handshake_input_definitions(data_dict, log)
            errors += comp.export_output_definitions(data_dict, log)

        # Log final definition.
        if errors == 0 and log:
            self.logger.info("Handshake successfull")
            def_str = "Final definition of DataDict used in pipeline:"
            def_str += '\n' + '='*80 + '\n'
            def_str += '{}'.format(data_dict)
            def_str += '\n' + '='*80 + '\n'
            self.logger.info(def_str)

        return errors


    def forward(self, data_dict):
        """
        Method responsible for processing the data dict, using all components in the components queue.

        :param data_dict: :py:class:`ptp.utils.DataDict` object containing both input data to be processed and that will be extended by the results.

        """
        # TODO: Convert to gpu/CUDA.
        #if self.app_state.use_gpu:
        #    data_dict = data_dict.cuda()

        for prio in self.__priorities:
            # Get component
            comp = self.__components[prio]
            # Forward step.
            comp(data_dict)
            # TODO: Move to gpu!

    def eval(self):
        """ 
        Sets evaluation mode for all models in the pipeline.
        """
        for model in self.models:
            model.eval()


    def train(self):
        """ 
        Sets evaluation mode for all models in the pipeline.
        """
        for model in self.models:
            model.train()


    def zero_grad(self):
        """ 
        Resets gradients in all trainable components of the pipeline.
        """
        for model in self.models:
            model.zero_grad()


    def backward(self, data_dict):
        """
        Propagates gradients backwards, starting from losses returned by every loss component in the pipeline.
        If using many losses the components derived from loss must overwrite the ''loss_keys()'' method.

        :param data_dict: :py:class:`ptp.utils.DataDict` object containing both input data to be processed and that will be extended by the results.

        """
        for loss in self.losses:
            for key in loss.loss_keys():
                data_dict[key].backward()


    def parameters(self, recurse=True):
        """
        Returns an iterator over parameters of all trainable components.

        This is typically passed to an optimizer.

        Args:
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.

        Yields:
            Parameter: module parameter

        Example::

        """
        for model in self.models:
            for _, param in model.named_parameters(recurse=recurse):
                yield param


    def named_parameters(self, recurse=True):
        """
        Returns an iterator over all named parameters of all trainable components.
        """
        for model in self.models:
            for name, param in model.named_parameters(recurse=recurse):
                yield name, param


    def add_statistics(self, stat_col):
        """
        Adds statistics for every component in the pipeline.

        :param stat_col: ``StatisticsCollector``.

        """
        for prio in self.__priorities:
            comp = self.__components[prio]
            comp.add_statistics(stat_col)


    def collect_statistics(self, stat_col, data_dict):
        """
        Collects statistics for every component in the pipeline.

        :param stat_col: :py:class:`ptp.utils.StatisticsCollector`.

        :param data_dict: ``DataDict`` containing inputs, targets etc.
        :type data_dict: :py:class:`ptp.core_types.DataDict`

        """
        for prio in self.__priorities:
            comp = self.__components[prio]
            comp.collect_statistics(stat_col, data_dict)


    def add_aggregators(self, stat_agg):
        """
        Aggregates statistics by calling adequate aggregation method of every component in the pipeline.

        :param stat_agg: ``StatisticsAggregator``.

        """
        for prio in self.__priorities:
            comp = self.__components[prio]
            comp.add_aggregators(stat_agg)


    def aggregate_statistics(self, stat_col, stat_agg):
        """
        Aggregates statistics by calling adequate aggregation method of every component in the pipeline.

        :param stat_col: ``StatisticsCollector``

        :param stat_agg: ``StatisticsAggregator``

        """
        for prio in self.__priorities:
            comp = self.__components[prio]
            comp.aggregate_statistics(stat_col, stat_agg)
