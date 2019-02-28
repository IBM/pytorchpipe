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


import torch
import logging
from datetime import datetime
from numpy import inf

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
        Initializes the pipeline manager.

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

        # Initialization of best loss - as INF.
        self.best_loss = inf
        self.best_status = "Unknown"


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
        sections_to_skip = "name disable".split()
        disabled_components = ''
        # Add components to disable by the ones from configuration file.
        if "disable" in self.params:
            disabled_components = [*disabled_components, *self.params["disable"].split(",")]
        # Add components to disable by the ones from command line arguments.
        if (self.app_state.args is not None) and (self.app_state.args.disable != ''):
            disabled_components = [*disabled_components, *self.app_state.args.disable.split(",")]

        for c_key, c_params in self.params.items():
            # The section "key" will be used as "component" name.
            try:
                # Skip "special" sections.
                if c_key in sections_to_skip:
                    #self.logger.info("Skipping section '{}'".format(c_key))
                    continue
                # Skip "disabled" components.
                if c_key in disabled_components:
                    self.logger.info("Disabling component '{}'".format(c_key))
                    continue
        
                # Create component.
                component, class_obj = ComponentFactory.build(c_key, c_params)

                # Check if class is derived (even indirectly) from Problem.
                if ComponentFactory.check_inheritance(class_obj, ptp.Problem.__name__):
                    raise ConfigurationError("Object '{}' cannot be instantiated as part of pipeline, \
                        as its class type '{}' is derived from Problem class!".format(c_key, class_obj.__name__))

                # Check presence of priority.
                if 'priority' not in c_params:
                    raise KeyError("Section '{}' does not contain the key 'priority' defining the pipeline order".format(c_key))

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
            except KeyError as e:
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


    def save(self, chkpt_dir, training_status, loss, episode, epoch):
        """
        Generic method saving the parameters of all models in the pipeline to a file.

        :param chkpt_dir: Directory where the model will be saved.
        :type chkpt_dir: str

        :param training_status: String representing the current status of training.
        :type training_status: str


        :return: True if this is currently the best model (until the current episode, considering the loss).

        """
        # Checkpoint to be saved.
        chkpt = {'name': self.name,
                 'timestamp': datetime.now(),
                 'episode': episode,
                 'loss': loss,
                 'status': training_status,
                 'status_timestamp': datetime.now(),
                }
        
        model_str = ''
        # Save state dicts of all models.
        for model in self.models:
            chkpt[model.name] = model.state_dict()
            model_str += "  + Model '{}' [{}] params saved \n".format(model.name, type(model).__name__)

        # Save the intermediate checkpoint.
        if self.app_state.args.save_intermediate:
            filename = chkpt_dir + self.name + '_episode_{:05d}.pt'.format(episode)
            torch.save(chkpt, filename)
            log_str = "Exporting pipeline '{}' parameters to checkpoint {}:\n".format(self.name, filename)
            log_str += model_str
            self.logger.info(log_str)

        # Save the best "model".
        # loss = loss.cpu()  # moving loss value to cpu type to allow (initial) comparison with numpy type
        if loss < self.best_loss:
            # Save best loss and status.
            self.best_loss = loss
            self.best_status = training_status
            # Save checkpoint.
            filename = chkpt_dir + self.name + '_best.pt'
            torch.save(chkpt, filename)
            log_str = "Exporting pipeline '{}' parameters to checkpoint {}:\n".format(self.name, filename)
            log_str += model_str
            self.logger.info(log_str)
            return True
        elif self.best_status != training_status:
            filename = chkpt_dir + self.name + '_best.pt'
            # Load checkpoint.
            chkpt_loaded = torch.load(filename, map_location=lambda storage, loc: storage)
            # Update status and status time.
            chkpt_loaded['status'] = training_status
            chkpt_loaded['status_timestamp'] = datetime.now()
            # Save updated checkpoint.
            torch.save(chkpt_loaded, filename)
            self.logger.info("Updated training status in checkpoint {}".format(filename))
        # Else: that was not the best "model".
        return False

    def load(self, checkpoint_file):
        """
        Loads parameters of models in the pipeline from the specified checkpoint file.

        :param checkpoint_file: File containing dictionary with states of all models in the pipeline with some additional checkpoint statistics.

        """
        # Load checkpoint
        # This is to be able to load a CUDA-trained model on CPU
        chkpt = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)

        log_str = "Importing pipeline '{}' parameters from checkpoint from {} (episode: {}, loss: {}, status: {}):\n".format(
                chkpt['name'],
                chkpt['timestamp'],
                chkpt['episode'],
                chkpt['loss'],
                chkpt['status']
                )
        model_str = ''
        warning = False
        # Save state dicts of all models.
        for model in self.models:
            if model.name in chkpt.keys():
                # Load model.
                model.load_state_dict(chkpt[model.name])
                model_str += "  + Model '{}' [{}] params loaded\n".format(model.name, type(model).__name__)
            else:
                model_str += "  + Model '{}' [{}] params not found in checkpoint!\n".format(model.name, type(model).__name__)
                warning = True

        # Log results.
        log_str += model_str
        if warning:
            self.logger.warning(log_str)
        else:
            self.logger.info(log_str)



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


    def summarize_all_components_header(self):
        """
        Creates the summary header containing components with inputs-outputs definitions.

        :return: Summary header as a str.
        """
        summary_str  = 'Summary of the created pipeline:\n'
        summary_str += '='*80 + '\n'
        summary_str += 'Pipeline\n'
        summary_str += '  + Component name (type) [priority]\n'
        summary_str += '      Inputs:\n' 
        summary_str += '        key: dims, types, description\n'
        summary_str += '      Outputs:\n' 
        summary_str += '        key: dims, types, description\n'
        summary_str += '=' * 80 + '\n'
        return summary_str


    def summarize_all_components(self):
        """
        Summarizes the pipeline by showing all its components (excluding problem).

        :return: Summary as a str.
        """
        summary_str = '' 
        for prio in self.__priorities:
            # Get component
            comp = self.__components[prio]
            summary_str += comp.summarize_io(prio)
        summary_str += '=' * 80 + '\n'
        return summary_str

    def summarize_models_header(self):
        """
        Creates the summary header containing details of models.

        :return: Summary header as a str.
        """
        summary_str  = 'Summary of the models in the pipeline:\n'
        summary_str += '='*80 + '\n'
        summary_str += 'Model name (Type) \n'
        summary_str += '  + Submodule name (Type) \n'
        summary_str += '      Matrices: [(name, dims), ...]\n'
        summary_str += '      Trainable Params: #\n'
        summary_str += '      Non-trainable Params: #\n'
        summary_str += '=' * 80 + '\n'
        return summary_str

    def summarize_models(self):
        """
        Summarizes the pipeline by showing all its components (excluding problem).

        :return: Summary as a str.
        """
        summary_str = '' 
        for model in self.models:
            summary_str += model.summarize()
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

    def cuda(self):
        """ 
        Moves all models to GPU.
        """
        self.logger.info("Moving model(s) to GPU")
        for model in self.models:
            model.cuda()


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
        if (len(self.losses) == 0):
            raise ConfigurationError("Cannot train using backpropagation as there are no 'Loss' components")
        for loss in self.losses:
            for key in loss.loss_keys():
                data_dict[key].backward()


    def get_loss(self, data_dict):
        """
        Sums all losses and returns a single value that can be used e.g. in terminal condition or model(s) saving.

        :param data_dict: :py:class:`ptp.utils.DataDict` object containing both input data to be processed and that will be extended by the results.

        :return: Loss (scalar value).
        """
        if (len(self.losses) == 0):
            raise ConfigurationError("Cannot train using backpropagation as there are no 'Loss' components")
        loss_sum = 0
        for loss in self.losses:
            for key in loss.loss_keys():
                loss_sum += data_dict[key].cpu().item()
        return loss_sum


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