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

__author__ = "Tomasz Kornuta & Vincent Marois"

import logging
import numpy as np
from datetime import datetime
from abc import abstractmethod

import torch
from torch.nn import Module

from ptp.core_types.component import Component
from ptp.utils.app_state import AppState


class Model(Module, Component):
    """
    Class representing base class for all Models.

    Inherits from :py:class:`torch.nn.Module` as all subclasses will represent a trainable model.

    Hence, all subclasses should override the ``forward`` function.

    Implements features & attributes used by all subclasses.

    """

    def __init__(self, name, params):
        """
        Initializes a Model object.

        :param name: Model name.
        :type name: str

        :param params: Parameters read from configuration file.
        :type params: ``ptp.utils.ParamInterface``

        This constructor:

        - calls base class constructors (save params, name, logger, app_state etc.)

        - initializes the best model loss (used to select which model to save) to ``np.inf``:

            >>> self.best_loss = np.inf

        """
        # Call constructors of parent classes.
        Component.__init__(self, name, params)
        Module.__init__(self)

        # Flag indicating whether intermediate checkpoints should be saved or
        # not (DEFAULT: False).
        params.add_default_params({"save_intermediate": False})
        self.save_intermediate = params["save_intermediate"]

        # Initialization of best loss - as INF.
        self.best_loss = np.inf
        self.best_status = "Unknown"


    def save(self, model_dir, training_status, training_stats, validation_stats):
        """
        Generic method saving the model parameters to file. It can be \
        overloaded if one needs more control.

        :param model_dir: Directory where the model will be saved.
        :type model_dir: str

        :param training_status: String representing the current status of training.
        :type training_status: str

        :param training_stats: Training statistics that will be saved to checkpoint along with the model.
        :type training_stats: :py:class:`miprometheus.utils.StatisticsCollector` or \
        :py:class:`miprometheus.utils.StatisticsAggregator`

        :param validation_stats: Validation statistics that will be saved to checkpoint along with the model.
        :type validation_stats: :py:class:`miprometheus.utils.StatisticsCollector` or \
        :py:class:`miprometheus.utils.StatisticsAggregator`

        :return: True if this is currently the best model (until the current episode, considering the loss).

        """
        # Process validation statistics, get the episode and loss.
        if validation_stats.__class__.__name__ == 'StatisticsCollector':
            # Get data from collector.
            episode = validation_stats['episode'][-1]
            loss = validation_stats['loss'][-1]

        else:
            # Get data from StatisticsAggregator.
            episode = validation_stats['episode']
            loss = validation_stats['loss']

        # Checkpoint to be saved.
        chkpt = {'name': self.name,
                 'state_dict': self.state_dict(),
                 'model_timestamp': datetime.now(),
                 'episode': episode,
                 'loss': loss,
                 'status': training_status,
                 'status_timestamp': datetime.now(),
                 'training_stats': training_stats.export_to_checkpoint(),
                 'validation_stats': validation_stats.export_to_checkpoint()
                }

        # Save the intermediate checkpoint.
        if self.save_intermediate:
            filename = model_dir + 'model_episode_{:05d}.pt'.format(episode)
            torch.save(chkpt, filename)
            self.logger.info(
                "Model and statistics exported to checkpoint {}".format(filename))

        # Save the best model.
        # loss = loss.cpu()  # moving loss value to cpu type to allow (initial) comparison with numpy type
        if loss < self.best_loss:
            # Save best loss and status.
            self.best_loss = loss
            self.best_status = training_status
            # Save checkpoint.
            filename = model_dir + 'model_best.pt'
            torch.save(chkpt, filename)
            self.logger.info("Model and statistics exported to checkpoint {}".format(filename))
            return True
        elif self.best_status != training_status:
            filename = model_dir + 'model_best.pt'
            # Load checkpoint.
            chkpt_loaded = torch.load(filename, map_location=lambda storage, loc: storage)
            # Update status and status time.
            chkpt_loaded['status'] = training_status
            chkpt_loaded['status_timestamp'] = datetime.now()
            # Save updated checkpoint.
            torch.save(chkpt_loaded, filename)
            self.logger.info("Updated training status in checkpoint {}".format(filename))
        # Else: that was not the best model.
        return False

    def load(self, checkpoint_file):
        """
        Loads a model from the specified checkpoint file.

        :param checkpoint_file: File containing dictionary with model state and statistics.

        """
        # Load checkpoint
        # This is to be able to load a CUDA-trained model on CPU
        chkpt = torch.load(
            checkpoint_file, map_location=lambda storage, loc: storage)

        # Load model.
        self.load_state_dict(chkpt['state_dict'])

        # Print statistics.
        self.logger.info(
            "Imported {} parameters from checkpoint from {} (episode: {}, loss: {}, status: {})".format(
                chkpt['name'],
                chkpt['model_timestamp'],
                chkpt['episode'],
                chkpt['loss'],
                chkpt['status']
                ))

    def summarize(self):
        """
        Summarizes the model by showing the trainable/non-trainable parameters and weights\
         per layer ( ``nn.Module`` ).

        Uses ``recursive_summarize`` to iterate through the nested structure of the model (e.g. for RNNs).

        :return: Summary as a str.

        """
        # add name of the current module
        summary_str = '\n' + '='*80 + '\n'
        summary_str += 'Model name (Type) \n'
        summary_str += '  + Submodule name (Type) \n'
        summary_str += '      Matrices: [(name, dims), ...]\n'
        summary_str += '      Trainable Params: #\n'
        summary_str += '      Non-trainable Params: #\n'
        summary_str += '=' * 80 + '\n'

        # go recursively in the model architecture
        summary_str += self.recursive_summarize(self, 0, self.name)

        # Sum the model parameters.
        num_total_params = sum([np.prod(p.size()) for p in self.parameters()])
        mod_trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        num_trainable_params = sum([np.prod(p.size()) for p in mod_trainable_params])

        summary_str += '\nTotal Trainable Params: {}\n'.format(num_trainable_params)
        summary_str += 'Total Non-trainable Params: {}\n'.format(num_total_params-num_trainable_params) 
        summary_str += '='*80 + '\n'

        return summary_str

    def recursive_summarize(self, module_, indent_, module_name_):
        """
        Function that recursively inspects the (sub)modules and records their statistics\
          (like names, types, parameters, their numbers etc.)

        :param module_: Module to be inspected.
        :type module_: ``nn.Module`` or subclass

        :param indent_: Current indentation level.
        :type indent_: int

        :param module_name_: Name of the module that will be displayed before its type.
        :type module_name_: str

        :return: Str summarizing the module.
        """
        # Recursively inspect the children.
        child_lines = []
        for key, module in module_._modules.items():
            child_lines.append(self.recursive_summarize(module, indent_+1, key))

        # "Leaf information". 
        mod_str = ''

        if indent_ > 0:
            mod_str += '  ' + '| ' * (indent_-1) + '+ '

        mod_str += module_name_ + " (" + module_._get_name() + ')'

        mod_str += '\n'
        mod_str += ''.join(child_lines)

        # Get leaf weights and number of params - only for leafs!
        if not child_lines:
            # Collect names and dimensions of all (named) params. 
            mod_weights = [(n, tuple(p.size())) for n, p in module_.named_parameters()]
            mod_str += '  ' + '| ' * indent_ + '  Matrices: {}\n'.format(mod_weights)

            # Sum the parameters.
            num_total_params = sum([np.prod(p.size()) for p in module_.parameters()])
            mod_trainable_params = filter(lambda p: p.requires_grad, module_.parameters())
            num_trainable_params = sum([np.prod(p.size()) for p in mod_trainable_params])

            mod_str += '  ' + '| ' * indent_ + '  Trainable Params: {}\n'.format(num_trainable_params)
            mod_str += '  ' + '| ' * indent_ + '  Non-trainable Params: {}\n'.format(num_total_params -
                                                                                     num_trainable_params)
            mod_str += '  ' + '| ' * indent_ + '\n'
    
        return mod_str
