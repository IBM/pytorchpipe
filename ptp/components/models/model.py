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

import numpy as np

from torch.nn import Module

from ptp.components.component import Component


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

        # Flag indicating whether the model is frozen or not.
        self.frozen = False


    def save_to_checkpoint(self, chkpt):
        """
        Adds model's state dictionary to checkpoint, which will be next stored to file.

        :param: Checkpoint (dictionary) that will be saved to file.
        """
        chkpt[self.name] = self.state_dict()


    def load_from_checkpoint(self, chkpt):
        """
        Loads state dictionary from checkpoint.

        :param: Checkpoint (dictionary) loaded from file.
        """
        self.load_state_dict(chkpt[self.name])

    def freeze(self):
        """
        Freezes the trainable weigths of the model.
        """
        # Freeze.
        print("FREEZEING ",self.name)
        self.frozen = True
        for param in self.parameters():
            param.requires_grad = False


    def summarize(self):
        """
        Summarizes the model by showing the trainable/non-trainable parameters and weights\
         per layer ( ``nn.Module`` ).

        Uses ``recursive_summarize`` to iterate through the nested structure of the model (e.g. for RNNs).

        :return: Summary as a str.

        """
        # go recursively in the model architecture
        summary_str = self.recursive_summarize(self, 0, self.name)

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

        if indent_ == 0:
            if self.frozen:
                mod_str += "\t\t[FROZEN]"
            else:
                mod_str += "\t\t[TRAINABLE]"


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
