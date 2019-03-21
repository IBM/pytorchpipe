# -*- coding: utf-8 -*-
#
# Copyright (C) IBM Corporation 2018-2019
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
from torch.utils.data import Dataset

from ptp.components.component import Component
from ptp.data_types.data_dict import DataDict


class Problem(Component, Dataset):
    """
    Class representing base class for all Problems.

    Inherits from :py:class:`torch.utils.data.Dataset` as all subclasses will represent a problem with an associated dataset,\
    and the `worker` will use :py:class:`torch.utils.data.DataLoader` to generate batches.

    Implements features & attributes used by all subclasses.

    """

    def __init__(self, name, class_type, params):
        """
        Initializes problem object:
            - calls base class constructors.
            - sets key_indices variable (used for storing indices of samples)

                >>> self.key_indices = self.mapkey("indices")

            - sets empry curriculim learning params

                >>> self.curriculum_params = {}
        
        :param name: Problem name.
        :type name: str

        :param class_type: Class type of the component.

        :param params: Dictionary of parameters (read from the configuration ``.yaml`` file).
        :type params: :py:class:`ptp.utils.ParamInterface`

        .. note::

            It is likely to encounter a case where the model needs a parameter value only known when the problem has been
            instantiated, like the size of a vocabulary set or the number of marker bits.

            The user can pass those values in this app_state. All objects will be able to access it later:

                >>> self.app_state["new_global_value"] = 1 # Sets global value.
                >>> val = self.app_state["new_global_value" # Gets global value.
        """
        # Call constructors of parent classes.
        Component.__init__(self, name, class_type, params)
        Dataset.__init__(self)

        # Set default key mappings.
        self.key_indices = self.mapkey("indices")

        # Empty curriculum learning params - for now.
        self.curriculum_params = {}


    def summarize_io(self, priority = -1):
        """
        Summarizes the problem by showing its name, type and output definitions.

        :param priority: Problem priority (DEFAULT: -1)

        :return: Summary as a str.

        """
        summary_str = "  + {} ({}) [{}]\n".format(self.name, type(self).__name__, priority)
        # Get outputs.
        summary_str += '      Outputs:\n' 
        for key,value in self.output_data_definitions().items():
            summary_str += '        {}: {}, {}, {}\n'.format(key, value.dimensions, value.types, value. description)
        return summary_str

    def __call__(self, data_dict):
        """
        Method responsible for processing the data dict. Empty for all problem-derived classes.

        :param data_dict: :py:class:`ptp.utils.DataDict` object containing both input data to be proces and that will be extended by the results.
        """
        pass

    def input_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of input data that are required by the component.
        As there is assumption made (problems do not accept inputs) it returns empty dictionary.

        :return: Empty dictionary.
        """
        return {}


    def create_data_dict(self, index, data_definitions = None):
        """
        Returns a :py:class:`ptp.utils.DataDict` object with keys created on the \
        problem data_definitions and empty values (None).

        :param data_definitions: Data definitions that will be used (DEFAULT: None, meaninng that self.output_data_definitions() will be used)

        :return: new :py:class:`ptp.utils.DataDict` object.
        """
        # Use self.output_data_definitions() if required
        data_definitions = data_definitions if data_definitions is not None else self.output_data_definitions()
        # Add index - just in case. This key is required!
        if self.key_indices not in data_definitions:
            data_definitions[self.key_indices] = None
        data_dict = DataDict({key: None for key in data_definitions.keys()})
        # Set index.
        data_dict[self.key_indices] = index
        return data_dict


    def collate_fn(self, batch):
        """
        Generates a batch of samples from a list of individuals samples retrieved by :py:func:`__getitem__`.

        The method calls :py:func:`torch.utils.data.dataloader.default_collate` for every item in data_dict !
        
        .. note::

            This base :py:func:`collate_fn` method only calls the default \
            :py:func:`torch.utils.data.dataloader.default_collate`, as it can handle several cases \
            (mainly tensors, numbers, dicts and lists).

            If your dataset can yield variable-length samples within a batch, or generate batches `on-the-fly`\
            , or possesses another `non regular` characteristic, it is most likely that you will need to \
            override this default :py:func:`collate_fn`.


        :param batch: List of :py:class:`ptp.utils.DataDict` retrieved by :py:func:`__getitem__`, each containing \
        tensors, numbers, dicts or lists.
        :type batch: list

        :return: DataDict containing the created batch.

        """
        return DataDict({key: torch.utils.data.dataloader.default_collate([sample[key] for sample in batch]) for key in batch[0]})


    def initialize_epoch(self, epoch):
        """
        Function called to initialize a new epoch.

        .. note::


            Empty - To be redefined in inheriting classes.

        :param epoch: current epoch index
        :type epoch: int


        """
        pass

    def finalize_epoch(self, epoch):
        """
        Function called at the end of an epoch to execute a few tasks.

        .. note::


            Empty - To be redefined in inheriting classes.


        :param epoch: current epoch index
        :type epoch: int

        """
        pass


    def curriculum_learning_initialize(self, curriculum_params):
        """
        Initializes curriculum learning - simply saves the curriculum params.

        .. note::

            This method can be overwritten in the derived classes.


        :param curriculum_params: Interface to parameters accessing curriculum learning view of the registry tree.
        :type param: :py:class:`ptp.utils.ParamInterface`


        """
        # Save params.
        self.curriculum_params = curriculum_params


    def curriculum_learning_update_params(self, episode):
        """
        Updates problem parameters according to curriculum learning.

        .. note::

            This method can be overwritten in the derived classes.

        :param episode: Number of the current episode.
        :type episode: int

        :return: True informing that Curriculum Learning wasn't active at all (i.e. is finished).

        """

        return True
