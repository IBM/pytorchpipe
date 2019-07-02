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

import signal
import logging
import numpy as np

from torch.utils.data import DataLoader

import ptp

import ptp.utils.logger as logging
from ptp.utils.app_state import AppState
from ptp.configuration.configuration_error import ConfigurationError
from ptp.application.component_factory import ComponentFactory
from ptp.application.sampler_factory import SamplerFactory


class TaskManager(object):
    """
    Class that instantiates and manages task and associated entities (dataloader, sampler etc.).
    """

    def __init__(self, name, config):
        """
        Initializes the manager.

        :param name: Name of the manager (associated with a given config section e.g. 'training', 'validation').

        :param config: 'ConfigInterface' object, referring to one of main sections (training/validation/test/...).
        :type config: :py:class:`ptp.configuration.ConfigInterface`

        """
        self.name = name
        self.config = config

        # Get access to AppState: for command line args, globals etc.
        self.app_state = AppState()

        # Initialize logger.
        self.logger = logging.initialize_logger(self.name)        

        # Single batch that will be used for validation (for validation task manager).
        self.batch = None


    def worker_init_fn(self, worker_id):
        """
        Function to be called by :py:class:`torch.utils.data.DataLoader` on each worker subprocess, \
        after seeding and before data loading.

        .. note::

            Set the ``NumPy`` random seed of the worker equal to the previous NumPy seed + its ``worker_id``\
             to avoid having all workers returning the same random numbers.


        :param worker_id: the worker id (in [0, :py:class:`torch.utils.data.DataLoader`.num_workers - 1])
        :type worker_id: int

        """
        # Set random seed of a worker.
        np.random.seed(seed=np.random.get_state()[1][0] + worker_id)

        # Ignores SIGINT signal - what enables "nice" termination of dataloader worker threads.
        # https://discuss.pytorch.org/t/dataloader-multiple-workers-and-keyboardinterrupt/9740/2
        signal.signal(signal.SIGINT, signal.SIG_IGN)


    def build(self, log=True):
        """
        Method creates a task on the basis of configuration section.

        :param log: Logs information and the detected errors (DEFAULT: TRUE)

        :return: number of detected errors
        """
        try: 
            # Create component.
            component, class_obj = ComponentFactory.build("task", self.config["task"])

            # Check if class is derived (even indirectly) from Task.
            if not ComponentFactory.check_inheritance(class_obj, ptp.Task.__name__):
                raise ConfigurationError("Class '{}' is not derived from the Task class!".format(class_obj.__name__))            

            # Set task.
            self.task = component

            # Try to build the sampler.
            # Check if sampler is required, i.e. 'sampler' section is empty.
            if "sampler" not in self.config:
                self.logger.info("The sampler configuration section is not present, using default 'random' sampling")
                # Set sampler to none.
                self.sampler = None
            else:
                self.sampler = SamplerFactory.build(self.task, self.config["sampler"], self.name)
                # Set shuffle to False - REQUIRED as those two are exclusive.
                self.config['dataloader'].add_config_params({'shuffle': False})

            # build the DataLoader on top of the validation task
            self.dataloader = DataLoader(dataset=self.task,
                    batch_size=self.config['task']['batch_size'],
                    shuffle=self.config['dataloader']['shuffle'],
                    sampler=self.sampler,
                    batch_sampler= None,
                    num_workers=self.config['dataloader']['num_workers'],
                    collate_fn=self.task.collate_fn,
                    pin_memory=self.config['dataloader']['pin_memory'],
                    drop_last=self.config['dataloader']['drop_last'],
                    timeout=self.config['dataloader']['timeout'],
                    worker_init_fn=self.worker_init_fn)

            # Display sizes.
            if log:
                self.logger.info("Task for '{}' loaded (size: {})".format(self.name, len(self.task)))
                if (self.sampler is not None):
                    self.logger.info("Sampler for '{}' created (size: {})".format(self.name, len(self.sampler)))

            # Ok, success.
            return 0

        except ConfigurationError as e:
            if log:
                self.logger.error("Detected configuration error while creating the task instance:\n  {}".format(e))
            # Return error.
            return 1
        except KeyError as e:
            if log:
                self.logger.error("Detected key error while creating the task instance: required key {} is missing".format(e))
            # Return error.
            return 1


    def __len__(self):
        """
        Returns total number of samples, calculated depending on the settings (batch size, dataloader, drop last etc.).
        """
        if self.dataloader.drop_last:
            # if we are supposed to drop the last (incomplete) batch.
            total_num_samples = len(self.dataloader) * self.dataloader.batch_size
        elif self.sampler is not None:
            total_num_samples = len(self.sampler)
        else:
            total_num_samples = len(self.task)

        return total_num_samples


    def get_epoch_size(self):
        """
        Compute the number of iterations ('episodes') to run given the size of the dataset and the batch size to cover
        the entire dataset once.

        Takes into account whether one used sampler or not.

        .. note::

            If the last batch is incomplete we are counting it in when ``drop_last`` in ``DataLoader()`` is set to Ttrue.

        .. warning::

            Leaving this method 'just in case', in most cases one might simply use ''len(dataloader)''.

        :return: Number of iterations to perform to go though the entire dataset once.

        """
        # "Estimate" dataset size.
        if (self.sampler is not None):
            task_size = len(self.sampler)
        else:
            task_size = len(self.task)

        # If task_size is a multiciplity of batch_size OR drop last is set.
        if (task_size % self.dataloader.batch_size) == 0 or self.dataloader.drop_last:
            return task_size // self.dataloader.batch_size
        else:
            return (task_size // self.dataloader.batch_size) + 1

    def initialize_epoch(self):
        """
        Function called to initialize a new epoch.
        """
        epoch = self.app_state.epoch
        # Update task settings depending on the epoch.
        self.task.initialize_epoch(epoch)

        # Generate a single batch used for partial validation.
        if self.name == 'validation':
            if self.batch is None or (self.sampler is not None and "kFold" in type(self.sampler).__name__):
                self.batch = next(iter(self.dataloader))
        # TODO refine partial validation section.
        # partial_validation:
        #   interval: 100 # How often to test.
        #   resample_at_epoch: True # at the beginning of new epoch.

    def finalize_epoch(self):
        """
        Function called at the end of an epoch to finalize it.
        """
        epoch = self.app_state.epoch
        self.task.initialize_epoch(epoch)
