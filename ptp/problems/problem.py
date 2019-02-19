#!/usr/bin/env python3
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

__author__ = "Tomasz Kornuta & Vincent Marois"

import signal
import torch
import os
import logging
import urllib
import time
import sys
import numpy as np
from torch.utils.data import Dataset

from ptp.utils.app_state import AppState
from ptp.utils.data_dict import DataDict


class Problem(Dataset):
    """
    Class representing base class for all Problems.

    Inherits from :py:class:`torch.utils.data.Dataset` as all subclasses will represent a problem with an associated dataset,\
    and the `worker` will use :py:class:`torch.utils.data.DataLoader` to generate batches.

    Implements features & attributes used by all subclasses.

    """

    def __init__(self, params_, name_='Problem'):
        """
        Initializes problem object.

        :param params_: Dictionary of parameters (read from the configuration ``.yaml`` file).
        :type params_: :py:class:`miprometheus.utils.ParamInterface`

        :param name_: Problem name (DEFAULT: 'Problem').
        :type name_: str

        This constructor:

        - stores a pointer to ``params``:

            >>> self.params = params_

        - sets a problem name:

            >>> self.name = name_

        - sets a default loss function:

            >>> self.loss_function = None

        - initializes the size of the dataset:

            >>> self.length = None

        - initializes the logger.

            >>> self.logger = logging.Logger(self.name)

        - initializes the data definitions: this is used for defining the ``DataDict`` keys.

        .. note::

            This dict contains information about the DataDict produced by the current problem class.

            This object will be used during handshaking between the model and the problem class to ensure that the model
            can accept the batches produced by the problem.

            This dict should at least contains the `targets` field:

                >>> self.data_definitions = {'targets': {'size': [-1, 1], 'type': [torch.Tensor]}}

        - initializes the default values: this is used to pass missing parameters values to the model.

        .. note::

            It is likely to encounter a case where the model needs a parameter value only known when the problem has been
            instantiated, like the size of a vocabulary set or the number of marker bits.

            The user can fill in those values in this dict, which will be passed to the model in its  `__init__`  . The
            model will then be able to fill it its missing parameters values, either from params or this dict.

                >>> self.default_values = {}

        - sets the access to ``AppState``: for dtype, visualization flag etc.

            >>> self.app_state = AppState()

        """
        # Store pointer to params.
        self.params = params_

        # Problem name.
        self.name = name_

        # Empty curriculum learning params - for now.
        self.curriculum_params = {}

        # Set default loss function.
        self.loss_function = None

        # Size of the dataset
        self.length = None

        # Initialize the logger.
        self.logger = logging.getLogger(self.name)

        # data_definitions: this is used for defining the DataDict keys.

        # This dict contains information about the DataDict produced by the current problem class.
        # This object will be used during handshaking between the model and the problem class to ensure that the model
        # can accept the batches produced by the problem.
        self.data_definitions = {}

        # default_values: this is used to pass missing parameters values to the model.

        # It is likely to encounter a case where the model needs a parameter value only known when the problem has been
        # instantiated, like the size of a vocabulary set or the number of marker bits.
        # The user can fill in those values in this dict, which will be passed to the model in its  `__init__`  . The
        # model will then be able to fill it its missing parameters values, either from params or this dict.
        self.default_values = {}

        # Get access to AppState: for dtype, visualization flag etc.
        self.app_state = AppState()

    def create_data_dict(self):
        """
        Returns a :py:class:`miprometheus.utils.DataDict` object with keys created on the \
        problem data_definitions and empty values (None).

        :return: new :py:class:`miprometheus.utils.DataDict` object.

        """
        return DataDict({key: None for key in self.data_definitions.keys()})

    def __len__(self):
        """
        :return: The size of the dataset.

        """
        return self.length

    def set_loss_function(self, loss_function):
        """
        Sets loss function.

        :param loss_function: Loss function (e.g. :py:class:`torch.nn.CrossEntropyLoss`) that will be set as \
        the optimization criterion.

        """
        self.loss_function = loss_function

    def collate_fn(self, batch):
        """
        Generates a batch of samples from a list of individuals samples retrieved by :py:func:`__getitem__`.

        The default collate_fn is :py:func:`torch.utils.data.dataloader.default_collate`.

        .. note::

            This base :py:func:`collate_fn` method only calls the default \
            :py:func:`torch.utils.data.dataloader.default_collate`, as it can handle several cases \
            (mainly tensors, numbers, dicts and lists).

            If your dataset can yield variable-length samples within a batch, or generate batches `on-the-fly`\
            , or possesses another `non regular` characteristic, it is most likely that you will need to \
            override this default :py:func:`collate_fn`.


        :param batch: :py:class:`miprometheus.utils.DataDict` retrieved by :py:func:`__getitem__`, each containing \
        tensors, numbers, dicts or lists.
        :type batch: list

        :return: DataDict containing the created batch.

        """
        return torch.utils.data.dataloader.default_collate(batch)

    def __getitem__(self, index):
        """
        Getter that returns an individual sample from the problem's associated dataset (that can be generated \
        `on-the-fly`, or retrieved from disk. It can also possibly be composed of several files.).

        .. note::

            **To be redefined in subclasses.**


        .. note::

            **The getter should return a DataDict: its keys should be defined by** ``self.data_definitions`` **keys.**

            This ensures consistency of the content of the :py:class:`miprometheus.utils.DataDict` when processing \
            to the `handshake` between the :py:class:`miprometheus.problems.Problem` class and the \
            :py:class:`miprometheus.models.Model` class. For more information, please see\
             :py:func:`miprometheus.models.Model.handshake_definitions`.

            e.g.:

                >>> data_dict = DataDict({key: None for key in self.data_definitions.keys()})
                >>> # you can now access each value by its key and assign the corresponding object (e.g. `torch.tensor` etc)
                >>> ...
                >>> return data_dict



        .. warning::

            `Mi-Prometheus` supports multiprocessing for data loading (through the use of\
             :py:class:`torch.utils.data.DataLoader`).

            To construct a batch (say 64 samples), the indexes are distributed among several workers (say 4, so that
            each worker has 16 samples to retrieve). It is best that samples can be accessed individually in the dataset
            folder so that there is no mutual exclusion between the workers and the performance is not degraded.

            If each sample is generated `on-the-fly`, this shouldn't cause a problem. There may be an issue with \
            randomness. Please refer to the official PyTorch documentation for this.


        :param index: index of the sample to return.
        :type index: int

        :return: Empty ``DataDict``, having the same key as ``self.data_definitions``.

        """
        return DataDict({key: None for key in self.data_definitions.keys()})

    def worker_init_fn(self, worker_id):
        """
        Function to be called by :py:class:`torch.utils.data.DataLoader` on each worker subprocess, \
        after seeding and before data loading. (default: ``None``).

        .. note::

            Set the ``NumPy`` random seed of the worker equal to the previous NumPy seed + its ``worker_id``\
             to avoid having all workers returning the same random numbers.


        :param worker_id: the worker id (in [0, :py:class:`torch.utils.data.DataLoader`.num_workers - 1])
        :type worker_id: int

        :return: ``None`` by default
        """
        # Set random seed of a worker.
        np.random.seed(seed=np.random.get_state()[1][0] + worker_id)

        # Ignores SIGINT signal - what enables "nice" termination of dataloader worker threads.
        # https://discuss.pytorch.org/t/dataloader-multiple-workers-and-keyboardinterrupt/9740/2
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    def get_data_definitions(self):
        """
        Getter for the data_definitions dict so that it can be accessed by a ``worker`` to establish handshaking with
        the :py:class:`miprometheus.models.Model` class.

        :return: self.data_definitions()

        """
        return self.data_definitions

    def evaluate_loss(self, data_dict, logits):
        """
        Calculates loss between the predictions / logits and targets (from ``data_dict``) using the selected \
        loss function.

        :param data_dict: DataDict containing (among others) inputs and targets.
        :type data_dict: :py:class:`miprometheus.utils.DataDict`

        :param logits: Predictions of the model.

        :return: Loss.
        """

        # Compute loss using the provided loss function. 
        loss = self.loss_function(logits, data_dict['targets'])

        return loss

    def add_statistics(self, stat_col):
        """
        Adds statistics to :py:class:`miprometheus.utils.StatisticsCollector`.

        .. note::


            Empty - To be redefined in inheriting classes.


        :param stat_col: :py:class:`miprometheus.utils.StatisticsCollector`.

        """
        pass
        
    def collect_statistics(self, stat_col, data_dict, logits):
        """
        Base statistics collection.

         .. note::


            Empty - To be redefined in inheriting classes. The user has to ensure that the corresponding entry \
            in the :py:class:`miprometheus.utils.StatisticsCollector` has been created with \
            :py:func:`add_statistics` beforehand.

        :param stat_col: :py:class:`miprometheus.utils.StatisticsCollector`.

        :param data_dict: ``DataDict`` containing inputs and targets.
        :type data_dict: :py:class:`miprometheus.utils.DataDict`

        :param logits: Predictions being output of the model (:py:class:`torch.Tensor`).

        """
        pass

    def add_aggregators(self, stat_agg):
        """
        Adds statistical aggregators to :py:class:`miprometheus.utils.StatisticsAggregator`.

        .. note::

            Empty - To be redefined in inheriting classes.


        :param stat_agg: :py:class:`miprometheus.utils.StatisticsAggregator`.

        """
        pass

    def aggregate_statistics(self, stat_col, stat_agg):
        """
        Aggregates the statistics collected by :py:class:`miprometheus.utils.StatisticsCollector` and adds the \
        results to :py:class:`miprometheus.utils.StatisticsAggregator`.

         .. note::

            Empty - To be redefined in inheriting classes.
            The user can override this function in subclasses but should call \
            :py:func:`aggregate_statistics` to collect basic statistical aggregators (if set).


        :param stat_col: :py:class:`miprometheus.utils.StatisticsCollector`.

        :param stat_agg: :py:class:`miprometheus.utils.StatisticsAggregator`.

        """
        pass

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

    def plot_preprocessing(self, data_dict, logits):
        """
        Allows for some data preprocessing before the model creates a plot for visualization during training or
        inference.

        .. note::


            Empty - To be redefined in inheriting classes.


        :param data_dict: ``DataDict``.
        :type data_dict: :py:class:`miprometheus.utils.DataDict`

        :param logits: Predictions of the model (:py:class:`torch.Tensor`).

        :return: data_dict, logits after preprocessing.

        """
        return data_dict, logits

    def curriculum_learning_initialize(self, curriculum_params):
        """
        Initializes curriculum learning - simply saves the curriculum params.

        .. note::

            This method can be overwritten in the derived classes.


        :param curriculum_params: Interface to parameters accessing curriculum learning view of the registry tree.
        :type param: :py:class:`miprometheus.utils.ParamInterface`


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

    # Function to make check and download easier
    def check_and_download(self, file_folder_to_check, url=None, download_name='~/data/downloaded'):
        """
        Checks whether a file or folder exists at given path (relative to storage folder), \
        otherwise downloads files from the given URL.

        :param file_folder_to_check: Relative path to a file or folder to check to see if it exists.
        :type file_folder_to_check: str

        :param url: URL to download files from.
        :type url: str

        :param download_name: What to name the downloaded file. (DEFAULT: "downloaded").
        :type download_name: str

        :return: False if file was found, True if a download was necessary.

        """

        file_folder_to_check = os.path.expanduser(file_folder_to_check)
        if not (os.path.isfile(file_folder_to_check) or os.path.isdir(file_folder_to_check)):
            if url is not None:
                self.logger.info('Downloading {}'.format(url))
                urllib.request.urlretrieve(url, os.path.expanduser(download_name), reporthook)
                return True
            else:
                return True
        else:
            self.logger.info('Dataset found at {}'.format(file_folder_to_check))
            return False

# Progress bar function
def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
            start_time = time.time()
            return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                     (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()


if __name__ == '__main__':
    """Unit test for Problem and DataDict"""
    from miprometheus.utils.param_interface import ParamInterface

    params = ParamInterface()

    problem = Problem(params)
    problem.data_definitions = {'inputs': {'size': [-1, -1], 'type': [torch.Tensor]},
                                'targets': {'size': [-1], 'type': [torch.Tensor]}
                                }
    problem.loss_function = torch.nn.CrossEntropyLoss()  # torch.nn.L1Loss, torch.nn.TripletMarginLoss

    datadict = DataDict({key: None for key in problem.data_definitions.keys()})

    # datadict['inputs'] = torch.ones([64, 20, 512]).type(torch.FloatTensor)
    # datadict['targets'] = torch.ones([64, 20]).type(torch.FloatTensor)

    # print(repr(datadict))


