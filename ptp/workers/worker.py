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

__author__ = "Vincent Marois, Tomasz Kornuta, Ryan L. McAvoy"

import torch
import argparse
import numpy as np
from random import randrange
from abc import abstractmethod

import ptp.utils.logger as logging
from ptp.utils.app_state import AppState

from ptp.configuration.config_interface import ConfigInterface
from ptp.configuration.config_parsing import load_class_default_config_file


class Worker(object):
    """
    Base abstract class for the workers.
    All base workers should subclass it and override the relevant methods.
    """

    def __init__(self, name, class_type, add_default_parser_args = True):
        """
        Base constructor for all workers:

            - Initializes the AppState singleton
            - Initializes the Configuration Registry
            - Loads default parameters
            - Creates parser and adds default worker command line arguments

        :param name: Name of the worker.
        :type name: str

        :param class_type: Class type of the component.

        :param add_default_parser_args: If set, adds default parser arguments (DEFAULT: True).
        :type add_default_parser_args: bool

        """
        # Call base constructor.
        super(Worker, self).__init__()

        # Set worker name.
        self.name = name

        # Initialize the application state singleton.
        self.app_state = AppState()

        # Initialize parameter interface/registry.
        self.config = ConfigInterface()

        # Load default configuration.
        if class_type is not None:
            self.config.add_default_params(load_class_default_config_file(class_type))


        # Create parser with a list of runtime arguments.
        self.parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

        # Add arguments to the specific parser.
        if add_default_parser_args:
            # These arguments will be shared by all basic workers.
            self.parser.add_argument(
                '--config',
                dest='config',
                type=str,
                default='',
                help='Name of the configuration file(s) to be loaded. '
                    'If specifying more than one file, they must be separated with coma ",".')

            self.parser.add_argument(
                '--disable',
                type=str,
                default='',
                dest='disable',
                help='Comma-separated list of components to be disabled (DEFAULT: empty)')

            self.parser.add_argument(
                '--load',
                type=str,
                default='',
                dest='load_checkpoint',
                help='Path and name of the checkpoint file containing the saved parameters'
                    ' of the pipeline models to load (should end with a .pt extension)')

            self.parser.add_argument(
                '--gpu',
                dest='use_gpu',
                action='store_true',
                help='The current worker will move the computations on GPU devices, if available '
                    'in the system. (Default: False)')

            self.parser.add_argument(
                '--expdir',
                dest='expdir',
                type=str,
                default="~/experiments",
                help='Path to the directory where the experiment(s) folders are/will be stored.'
                    ' (DEFAULT: ~/experiments)')

            self.parser.add_argument(
                '--savetag',
                dest='savetag',
                type=str,
                default='',
                help='Tag for the save directory.')

            self.parser.add_argument(
                '--logger',
                action='store',
                dest='log_level',
                type=str,
                default='INFO',
                choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'],
                help="Log level. (DEFAULT: INFO)")

            self.parser.add_argument(
                '--interval',
                dest='logging_interval',
                default=100,
                type=int,
                help='Statistics logging interval. Will impact logging to the logger and '
                    'exporting to TensorBoard. Writing to the csv file is not impacted '
                    '(exports at every step). (DEFAULT: 100, i.e. logs every 100 episodes).')

            self.parser.add_argument(
                '--agree',
                dest='confirm',
                action='store_true',
                help='Request user confirmation just after loading the settings, '
                    'before starting the experiment. (DEFAULT: False)')

    def setup_experiment(self):
        """
        Setups a specific experiment.

        Base method:

            - Parses command line arguments.

            - Initializes logger with worker name.

            - Sets the 3 default config sections (training / validation / test) and sets their dataloaders params.

        .. note::

            Child classes should override this method, but still call its parent to draw the basic functionality \
            implemented here.


        """
        # Parse arguments.
        self.app_state.args, self.unparsed = self.parser.parse_known_args()

        # Initialize logger using the configuration.
        # For now do not add file handler, as path to logfile is not known yet.
        self.logger = logging.initialize_logger(self.name, False)

        # add empty sections
        self.config.add_default_params({"training": {'terminal_conditions': {}}})
        self.config.add_default_params({"validation": {}})
        self.config.add_default_params({"testing": {}})


    def add_statistics(self, stat_col):
        """
        Adds most elementary shared statistics to ``StatisticsCollector``: episode.

        :param stat_col: ``StatisticsCollector``.

        """
        # Add default statistics with formatting.
        stat_col.add_statistics('episode', '{:06d}')


    def add_aggregators(self, stat_agg):
        """
        Adds basic statistical aggregators to ``StatisticsAggregator``: episode \
        episodes_aggregated.

        :param stat_agg: ``StatisticsAggregator``.

        """
        # add 'aggregators' for the episode.
        #stat_agg.add_aggregator('epoch', '{:02d}')
        stat_agg.add_aggregator('episode', '{:06d}')
        # Number of aggregated episodes.
        stat_agg.add_aggregator('episodes_aggregated', '{:06d}')


    @abstractmethod
    def run_experiment(self):
        """
        Main function of the worker which executes a specific experiment.

        .. note::

            Abstract. Should be implemented in the subclasses.
        """


    def collect_all_statistics(self, problem_mgr, pipeline_mgr, data_dict, stat_col):
        """
        Function that collects statistics

        :param pipeline: Pipeline containing both problem and list of components.
        :type pipeline: ``configuration.pipeline.Pipeline``

        :param problem_mgr: Problem manager.

        :param data_dict: contains the batch of samples to pass through the pipeline.
        :type data_dict: ``DataDict``

        :param stat_col: statistics collector used for logging accuracy etc.
        :type stat_col: ``StatisticsCollector``

        """
        # Collect "local" statistics.
        stat_col['episode'] = self.app_state.episode
        if ('epoch' in stat_col) and (self.app_state.epoch is not None):
            stat_col['epoch'] = self.app_state.epoch

        # Collect rest of statistics.
        problem_mgr.problem.collect_statistics(stat_col, data_dict)
        pipeline_mgr.collect_statistics(stat_col, data_dict)

        

    def aggregate_all_statistics(self, problem_mgr, pipeline_mgr, stat_col, stat_agg):
        """
        Aggregates the collected statistics. Exports the aggregations to logger, csv and TB. \
        Empties statistics collector for the next episode.

        :param pipeline: Pipeline containing both problem and list of components.
        :type pipeline: ``configuration.pipeline.Pipeline``

        :param problem_mgr: Problem manager.

        :param stat_col: ``StatisticsCollector`` object.

        :param stat_agg: ``StatisticsAggregator`` object.
        """ 
        # Aggregate "local" statistics.
        if ('epoch' in stat_col) and ('epoch' in stat_agg) and (self.app_state.epoch is not None):
            stat_agg.aggregators['epoch'] = self.app_state.epoch
        stat_agg.aggregators['episode'] = self.app_state.episode
        stat_agg.aggregators['episodes_aggregated'] = len(stat_col['episode'])
        # Aggregate rest of statistics.
        problem_mgr.problem.aggregate_statistics(stat_col, stat_agg)
        pipeline_mgr.aggregate_statistics(stat_col, stat_agg)
    

    def export_all_statistics(self, stat_obj, tag='', export_to_log = True):
        """
        Export the statistics/aggregations to logger, csv and TB.

        :param stat_obj: ``StatisticsCollector`` or ``StatisticsAggregato`` object.

        :param tag: Additional tag that will be added to string exported to logger, optional (DEFAULT = '').
        :type tag: str

        :param export_to_log: If True, exports statistics to logger (DEFAULT: True)
        :type export_to_log: bool

        """ 
        # Log to logger
        if export_to_log:
            self.logger.info(stat_obj.export_to_string(tag))

        # Export to csv
        stat_obj.export_to_csv()

        # Export to TensorBoard.
        stat_obj.export_to_tensorboard()


    def set_random_seeds(self, section_name, config):
        """
        Set ``torch`` & ``NumPy`` random seeds from the ``ParamRegistry``: \
        If one was indicated, use it, or set a random one.

        :param section_name: Name of the section (for logging purposes only).
        :type section_name: str

        :param config: Section in config registry that will be changed \
            ("training" or "testing" only will be taken into account.)

        """
        # Set the random seeds: either from the loaded configuration or a default randomly selected one.
        config.add_default_params({"seed_numpy": -1})
        if config["seed_numpy"] == -1:
            seed = randrange(0, 2 ** 32)
            # Overwrite the config param!
            config.add_config_params({"seed_numpy": seed})

        self.logger.info("Setting numpy random seed in {} to: {}".format(section_name, config["seed_numpy"]))
        np.random.seed(config["seed_numpy"])

        config.add_default_params({"seed_torch": -1})
        if config["seed_torch"] == -1:
            seed = randrange(0, 2 ** 32)
            # Overwrite the config param!
            config.add_config_params({"seed_torch": seed})

        self.logger.info("Setting torch random seed in {} to: {}".format(section_name, config["seed_torch"]))
        torch.manual_seed(config["seed_torch"])
        torch.cuda.manual_seed_all(config["seed_torch"])
