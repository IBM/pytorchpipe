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

import os
import yaml

import torch
import logging
import logging.config
import argparse
import numpy as np
from random import randrange
from abc import abstractmethod

# Import configuration.
from ptp.configuration.app_state import AppState
from ptp.configuration.param_interface import ParamInterface


class Worker(object):
    """
    Base abstract class for the workers.
    All base workers should subclass it and override the relevant methods.
    """

    def __init__(self, name, add_default_parser_args = True):
        """
        Base constructor for all workers:

            - Initializes the AppState singleton:

                >>> self.app_state = AppState()

            - Initializes the Parameter Registry:

                >>> self.params = ParamInterface()

            - Defines the logger:

                >>> self.logger = logging.getLogger(name=self.name)

            - Creates parser and adds default worker command line arguments.

        :param name: Name of the worker.
        :type name: str

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
        self.params = ParamInterface()

        # Initialize logger using the configuration.
        self.initialize_logger()

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

    def initialize_logger(self):
        """
        Initializes the logger, with a specific configuration:

        >>> logger_config = {'version': 1,
        >>>                  'disable_existing_loggers': False,
        >>>                  'formatters': {
        >>>                      'simple': {
        >>>                          'format': '[%(asctime)s] - %(levelname)s - %(name)s >>> %(message)s',
        >>>                          'datefmt': '%Y-%m-%d %H:%M:%S'}},
        >>>                  'handlers': {
        >>>                      'console': {
        >>>                          'class': 'logging.StreamHandler',
        >>>                          'level': 'INFO',
        >>>                          'formatter': 'simple',
        >>>                          'stream': 'ext://sys.stdout'}},
        >>>                  'root': {'level': 'DEBUG',
        >>>                           'handlers': ['console']}}

        """
        # Load the default logger configuration.
        logger_config = {'version': 1,
                         'disable_existing_loggers': False,
                         'formatters': {
                             'simple': {
                                 'format': '[%(asctime)s] - %(levelname)s - %(name)s >>> %(message)s',
                                 'datefmt': '%Y-%m-%d %H:%M:%S'}},
                         'handlers': {
                             'console': {
                                 'class': 'logging.StreamHandler',
                                 'level': 'INFO',
                                 'formatter': 'simple',
                                 'stream': 'ext://sys.stdout'}},
                         'root': {'level': 'DEBUG',
                                  'handlers': ['console']}}

        logging.config.dictConfig(logger_config)

        # Create the Logger, set its label and logging level.
        self.logger = logging.getLogger(name=self.name)


    def add_file_handler_to_logger(self, logfile):
        """
        Add a ``logging.FileHandler`` to the logger of the current ``Worker``.

        Specifies a ``logging.Formatter``:

            >>> logging.Formatter(fmt='[%(asctime)s] - %(levelname)s - %(name)s >>> %(message)s',
            >>>                   datefmt='%Y-%m-%d %H:%M:%S')


        :param logfile: File used by the ``FileHandler``.

        """
        # create file handler which logs even DEBUG messages
        fh = logging.FileHandler(logfile)

        # set logging level for this file
        fh.setLevel(logging.DEBUG)

        # create formatter and add it to the handlers
        formatter = logging.Formatter(fmt='[%(asctime)s] - %(levelname)s - %(name)s >>> %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)

        # add the handler to the logger
        self.logger.addHandler(fh)


    def setup_experiment(self):
        """
        Setups a specific experiment.

        Base method:

            - Parses command line arguments.

            - Sets the 3 default sections (training / validation / test) and sets their dataloaders params.

        .. note::

            Child classes should override this method, but still call its parent to draw the basic functionality \
            implemented here.


        """
        # Parse arguments.
        self.app_state.args, self.unparsed = self.parser.parse_known_args()

        # Set logger depending on the settings.
        self.logger.setLevel(getattr(logging, self.app_state.args.log_level.upper(), None))

        # add empty sections
        self.params.add_default_params({"training": {'terminal_conditions': {}}})
        self.params.add_default_params({"validation": {}})
        self.params.add_default_params({"testing": {}})


    def display_parsing_results(self):
        """
        Displays the properly & improperly parsed arguments (if any).

        """
        # Log the parsed flags.
        flags_str = 'Properly parsed command line arguments: \n'
        flags_str += '='*80 + '\n'
        for arg in vars(self.app_state.args): 
            flags_str += "{}= {} \n".format(arg, getattr(self.app_state.args, arg))
        flags_str += '='*80 + '\n'
        self.logger.info(flags_str)

        # Log the unparsed flags if any.
        if self.unparsed:
            flags_str = 'Invalid command line arguments: \n'
            flags_str += '='*80 + '\n'
            for arg in self.unparsed: 
                flags_str += "{} \n".format(arg)
            flags_str += '='*80 + '\n'
            self.logger.warning(flags_str)


    def export_experiment_configuration(self, log_dir, filename, user_confirm):
        """
        Dumps the configuration to ``yaml`` file.

        :param log_dir: Directory used to host log files (such as the collected statistics).
        :type log_dir: str

        :param filename: Name of the ``yaml`` file to write to.
        :type filename: str

        :param user_confirm: Whether to request user confirmation.
        :type user_confirm: bool


        """
        # -> At this point, all configuration for experiment is complete.

        # Display results of parsing.
        self.display_parsing_results()

        # Log the resulting training configuration.
        conf_str = 'Final parameter registry configuration:\n'
        conf_str += '='*80 + '\n'
        conf_str += yaml.safe_dump(self.params.to_dict(), default_flow_style=False)
        conf_str += '='*80 + '\n'
        self.logger.info(conf_str)

        # Save the resulting configuration into a .yaml settings file, under log_dir
        with open(log_dir + filename, 'w') as yaml_backup_file:
            yaml.dump(self.params.to_dict(), yaml_backup_file, default_flow_style=False)

        # Ask for confirmation - optional.
        if user_confirm:
            try:
                input('Press <Enter> to confirm and start the experiment\n')
            except KeyboardInterrupt:
                exit(0)            


    def add_statistics(self, stat_col):
        """
        Adds most elementary shared statistics to ``StatisticsCollector``: episode.

        :param stat_col: ``StatisticsCollector``.

        """
        # Add default statistics with formatting.
        #stat_col.add_statistic('epoch', '{:02d}')
        stat_col.add_statistic('episode', '{:06d}')


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


    def recurrent_config_parse(self, configs: str, configs_parsed: list, abs_config_path: str):
        """
        Parses names of configuration files in a recursive manner, i.e. \
        by looking for ``default_config`` sections and trying to load and parse those \
        files one by one.

        :param configs: String containing names of configuration files (with paths), separated by comas.
        :type configs: str

        :param configs_parsed: Configurations that were already parsed (so we won't parse them many times).
        :type configs_parsed: list

        :param abs_config_path: Absolute path to ``config`` directory.

        :return: list of parsed configuration files.

        """
        # Split and remove spaces.
        configs_to_parse = configs.replace(" ", "").split(',')

        # Terminal condition.
        while len(configs_to_parse) > 0:

            # Get config.
            config = configs_to_parse.pop(0)
            abs_config = abs_config_path + config

            # Skip empty names (after lose comas).
            if config == '':
                continue
            print("Info: Parsing the {} configuration file".format(abs_config))

            # Check if it was already loaded.
            if config in configs_parsed:
                print('Warning: Configuration file {} already parsed - skipping'.format(abs_config))
                continue

            # Check if file exists.
            if not os.path.isfile(abs_config):
                print('Error: Configuration file {} does not exist'.format(abs_config))
                exit(-1)

            try:
                # Open file and get parameter dictionary.
                with open(abs_config, 'r') as stream:
                    param_dict = yaml.safe_load(stream)
            except yaml.YAMLError as e:
                print("Error: Couldn't properly parse the {} configuration file".format(abs_config))
                print('yaml.YAMLERROR:', e)
                exit(-1)

            # Remember that we loaded that config.
            configs_parsed.append(config)

            # Check if there are any default configs to load.
            if 'default_configs' in param_dict:
                # If there are - recursion!
                configs_parsed = self.recurrent_config_parse(
                    param_dict['default_configs'], configs_parsed, abs_config_path)

        # Done, return list of loaded configs.
        return configs_parsed

    def recurrent_config_load(self,configs_to_load, abs_config_path):
        for config in reversed(configs_to_load):
            # Load params from YAML file.
            self.params.add_config_params_from_yaml(abs_config_path + config)
            print('Info: Loaded configuration from file {}'.format(abs_config_path + config))


    def collect_all_statistics(self, problem_mgr, pipeline_mgr, data_dict, stat_col, episode, epoch=None):
        """
        Function that collects statistics

        :param pipeline: Pipeline containing both problem and list of components.
        :type pipeline: ``configuration.pipeline.Pipeline``

        :param problem_mgr: Problem manager.

        :param data_dict: contains the batch of samples to pass through the pipeline.
        :type data_dict: ``DataDict``

        :param stat_col: statistics collector used for logging accuracy etc.
        :type stat_col: ``StatisticsCollector``

        :param episode: current episode index
        :type episode: int

        :param epoch: current epoch index (DEFAULT: None)
        :type epoch: int, optional

        """
        # Collect "local" statistics.
        stat_col['episode'] = episode
        if ('epoch' in stat_col) and (epoch is not None):
            stat_col['epoch'] = epoch

        # Collect rest of statistics.
        problem_mgr.problem.collect_statistics(stat_col, data_dict)
        pipeline_mgr.collect_statistics(stat_col, data_dict)

        

    def aggregate_all_statistics(self, problem_mgr, pipeline_mgr, stat_col, stat_agg, episode, epoch=None):
        """
        Aggregates the collected statistics. Exports the aggregations to logger, csv and TB. \
        Empties statistics collector for the next episode.

        :param pipeline: Pipeline containing both problem and list of components.
        :type pipeline: ``configuration.pipeline.Pipeline``

        :param problem_mgr: Problem manager.

        :param stat_col: ``StatisticsCollector`` object.

        :param stat_agg: ``StatisticsAggregator`` object.

        :param episode: current episode index
        :type episode: int

        :param epoch: current epoch index (DEFAULT: None)
        :type epoch: int, optional

        """ 
        # Aggregate "local" statistics.
        if ('epoch' in stat_col) and ('epoch' in stat_agg) and (epoch is not None):
            stat_agg.aggregators['epoch'] = epoch
        stat_agg.aggregators['episode'] = episode
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


    def set_random_seeds(self, section_name, params):
        """
        Set ``torch`` & ``NumPy`` random seeds from the ``ParamRegistry``: \
        If one was indicated, use it, or set a random one.

        :param section_name: Name of the section (for logging purposes only).
        :type section_name: str

        :param params: Section in config/param registry that will be changed \
            ("training" or "testing" only will be taken into account.)

        """
        # Set the random seeds: either from the loaded configuration or a default randomly selected one.
        params.add_default_params({"seed_numpy": -1})
        if params["seed_numpy"] == -1:
            seed = randrange(0, 2 ** 32)
            # Overwrite the config param!
            params.add_config_params({"seed_numpy": seed})

        self.logger.info("Setting numpy random seed in {} to: {}".format(section_name, params["seed_numpy"]))
        np.random.seed(params["seed_numpy"])

        params.add_default_params({"seed_torch": -1})
        if params["seed_torch"] == -1:
            seed = randrange(0, 2 ** 32)
            # Overwrite the config param!
            params.add_config_params({"seed_torch": seed})

        self.logger.info("Setting torch random seed in {} to: {}".format(section_name, params["seed_torch"]))
        torch.manual_seed(params["seed_torch"])
        torch.cuda.manual_seed_all(params["seed_torch"])
