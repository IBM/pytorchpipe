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

__author__ = "Vincent Marois, Tomasz Kornuta"

from os import path,makedirs
import yaml
import torch
from time import sleep
from datetime import datetime

import ptp.configuration.config_parsing as config_parse
import ptp.utils.logger as logging

from ptp.workers.worker import Worker

from ptp.application.problem_manager import ProblemManager
from ptp.application.pipeline_manager import PipelineManager

from ptp.utils.statistics_collector import StatisticsCollector
from ptp.utils.statistics_aggregator import StatisticsAggregator


class Trainer(Worker):
    """
    Base class for the trainers.

    Iterates over epochs on the dataset.

    All other types of trainers (e.g. ``OnlineTrainer`` & ``OfflineTrainer``) should subclass it.

    """

    def __init__(self, name, class_type):
        """
        Base constructor for all trainers:

            - Adds default trainer command line arguments

        :param name: Name of the worker
        :type name: str

        :param class_type: Class type of the component.

        """ 
        # Call base constructor to set up app state, registry and add default arguments.
        super(Trainer, self).__init__(name, class_type)

        # Add arguments to the specific parser.
        # These arguments will be shared by all basic trainers.
        self.parser.add_argument(
            '--tensorboard',
            action='store',
            dest='tensorboard', choices=[0, 1, 2],
            type=int,
            help="If present, enable logging to TensorBoard. Available log levels:\n"
                "0: Log the collected statistics.\n"
                "1: Add the histograms of the model's biases & weights (Warning: Slow).\n"
                "2: Add the histograms of the model's biases & weights gradients "
                "(Warning: Even slower).")

        self.parser.add_argument(
            '--save',
            dest='save_intermediate',
            action='store_true',
            help='Setting to true results in saving intermediate models during training (DEFAULT: False)')


    def setup_experiment(self):
        """
        Sets up experiment of all trainers:

        - Calls base class setup_experiment to parse the command line arguments,

        - Loads the config file(s)

        - Set up the log directory path

        - Add a ``FileHandler`` to the logger

        - Set random seeds

        - Creates the pipeline consisting of many components

        - Creates training problem manager

        - Handles curriculum learning if indicated

        - Creates validation problem manager

        - Set optimizer

        - Performs testing of compatibility of both training and validation problems and created pipeline.

        """
        # Call base method to parse all command line arguments and add default sections.
        super(Trainer, self).setup_experiment()

        # Check the presence of the CUDA-compatible devices.
        if self.app_state.args.use_gpu and (torch.cuda.device_count() == 0):
            self.logger.error("Cannot use GPU as there are no CUDA-compatible devices present in the system!")
            exit(-1)

        # Check if config file was selected.
        if self.app_state.args.config == '':
            print('Please pass configuration file(s) as --c parameter')
            exit(-2)

        # Split and make them absolute.
        root_configs = self.app_state.args.config.replace(" ", "").split(',')
        # If there are - expand them to absolute paths.
        abs_root_configs = [path.expanduser(config) for config in root_configs]
        
        # Get the list of configurations which need to be loaded.
        configs_to_load = config_parse.recurrent_config_parse(abs_root_configs, [], self.app_state.absolute_config_path)

        # Read the YAML files one by one - but in reverse order -> overwrite the first indicated config(s)
        config_parse.reverse_order_config_load(self.config, configs_to_load)

        # -> At this point, the Param Registry contains the configuration loaded (and overwritten) from several files.
        # Log the resulting training configuration.
        conf_str = 'Loaded (initial) configuration:\n'
        conf_str += '='*80 + '\n'
        conf_str += yaml.safe_dump(self.config.to_dict(), default_flow_style=False)
        conf_str += '='*80 + '\n'
        print(conf_str)

        # Get training problem name.
        try:
            training_problem_type = self.config['training']['problem']['type']
        except KeyError:
            print("Error: Couldn't retrieve the problem 'type' from the 'training' section in the loaded configuration")
            exit(-1)

        # Get validation problem name
        try:
            _ = self.config['validation']['problem']['type']
        except KeyError:
            print("Error: Couldn't retrieve the problem 'type' from the 'validation' section in the loaded configuration")
            exit(-1)

        # Get pipeline name.
        try:
            pipeline_name = self.config['pipeline']['name']
        except KeyError:
            # Using name of the first configuration file from command line.
            basename = path.basename(root_configs[0])
            # Take config filename without extension.
            pipeline_name = path.splitext(basename)[0] 
            # Set pipeline name, so processor can use it afterwards.
            self.config['pipeline'].add_config_params({'name': pipeline_name})

        # Prepare the output path for logging
        while True:  # Dirty fix: if log_dir already exists, wait for 1 second and try again
            try:
                time_str = '{0:%Y%m%d_%H%M%S}'.format(datetime.now())
                if self.app_state.args.savetag != '':
                    time_str = time_str + "_" + self.app_state.args.savetag
                self.app_state.log_dir = path.expanduser(self.app_state.args.expdir) + '/' + training_problem_type + '/' + pipeline_name + '/' + time_str + '/'
                # Lowercase dir.
                self.app_state.log_dir = self.app_state.log_dir.lower()
                makedirs(self.app_state.log_dir, exist_ok=False)
            except FileExistsError:
                sleep(1)
            else:
                break

        # Set log dir.
        self.app_state.log_file = self.app_state.log_dir + 'trainer.log'
        # Initialize logger in app state.
        self.app_state.logger = logging.initialize_logger("AppState")
        # Add handlers for the logfile to worker logger.
        logging.add_file_handler_to_logger(self.logger)
        self.logger.info("Logger directory set to: {}".format(self.app_state.log_dir))

        # Set cpu/gpu types.
        self.app_state.set_types()

        # Models dir.
        self.checkpoint_dir = self.app_state.log_dir + 'checkpoints/'
        makedirs(self.checkpoint_dir, exist_ok=False)

        # Set random seeds in the training section.
        self.set_random_seeds('training', self.config['training'])

        # Total number of detected errors.
        errors =0

        ################# TRAINING PROBLEM ################# 

        # Build training problem manager.
        self.training = ProblemManager('training', self.config['training']) 
        errors += self.training.build()
        
        # parse the curriculum learning section in the loaded configuration.
        if 'curriculum_learning' in self.config['training']:

            # Initialize curriculum learning - with values from loaded configuration.
            self.training.problem.curriculum_learning_initialize(self.config['training']['curriculum_learning'])

            # If the 'must_finish' key is not present in config then then it will be finished by default
            self.config['training']['curriculum_learning'].add_default_params({'must_finish': True})

            self.must_finish_curriculum = self.config['training']['curriculum_learning']['must_finish']
            self.logger.info("Curriculum Learning activated")

        else:
            # If not using curriculum learning then it does not have to be finished.
            self.must_finish_curriculum = False
            self.curric_done = True

        ################# VALIDATION PROBLEM ################# 
        
        # Build validation problem manager.
        self.validation = ProblemManager('validation', self.config['validation'])
        errors += self.validation.build()

        ###################### PIPELINE ######################
        
        # Build the pipeline using the loaded configuration.
        self.pipeline = PipelineManager(pipeline_name, self.config['pipeline'])
        errors += self.pipeline.build()

        # Check errors.
        if errors > 0:
            self.logger.error('Found {} errors, terminating execution'.format(errors))
            exit(-2)

        # Show pipeline.
        summary_str = self.pipeline.summarize_all_components_header()
        summary_str += self.training.problem.summarize_io("training")
        summary_str += self.validation.problem.summarize_io("validation")
        summary_str += self.pipeline.summarize_all_components()
        self.logger.info(summary_str)
        
        # Handshake definitions.
        self.logger.info("Handshaking training pipeline")
        defs_training = self.training.problem.output_data_definitions()
        errors += self.pipeline.handshake(defs_training)

        self.logger.info("Handshaking validation pipeline")
        defs_valid = self.validation.problem.output_data_definitions()
        errors += self.pipeline.handshake(defs_valid)

        # Check errors.
        if errors > 0:
            self.logger.error('Found {} errors, terminating execution'.format(errors))
            exit(-2)

        ################## MODEL LOAD/FREEZE #################

        # Load the pretrained models params from checkpoint.
        try: 
            # Check command line arguments, then check load option in config.
            if self.app_state.args.load_checkpoint != "":
                pipeline_name = self.app_state.args.load_checkpoint
                msg = "command line (--load)"
            elif "load" in self.config['pipeline']:
                pipeline_name = self.config['pipeline']['load']
                msg = "'pipeline' section of the configuration file"
            else:
                pipeline_name = ""
            # Try to load the model.
            if pipeline_name != "":
                if path.isfile(pipeline_name):
                    # Load parameters from checkpoint.
                    self.pipeline.load(pipeline_name)
                else:
                    raise Exception("Couldn't load the checkpoint {} indicated in the {}: file does not exist".format(pipeline_name, msg))
                # If we succeeded, we do not want to load the models from the file anymore!
            else:
                # Try to load the models parameters - one by one, if set so in the configuration file.
                self.pipeline.load_models()

        except KeyError:
            self.logger.error("File {} indicated in the {} seems not to be a valid model checkpoint".format(pipeline_name, msg))
            exit(-5)
        except Exception as e:
            self.logger.error(e)
            # Exit by following the logic: if user wanted to load the model but failed, then continuing the experiment makes no sense.
            exit(-6)

        # Finally, freeze the models (that the user wants to freeze).
        self.pipeline.freeze_models()

        # Log the model summaries.
        summary_str = self.pipeline.summarize_models_header()
        summary_str += self.pipeline.summarize_models()
        self.logger.info(summary_str)

        # Move the models in the pipeline to GPU.
        if self.app_state.args.use_gpu:
            self.pipeline.cuda()        

        ################# OPTIMIZER ################# 

        # Set the optimizer.
        optimizer_conf = dict(self.config['training']['optimizer'])
        optimizer_name = optimizer_conf['name']
        del optimizer_conf['name']

        # Check if there are any models in the pipeline.
        if len(list(filter(lambda p: p.requires_grad, self.pipeline.parameters()))) == 0:
            self.logger.error('Cannot proceed with training, as there are no trainable models in the pipeline (or all models are frozen)')
            exit(-7)

        # Instantiate the optimizer and filter the model parameters based on if they require gradients.
        self.optimizer = getattr(torch.optim, optimizer_name)(
            filter(lambda p: p.requires_grad, self.pipeline.parameters()), **optimizer_conf)

        log_str = 'Optimizer:\n' + '='*80 + "\n"
        log_str += "  Name: " + optimizer_name + "\n"
        log_str += "  Params: {}".format(optimizer_conf)

        self.logger.info(log_str)

    def add_statistics(self, stat_col):
        """
        Calls base method and adds epoch statistics to ``StatisticsCollector``.

        :param stat_col: ``StatisticsCollector``.

        """
        # Add loss and episode.
        super(Trainer, self).add_statistics(stat_col)

        # Add default statistics with formatting.
        stat_col.add_statistics('epoch', '{:02d}')


    def add_aggregators(self, stat_agg):
        """
        Adds basic aggregators to to ``StatisticsAggregator`` and extends them with: epoch.

        :param stat_agg: ``StatisticsAggregator``.

        """
        # Add basic aggregators.
        super(Trainer, self).add_aggregators(stat_agg)

        # add 'aggregators' for the epoch.
        stat_agg.add_aggregator('epoch', '{:02d}')


    def initialize_statistics_collection(self):
        """
        - Initializes all ``StatisticsCollectors`` and ``StatisticsAggregators`` used by a given worker: \

            - For training statistics (adds the statistics of the model & problem),
            - For validation statistics (adds the statistics of the model & problem).

        - Creates the output files (csv).

        """
        # TRAINING.
        # Create statistics collector for training.
        self.training_stat_col = StatisticsCollector()
        self.add_statistics(self.training_stat_col)
        self.training.problem.add_statistics(self.training_stat_col)
        self.pipeline.add_statistics(self.training_stat_col)
        # Create the csv file to store the training statistics.
        self.training_batch_stats_file = self.training_stat_col.initialize_csv_file(self.app_state.log_dir, 'training_statistics.csv')

        # Create statistics aggregator for training.
        self.training_stat_agg = StatisticsAggregator()
        self.add_aggregators(self.training_stat_agg)
        self.training.problem.add_aggregators(self.training_stat_agg)
        self.pipeline.add_aggregators(self.training_stat_agg)
        # Create the csv file to store the training statistic aggregations.
        self.training_set_stats_file = self.training_stat_agg.initialize_csv_file(self.app_state.log_dir, 'training_set_agg_statistics.csv')

        # VALIDATION.
        # Create statistics collector for validation.
        self.validation_stat_col = StatisticsCollector()
        self.add_statistics(self.validation_stat_col)
        self.validation.problem.add_statistics(self.validation_stat_col)
        self.pipeline.add_statistics(self.validation_stat_col)
        # Create the csv file to store the validation statistics.
        self.validation_batch_stats_file = self.validation_stat_col.initialize_csv_file(self.app_state.log_dir, 'validation_statistics.csv')

        # Create statistics aggregator for validation.
        self.validation_stat_agg = StatisticsAggregator()
        self.add_aggregators(self.validation_stat_agg)
        self.validation.problem.add_aggregators(self.validation_stat_agg)
        self.pipeline.add_aggregators(self.validation_stat_agg)
        # Create the csv file to store the validation statistic aggregations.
        self.validation_set_stats_file = self.validation_stat_agg.initialize_csv_file(self.app_state.log_dir, 'validation_set_agg_statistics.csv')


    def finalize_statistics_collection(self):
        """
        Finalizes the statistics collection by closing the csv files.

        """
        # Close all files.
        self.training_batch_stats_file.close()
        self.training_set_stats_file.close()
        self.validation_batch_stats_file.close()
        self.validation_set_stats_file.close()


    def initialize_tensorboard(self):
        """
        Initializes the TensorBoard writers, and log directories.

        """
        # Create TensorBoard outputs - if TensorBoard is supposed to be used.
        if self.app_state.args.tensorboard is not None:
            from tensorboardX import SummaryWriter
            self.training_batch_writer = SummaryWriter(self.app_state.log_dir + '/training')
            self.training_stat_col.initialize_tensorboard(self.training_batch_writer)

            self.training_set_writer = SummaryWriter(self.app_state.log_dir + '/training_set_agg')
            self.training_stat_agg.initialize_tensorboard(self.training_set_writer)
            
            self.validation_batch_writer = SummaryWriter(self.app_state.log_dir + '/validation')
            self.validation_stat_col.initialize_tensorboard(self.validation_batch_writer)

            self.validation_set_writer = SummaryWriter(self.app_state.log_dir + '/validation_set_agg')
            self.validation_stat_agg.initialize_tensorboard(self.validation_set_writer)
        else:
            self.training_batch_writer = None
            self.training_set_writer = None
            self.validation_batch_writer = None
            self.validation_set_writer = None

    def finalize_tensorboard(self):
        """ 
        Finalizes the operation of TensorBoard writers by closing them.
        """
        # Close the TensorBoard writers.
        if self.training_batch_writer is not None:
            self.training_batch_writer.close()
        if self.training_set_writer is not None:
            self.training_set_writer.close()
        if self.validation_batch_writer is not None:
            self.validation_batch_writer.close()
        if self.validation_set_writer is not None:
            self.validation_set_writer.close()

    def validate_on_batch(self, valid_batch):
        """
        Performs a validation of the model using the provided batch.

        Additionally logs results (to files, TensorBoard) and handles visualization.

        :param valid_batch: data batch generated by the problem and used as input to the model.
        :type valid_batch: ``DataDict``

        :return: Validation loss.

        """
        # Turn on evaluation mode.
        self.pipeline.eval()
        # Empty the statistics collector.
        self.validation_stat_col.empty()

        # Compute the validation loss using the provided data batch.
        with torch.no_grad():
            # Forward pass.
            self.pipeline.forward(valid_batch)
            # Collect the statistics.
            self.collect_all_statistics(self.validation, self.pipeline, valid_batch, self.validation_stat_col)

        # Export collected statistics.
        self.export_all_statistics(self.validation_stat_col, '[Partial Validation]')

    def validate_on_set(self):
        """
        Performs a validation of the model on the whole validation set, using the validation ``DataLoader``.

        Iterates over the entire validation set (through the `DataLoader``), aggregates the collected statistics \
        and logs that to the console, csv and TensorBoard (if set).

        """
        # Get number of samples.
        num_samples = len(self.validation)
        
        self.logger.info('Validating over the entire validation set ({} samples in {} episodes)'.format(
            num_samples, len(self.validation.dataloader)))

        # Turn on evaluation mode.
        self.pipeline.eval()

        # Reset the statistics.
        self.validation_stat_col.empty()

        # Remember global episode number.
        old_episode = self.app_state.episode

        with torch.no_grad():
            for ep, valid_batch in enumerate(self.validation.dataloader):

                self.app_state.episode = ep
                # Forward pass.
                self.pipeline.forward(valid_batch)
                # Collect the statistics.
                self.collect_all_statistics(self.validation, self.pipeline, valid_batch,
                        self.validation_stat_col)

        # Revert to global episode number.
        self.app_state.episode = old_episode

        # Aggregate statistics for the whole set.
        self.aggregate_all_statistics(self.validation, self.pipeline,
            self.validation_stat_col, self.validation_stat_agg)

        # Export aggregated statistics.
        self.export_all_statistics(self.validation_stat_agg, '[Full Validation]')


if __name__ == '__main__':
    print("The trainer.py file contains only an abstract base class. Please try to use the \
online_trainer (mip-online-trainer) or  offline_trainer (mip-offline-trainer) instead.")
