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

__author__ = "Tomasz Kornuta, Vincent Marois, Younes Bouhadjar"

import os
import torch
from time import sleep
from datetime import datetime

import ptp.configuration.config_parsing as config_parsing
import ptp.utils.logger as logging

from ptp.workers.worker import Worker

from ptp.application.problem_manager import ProblemManager
from ptp.application.pipeline_manager import PipelineManager

from ptp.utils.statistics_collector import StatisticsCollector
from ptp.utils.statistics_aggregator import StatisticsAggregator


class Processor(Worker):
    """
    Defines the basic ``Processor``.

    If defining another type of Processor, it should subclass it.

    """

    def __init__(self, name="Processor"):
        """
        Calls the ``Worker`` constructor, adds some additional arguments to parser.

       :param name: Name of the worker (DEFAULT: "Processor").
       :type name: str

        """ 
        # Call base constructor to set up app state, registry and add default params.
        super(Processor, self).__init__(name)

        self.parser.add_argument(
            '--set',
            dest='set',
            type=str,
            default="testing",
            help='Name of the specific set (section containing problem) to be processed (DEFAULT: testing)')

    def setup_global_experiment(self):
        """
        Sets up the global test experiment for the ``Processor``:

            - Checks that the model to use exists

            - Checks that the configuration file exists

            - Creates the configuration

        The rest of the experiment setup is done in :py:func:`setup_individual_experiment()` \
        to allow for multiple tests suppport.

        """
        # Call base method to parse all command line arguments and add default sections.
        super(Processor, self).setup_experiment()
        
        # Retrieve checkpoint file and section
        chkpt_file = self.app_state.args.load_checkpoint
        self.set = self.app_state.args.set

        # Check if checkpoint file was indicated.
        if chkpt_file == "":
            print('Please pass path to and name of the file containing pipeline to be loaded as --load parameter')
            exit(-1)

        # Check if file with model exists.
        if not os.path.isfile(chkpt_file):
            print('Checkpoint file {} does not exist'.format(chkpt_file))
            exit(-2)

        # Extract path.
        self.abs_path, _ = os.path.split(os.path.dirname(os.path.expanduser(chkpt_file)))
        print(self.abs_path)

        # Check if config file was indicated by the user.
        if self.app_state.args.config != '':
            root_config = self.app_state.args.config
        else:
            # Use the "default one".
            root_config = os.path.join(self.abs_path, 'training_configuration.yml')

        # Check if configuration file exists.
        if not os.path.isfile(root_config):
            print('Config file {} does not exist'.format(root_config))
            exit(-3)

        # Check the presence of the CUDA-compatible devices.
        if self.app_state.args.use_gpu and (torch.cuda.device_count() == 0):
            self.logger.error("Cannot use GPU as there are no CUDA-compatible devices present in the system!")
            exit(-4)

        # Get the list of configurations which need to be loaded.
        configs_to_load = config_parsing.recurrent_config_parse(root_config, [], self.app_state.absolute_config_path)

        # Read the YAML files one by one - but in reverse order -> overwrite the first indicated config(s)
        config_parsing.reverse_order_config_load(self.config, configs_to_load)

        # -> At this point, the Config Registry contains the configuration loaded (and overwritten) from several files.

    def setup_individual_experiment(self):
        """
        Setup individual test experiment in the case of multiple tests, or the main experiment in the case of \
        one test experiment.

        - Set up the log directory path

        - Set random seeds

        - Creates the pipeline consisting of many components

        - Creates testing problem manager

        - Performs testing of compatibility of testing pipeline

        """

        # Get testing problem type.
        try:
            _ = self.config[self.set]['problem']['type']
        except KeyError:
            print("Error: Couldn't retrieve the problem 'type' from the '{}' section in the loaded configuration".format(self.set))
            exit(-5)

        # Get pipeline name.
        try:
            pipeline_name = self.config['pipeline']['name']
        except KeyError:
            print("Error: Couldn't retrieve the pipeline 'name' from the loaded configuration")
            exit(-6)
            
        # Prepare output paths for logging
        while True:
            # Dirty fix: if log_dir already exists, wait for 1 second and try again
            try:
                time_str = self.set+'_{0:%Y%m%d_%H%M%S}'.format(datetime.now())
                if self.app_state.args.savetag != '':
                    time_str = time_str + "_" + self.app_state.args.savetag
                self.app_state.log_dir = self.abs_path + '/' + time_str + '/'
                # Lowercase dir.
                self.app_state.log_dir = self.app_state.log_dir.lower()
                os.makedirs(self.app_state.log_dir, exist_ok=False)
            except FileExistsError:
                sleep(1)
            else:
                break

        # Set log dir.
        self.app_state.log_file = self.app_state.log_dir + 'processor.log'
        # Initialize logger in app state.
        self.app_state.logger = logging.initialize_logger("AppState")
        # Add handlers for the logfile to worker logger.
        logging.add_file_handler_to_logger(self.logger)
        self.logger.info("Logger directory set to: {}".format(self.app_state.log_dir ))

        # Set cpu/gpu types.
        self.app_state.set_types()

        # Set random seeds in the testing section.
        self.set_random_seeds(self.set, self.config[self.set])

        # Total number of detected errors.
        errors =0

        ################# TESTING PROBLEM ################# 

        # Build the used problem manager.
        self.pm = ProblemManager(self.set, self.config[self.set]) 
        errors += self.pm.build()


        # check if the maximum number of episodes is specified, if not put a
        # default equal to the size of the dataset (divided by the batch size)
        # So that by default, we loop over the test set once.
        max_test_episodes = len(self.pm)

        self.config[self.set]['problem'].add_default_params({'max_test_episodes': max_test_episodes})
        if self.config[self.set]["problem"]["max_test_episodes"] == -1:
            # Overwrite the config value!
            self.config[self.set]['problem'].add_config_params({'max_test_episodes': max_test_episodes})

        # Warn if indicated number of episodes is larger than an epoch size:
        if self.config[self.set]["problem"]["max_test_episodes"] > max_test_episodes:
            self.logger.warning('Indicated maximum number of episodes is larger than one epoch, reducing it.')
            self.config[self.set]['problem'].add_config_params({'max_test_episodes': max_test_episodes})

        self.logger.info("Setting the max number of episodes to: {}".format(
            self.config[self.set]["problem"]["max_test_episodes"]))

        ###################### PIPELINE ######################
        
        # Build the pipeline using the loaded configuration and global variables.
        self.pipeline = PipelineManager(pipeline_name, self.config['pipeline'])
        errors += self.pipeline.build()

        # Show pipeline.
        summary_str = self.pipeline.summarize_all_components_header()
        summary_str += self.pm.problem.summarize_io(self.set)
        summary_str += self.pipeline.summarize_all_components()
        self.logger.info(summary_str)

        # Check errors.
        if errors > 0:
            self.logger.error('Found {} errors, terminating execution'.format(errors))
            exit(-7)

        # Handshake definitions.
        self.logger.info("Handshaking testing pipeline")
        defs_testing = self.pm.problem.output_data_definitions()
        errors += self.pipeline.handshake(defs_testing)

        # Check errors.
        if errors > 0:
            self.logger.error('Found {} errors, terminating execution'.format(errors))
            exit(-2)

        # Check if there are any models in the pipeline.
        if len(self.pipeline.models) == 0:
            self.logger.error('Cannot proceed with training, as there are no trainable models in the pipeline')
            exit(-3)


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
                if os.path.isfile(pipeline_name):
                    # Load parameters from checkpoint.
                    self.pipeline.load(pipeline_name)
                else:
                    raise Exception("Couldn't load the checkpoint {} indicated in the {}: file does not exist".format(pipeline_name, msg))

            # Try to load the models parameters - one by one, if set so in the configuration file.
            self.pipeline.load_models()
            
        except KeyError:
            self.logger.error("File {} indicated in the {} seems not to be a valid model checkpoint".format(pipeline_name, msg))
            exit(-5)
        except Exception as e:
            self.logger.error(e)
            # Exit by following the logic: if user wanted to load the model but failed, then continuing the experiment makes no sense.
            exit(-6)


        # Log the model summaries.
        summary_str = self.pipeline.summarize_models_header()
        summary_str += self.pipeline.summarize_models()
        self.logger.info(summary_str)

        # Move the models in the pipeline to GPU.
        if self.app_state.args.use_gpu:
            self.pipeline.cuda()

        # Turn on evaluation mode.
        self.pipeline.eval()

        # Export and log configuration, optionally asking the user for confirmation.
        config_parsing.display_parsing_results(self.logger, self.app_state.args, self.unparsed)
        config_parsing.display_globals(self.logger, self.app_state.globalitems())
        config_parsing.export_experiment_configuration_to_yml(self.logger, self.app_state.log_dir, "training_configuration.yml", self.config, self.app_state.args.confirm)

    def initialize_statistics_collection(self):
        """
        Function initializes all statistics collectors and aggregators used by a given worker,
        creates output files etc.
        """
        # Create statistics collector.
        self.stat_col = StatisticsCollector()
        self.add_statistics(self.stat_col)
        self.pm.problem.add_statistics(self.stat_col)
        self.pipeline.add_statistics(self.stat_col)
        # Create the csv file to store the statistics.
        self.pm_batch_stats_file = self.stat_col.initialize_csv_file(self.app_state.log_dir, self.set+'_statistics.csv')

        # Create statistics aggregator.
        self.stat_agg = StatisticsAggregator()
        self.add_aggregators(self.stat_agg)
        self.pm.problem.add_aggregators(self.stat_agg)
        self.pipeline.add_aggregators(self.stat_agg)
        # Create the csv file to store the statistic aggregations.
        # Will contain a single row with aggregated statistics.
        self.pm_set_stats_file = self.stat_agg.initialize_csv_file(self.app_state.log_dir, self.set+'_set_agg_statistics.csv')

    def finalize_statistics_collection(self):
        """
        Finalizes statistics collection, closes all files etc.
        """
        # Close all files.
        self.pm_batch_stats_file.close()
        self.pm_set_stats_file.close()

    def run_experiment(self):
        """
        Main function of the ``Processor``: Test the loaded model over the set.

        Iterates over the ``DataLoader`` for a maximum number of episodes equal to the set size.

        The function does the following for each episode:

            - Forwards pass of the model,
            - Logs statistics & accumulates loss,
            - Activate visualization if set.


        """
        # Initialize tensorboard and statistics collection.
        self.initialize_statistics_collection()

        num_samples = len(self.pm)

        self.logger.info('Processing the entire set ({} samples in {} episodes)'.format(
            num_samples, len(self.pm.dataloader)))

        try:
            # Run in no_grad mode.
            with torch.no_grad():
                # Reset the counter.
                self.app_state.episode = -1

                # Inform the problem manager that epoch has started.
                self.pm.initialize_epoch()

                for batch in self.pm.dataloader:
                    # Increment counter.
                    self.app_state.episode += 1
                    # Terminal condition 0: max test episodes reached.
                    if self.app_state.episode == self.config[self.set]["problem"]["max_test_episodes"]:
                        break

                    # Forward pass.
                    self.pipeline.forward(batch)
                    # Collect the statistics.
                    self.collect_all_statistics(self.pm, self.pipeline, batch, self.stat_col)

                    # Export to csv - at every step.
                    self.stat_col.export_to_csv()

                    # Log to logger - at logging frequency.
                    if self.app_state.episode % self.app_state.args.logging_interval == 0:
                        self.logger.info(self.stat_col.export_to_string('[Partial]'))

                    # move to next episode.
                    self.app_state.episode += 1

                # End for.
                # Inform the problem managers that the epoch has ended.
                self.pm.finalize_epoch()

                self.logger.info('\n' + '='*80)
                self.logger.info('Processing finished')

                # Aggregate statistics for the whole set.
                self.aggregate_all_statistics(self.pm, self.pipeline, self.stat_col, self.stat_agg)

                # Export aggregated statistics.
                self.export_all_statistics(self.stat_agg, '[Full Set]')


        except SystemExit as e:
            # the training did not end properly
            self.logger.error('Experiment interrupted because {}'.format(e))
        except KeyboardInterrupt:
            # the training did not end properly
            self.logger.error('Experiment interrupted!')
        finally:
            # Finalize statistics collection.
            self.finalize_statistics_collection()
            self.logger.info("Experiment logged to: {}".format(self.app_state.log_dir))
        

def main():
    """
    Entry point function for the ``Processor``.

    """
    processor = Processor()
    # parse args, load configuration and create all required objects.
    processor.setup_global_experiment()

    # finalize the experiment setup
    processor.setup_individual_experiment()

    # run the experiment
    processor.run_experiment()


if __name__ == '__main__':
    main()
