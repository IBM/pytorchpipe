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

from ptp.workers.worker import Worker

from ptp.configuration.problem_manager import ProblemManager
from ptp.configuration.pipeline_manager import PipelineManager

from ptp.utils.statistics_collector import StatisticsCollector
from ptp.utils.statistics_aggregator import StatisticsAggregator


class Tester(Worker):
    """
    Defines the basic ``Tester``.

    If defining another type of tester, it should subclass it.

    """

    def __init__(self, name="Tester"):
        """
        Calls the ``Worker`` constructor, adds some additional params to parser.

       :param name: Name of the worker (DEFAULT: "Tester").
       :type name: str

        """ 
        # Call base constructor to set up app state, registry and add default params.
        super(Tester, self).__init__(name)


    def setup_global_experiment(self):
        """
        Sets up the global test experiment for the ``Tester``:

            - Checks that the model to use exists on file:

                >>> if not os.path.isfile(flags.model)

            - Checks that the configuration file exists:

                >>> if not os.path.isfile(config_file)

            - Create the configuration:

                >>> self.params.add_config_params_from_yaml(config)

        The rest of the experiment setup is done in :py:func:`setup_individual_experiment()` \
        to allow for multiple tests suppport.

        """
        # Call base method to parse all command line arguments and add default sections.
        super(Tester, self).setup_experiment()

        chkpt_file = self.app_state.args.load_checkpoint

        # Check if checkpoint file was indicated.
        if chkpt_file == "":
            print('Please pass path to and name of the file containing pipeline to be loaded as --load parameter')
            exit(-1)


        # Check if file with model exists.
        if not os.path.isfile(chkpt_file):
            print('Checkpoint file {} does not exist'.format(chkpt_file))
            exit(-2)

        # Extract path.
        self.abs_path, _ = os.path.split(os.path.dirname(os.path.abspath(chkpt_file)))

        # Check if config file was indicated by the user.
        if self.app_state.args.config != '':
            config_file = self.app_state.args.config
        else:
            # Use the "default one".
            config_file = self.abs_path + '/training_configuration.yaml'

        # Check if configuration file exists.
        if not os.path.isfile(config_file):
            print('Config file {} does not exist'.format(config_file))
            exit(-3)

        # Check the presence of the CUDA-compatible devices.
        if self.app_state.args.use_gpu and (torch.cuda.device_count() == 0):
            self.logger.error("Cannot use GPU as there are no CUDA-compatible devices present in the system!")
            exit(-4)

        # Set cpu/gpu types.
        self.app_state.set_types()

        # Get the list of configurations which need to be loaded.
        configs_to_load = self.recurrent_config_parse(config_file, [])

        # Read the YAML files one by one - but in reverse order -> overwrite the first indicated config(s)
        self.recurrent_config_load(configs_to_load)

        # -> At this point, the Param Registry contains the configuration loaded (and overwritten) from several files.

    def setup_individual_experiment(self):
        """
        Setup individual test experiment in the case of multiple tests, or the main experiment in the case of \
        one test experiment.

        - Set up the log directory path:

            >>> os.makedirs(self.log_dir, exist_ok=False)

        - Add a FileHandler to the logger (defined in BaseWorker):

            >>>  self.logger.addHandler(fh)

        - Set random seeds:

            >>>  self.set_random_seeds('testing', self.params['testing'])

        - Creates the pipeline consisting of many components

        - Creates testing problem manager

        - Performs testing of compatibility of testing pipeline.

        """

        # Get testing problem type.
        try:
            _ = self.params['testing']['problem']['type']
        except KeyError:
            print("Error: Couldn't retrieve the problem 'type' from the 'testing' section in the loaded configuration")
            exit(-5)

        # Get pipeline name.
        try:
            pipeline_name = self.params['pipeline']['name']
        except KeyError:
            print("Error: Couldn't retrieve the pipeline 'name' from the loaded configuration")
            exit(-6)
            
        # Prepare output paths for logging
        while True:
            # Dirty fix: if log_dir already exists, wait for 1 second and try again
            try:
                time_str = 'test_{0:%Y%m%d_%H%M%S}'.format(datetime.now())
                if self.app_state.args.savetag != '':
                    time_str = time_str + "_" + self.app_state.args.savetag
                self.log_dir = self.abs_path + '/' + time_str + '/'
                # Lowercase dir.
                self.log_dir = self.log_dir.lower()
                os.makedirs(self.log_dir, exist_ok=False)
            except FileExistsError:
                sleep(1)
            else:
                break

        # Set log dir and add the handler for the logfile to the logger.
        self.log_file = self.log_dir + 'tester.log'
        self.add_file_handler_to_logger(self.log_file)
        self.logger.info("Logger directory set to: {}".format(self.log_dir ))

        # Set random seeds in the testing section.
        self.set_random_seeds('testing', self.params['testing'])

        # Total number of detected errors.
        errors =0

        ################# TESTING PROBLEM ################# 

        # Build training problem manager.
        self.testing = ProblemManager('testing', self.params['testing']) 
        errors += self.testing.build()


        # check if the maximum number of episodes is specified, if not put a
        # default equal to the size of the dataset (divided by the batch size)
        # So that by default, we loop over the test set once.
        max_test_episodes = len(self.testing)

        self.params['testing']['problem'].add_default_params({'max_test_episodes': max_test_episodes})
        if self.params["testing"]["problem"]["max_test_episodes"] == -1:
            # Overwrite the config value!
            self.params['testing']['problem'].add_config_params({'max_test_episodes': max_test_episodes})

        # Warn if indicated number of episodes is larger than an epoch size:
        if self.params["testing"]["problem"]["max_test_episodes"] > max_test_episodes:
            self.logger.warning('Indicated maximum number of episodes is larger than one epoch, reducing it.')
            self.params['testing']['problem'].add_config_params({'max_test_episodes': max_test_episodes})

        self.logger.info("Setting the max number of episodes to: {}".format(
            self.params["testing"]["problem"]["max_test_episodes"]))

        ###################### PIPELINE ######################
        
        # Build the pipeline using the loaded configuration and global variables.
        self.pipeline = PipelineManager(pipeline_name, self.params['pipeline'])
        errors += self.pipeline.build()

        # Show pipeline.
        summary_str = self.pipeline.summarize_all_components_header()
        summary_str += self.testing.problem.summarize_io("testing")
        summary_str += self.pipeline.summarize_all_components()
        self.logger.info(summary_str)

        # Check errors.
        if errors > 0:
            self.logger.error('Found {} errors, terminating execution'.format(errors))
            exit(-7)

        # Handshake definitions.
        self.logger.info("Handshaking testing pipeline")
        defs_testing = self.testing.problem.output_data_definitions()
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
            elif "load" in self.params['pipeline']:
                pipeline_name = self.params['pipeline']['load']
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
            # Load individual models.

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
        self.export_experiment_configuration(self.log_dir, "testing_configuration.yaml",self.app_state.args.confirm)

    def initialize_statistics_collection(self):
        """
        Function initializes all statistics collectors and aggregators used by a given worker,
        creates output files etc.
        """
        # Create statistics collector for testing.
        self.testing_stat_col = StatisticsCollector()
        self.add_statistics(self.testing_stat_col)
        self.testing.problem.add_statistics(self.testing_stat_col)
        self.pipeline.add_statistics(self.testing_stat_col)
        # Create the csv file to store the testing statistics.
        self.testing_batch_stats_file = self.testing_stat_col.initialize_csv_file(self.log_dir, 'testing_statistics.csv')

        # Create statistics aggregator for testing.
        self.testing_stat_agg = StatisticsAggregator()
        self.add_aggregators(self.testing_stat_agg)
        self.testing.problem.add_aggregators(self.testing_stat_agg)
        self.pipeline.add_aggregators(self.testing_stat_agg)
        # Create the csv file to store the testing statistic aggregations.
        # Will contain a single row with aggregated statistics.
        self.testing_set_stats_file = self.testing_stat_agg.initialize_csv_file(self.log_dir, 'testing_set_agg_statistics.csv')

    def finalize_statistics_collection(self):
        """
        Finalizes statistics collection, closes all files etc.
        """
        # Close all files.
        self.testing_batch_stats_file.close()
        self.testing_set_stats_file.close()

    def run_experiment(self):
        """
        Main function of the ``Tester``: Test the loaded model over the test set.

        Iterates over the ``DataLoader`` for a maximum number of episodes equal to the test set size.

        The function does the following for each episode:

            - Forwards pass of the model,
            - Logs statistics & accumulates loss,
            - Activate visualization if set.


        """
        # Initialize tensorboard and statistics collection.
        self.initialize_statistics_collection()

        num_samples = len(self.testing)

        self.logger.info('Testing over the entire test set ({} samples in {} episodes)'.format(
            num_samples, len(self.testing.dataloader)))

        try:
            # Run test
            with torch.no_grad():

                episode = 0
                for test_dict in self.testing.dataloader:

                    # Terminal condition 0: max test episodes reached.
                    if episode == self.params["testing"]["problem"]["max_test_episodes"]:
                        break

                    # Forward pass.
                    self.pipeline.forward(test_dict)
                    # Collect the statistics.
                    self.collect_all_statistics(self.testing, self.pipeline, test_dict,
                            self.testing_stat_col, episode)

                    # Export to csv - at every step.
                    self.testing_stat_col.export_to_csv()

                    # Log to logger - at logging frequency.
                    if episode % self.app_state.args.logging_interval == 0:
                        self.logger.info(self.testing_stat_col.export_to_string('[Partial Test]'))

                    # move to next episode.
                    episode += 1

                # End for.

                self.logger.info('\n' + '='*80)
                self.logger.info('Test finished')

                # Aggregate statistics for the whole set.
                self.aggregate_all_statistics(self.testing, self.pipeline,
                    self.testing_stat_col, self.testing_stat_agg, episode)

                # Export aggregated statistics.
                self.export_all_statistics(self.testing_stat_agg, '[Full Test]')


        except SystemExit as e:
            # the training did not end properly
            self.logger.error('Experiment interrupted because {}'.format(e))
        except KeyboardInterrupt:
            # the training did not end properly
            self.logger.error('Experiment interrupted!')
        finally:
            # Finalize statistics collection.
            self.finalize_statistics_collection()
        

def main():
    """
    Entry point function for the ``Tester``.

    """
    tester = Tester()
    # parse args, load configuration and create all required objects.
    tester.setup_global_experiment()

    # finalize the experiment setup
    tester.setup_individual_experiment()

    # run the experiment
    tester.run_experiment()


if __name__ == '__main__':
    main()
