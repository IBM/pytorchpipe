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

__author__ = "Vincent Marois, Tomasz Kornuta"

import torch
import numpy as np

from ptp.workers.trainer import Trainer
import ptp.configuration.config_parsing as config_parsing
from ptp.configuration.configuration_error import ConfigurationError


class OnlineTrainer(Trainer):
    """
    Implementation for the episode-based ``OnlineTrainer``.

    ..note ::

        The ``OfflineTrainer`` is based on epochs. While an epoch can be defined for all finite-size datasets, \
        it makes less sense for problems which have a very large, almost infinite, dataset (like algorithmic \
        tasks, which generate random data on-the-fly). \
         
        This is why this OnlineTrainer was implemented. Instead of looping on epochs, it iterates directly on \
        episodes (we call an iteration on a single batch an episode).


    """

    def __init__(self, name="OnlineTrainer"):
        """
        Only calls the ``Trainer`` constructor as the initialization phase is identical to the ``Trainer``.

       :param name: Name of the worker (DEFAULT: "OnlineTrainer").
       :type name: str

        """ 
        # Call base constructor to set up app state, registry and add default config.
        super(OnlineTrainer, self).__init__(name)

    def setup_experiment(self):
        """
        Sets up experiment for episode trainer:

            - Calls base class setup_experiment to parse the command line arguments,
            - Sets up the terminal conditions (loss threshold, episodes & epochs (optional) limits).

        """
        # Call base method to parse all command line arguments, load configuration, create problems and model etc.
        super(OnlineTrainer, self).setup_experiment()

        ################# TERMINAL CONDITIONS ################# 
        log_str = 'Terminal conditions:\n' + '='*80 + "\n"

        # Terminal condition I: loss. 
        self.config['training']['terminal_conditions'].add_default_params({'loss_stop': 1e-5})
        self.loss_stop = self.config['training']['terminal_conditions']['loss_stop']
        log_str += "  Setting Loss Stop threshold to {}\n".format(self.loss_stop)

        # In this trainer Partial Validation is mandatory, hence interval must be > 0.
        self.config['validation'].add_default_params({'partial_validation_interval': 100})
        self.partial_validation_interval = self.config['validation']['partial_validation_interval']
        if self.partial_validation_interval <= 0:
            self.logger.error("Episodic Trainer relies on Partial Validation, thus 'partial_validation_interval' must be a positive number!")
            exit(-4)
        else:
            log_str += "  Partial Validation activated with interval equal to {} episodes\n".format(self.partial_validation_interval)

        # Terminal condition II: max epochs. Optional.
        self.config["training"]["terminal_conditions"].add_default_params({'epoch_limit': -1})
        self.epoch_limit = self.config["training"]["terminal_conditions"]["epoch_limit"]
        if self.epoch_limit <= 0:
            log_str += "  Termination based on Epoch Limit is disabled\n"
            # Set to infinity.
            self.epoch_limit = np.Inf
        else:
            log_str += "  Setting the Epoch Limit to: {}\n".format(self.epoch_limit)

        # Calculate the epoch size in terms of episodes.
        self.epoch_size = len(self.training)
        log_str += "  Epoch size in terms of training episodes: {}\n".format(self.epoch_size)

        # Terminal condition III: max episodes. Mandatory.
        self.config["training"]["terminal_conditions"].add_default_params({'episode_limit': 100000})
        self.episode_limit = self.config['training']['terminal_conditions']['episode_limit']
        if self.episode_limit <= 0:
            self.logger.error("OnLine Trainer relies on episodes, thus 'episode_limit' must be a positive number!")
            exit(-5)
        else:
            log_str += "  Setting the Episode Limit to: {}\n".format(self.episode_limit)
        # Ok, finally print it.
        log_str += '='*80          
        self.logger.info(log_str)

        # Export and log configuration, optionally asking the user for confirmation.
        config_parsing.display_parsing_results(self.logger, self.app_state.args, self.unparsed)
        config_parsing.export_experiment_configuration_to_yml(self.logger, self.log_dir, "training_configuration.yaml", self.config, self.app_state.args.confirm)

    def run_experiment(self):
        """
        Main function of the ``OnlineTrainer``, runs the experiment.

        Iterates over the (cycled) DataLoader (one iteration = one episode).

        .. note::

            The test for terminal conditions (e.g. convergence) is done at the end of each episode. \
            The terminal conditions are as follows:

                - I. The loss is below the specified threshold (using the partial validation loss),
                - TODO: II. Early stopping is set and the full validation loss did not change by delta \
                    for the indicated number of epochs,
                - III. The maximum number of episodes has been met,
                - IV. The maximum number of epochs has been met (OPTIONAL).
            
            Additionally, experiment can be stopped by the user by pressing 'Stop experiment' \
            during visualization.


        The function does the following for each episode:

            - Handles curriculum learning if set,
            - Resets the gradients
            - Forwards pass of the model,
            - Logs statistics and exports to TensorBoard (if set),
            - Computes gradients and update weights
            - Activate visualization if set,
            - Validate the model on a batch according to the validation frequency.
            - Checks the above terminal conditions.


        """
        # Initialize TensorBoard and statistics collection.
        self.initialize_statistics_collection()
        self.initialize_tensorboard()

        # cycle the DataLoader -> infinite iterator
        self.training.dataloader = self.training.cycle(self.training.dataloader)

        try:
            '''
            Main training and validation loop.
            '''
            # Reset the counters.
            self.app_state.episode = 0
            self.app_state.epoch = 0
            self.logger.info('Starting next epoch: {}'.format(self.app_state.epoch))

            # Inform the training problem class that epoch has started.
            self.training.problem.initialize_epoch(self.app_state.epoch)

            # Set initial status.
            training_status = "Not Converged"
            for training_dict in self.training.dataloader:

                # reset all gradients
                self.optimizer.zero_grad()

                # Turn on training mode for the model.
                self.pipeline.train()

                # 1. Perform forward step.
                self.pipeline.forward(training_dict)

                # 2. Calculate statistics.
                self.collect_all_statistics(self.training, self.pipeline, training_dict, self.training_stat_col)

                # 3. Backward gradient flow.
                self.pipeline.backward(training_dict)

                # Check the presence of the 'gradient_clipping'  parameter.
                try:
                    # if present - clip gradients to a range (-gradient_clipping, gradient_clipping)
                    val = self.config['training']['gradient_clipping']
                    torch.nn.utils.clip_grad_value_(self.pipeline.parameters(), val)
                except KeyError:
                    # Else - do nothing.
                    pass

                # 4. Perform optimization.
                self.optimizer.step()

                # 5. Log collected statistics.
                # 5.1. Export to csv - at every step.
                self.training_stat_col.export_to_csv()

                # 5.2. Export data to TensorBoard - at logging frequency.
                if (self.training_batch_writer is not None) and \
                        (self.app_state.episode % self.app_state.args.logging_interval == 0):
                    self.training_stat_col.export_to_tensorboard()

                    # Export histograms.
                    if self.app_state.args.tensorboard >= 1:
                        for name, param in self.pipeline.named_parameters():
                            try:
                                self.training_batch_writer.add_histogram(name, 
                                    param.data.cpu().numpy(), self.app_state.episode, bins='doane')

                            except Exception as e:
                                self.logger.error("  {} :: data :: {}".format(name, e))

                    # Export gradients.
                    if self.app_state.args.tensorboard >= 2:
                        for name, param in self.pipeline.named_parameters():
                            try:
                                self.training_batch_writer.add_histogram(name + '/grad', 
                                    param.grad.data.cpu().numpy(), self.app_state.episode, bins='doane')

                            except Exception as e:
                                self.logger.error("  {} :: grad :: {}".format(name, e))

                # 5.3. Log to logger - at logging frequency.
                if self.app_state.episode % self.app_state.args.logging_interval == 0:
                    self.logger.info(self.training_stat_col.export_to_string())

                #  6. Validate and (optionally) save the model.
                if (self.app_state.episode % self.partial_validation_interval) == 0:

                    # Clear the validation batch from all items aside of the ones originally returned by the problem.
                    self.validation_dict.reinitialize(self.validation.problem.output_data_definitions())
                    # Perform validation.
                    self.validate_on_batch(self.validation_dict)
                    # Get loss.
                    validation_loss = self.pipeline.get_loss(self.validation_dict)

                    # Save the pipeline using the latest validation statistics.
                    self.pipeline.save(self.checkpoint_dir, training_status, validation_loss)

                    # Terminal conditions.
                    # I. the loss is < threshold (only when curriculum learning is finished if set.)
                    # We check that condition only in validation step!
                    if self.curric_done or not self.must_finish_curriculum:

                        # Check the Partial Validation loss.
                        if (validation_loss < self.loss_stop):
                            # Change the status...
                            training_status = "Converged (Partial Validation Loss went below " \
                                "Loss Stop threshold)"

                            # ... and THEN save the pipeline (update its statistics).
                            self.pipeline.save(self.checkpoint_dir, training_status, validation_loss)
                            break

                    # II. Early stopping is set and loss hasn't improved by delta in n epochs.
                    # early_stopping(index=epoch, avg_valid_loss). (TODO: coming in next release)
                    # training_status = 'Early Stopping.'

                # III. The episodes number limit has been reached.
                if self.app_state.episode+1 >= self.episode_limit:
                    # If we reach this condition, then it is possible that the model didn't converge correctly
                    # but it currently might get better since last validation.
                    training_status = "Not converged: Episode Limit reached"
                    break

                # Check if we are at the end of the 'epoch': indicate that the DataLoader is now cycling.
                if ((self.app_state.episode+1) % self.epoch_size) == 0:

                    # Epoch just ended!
                    # Inform the problem class that the epoch has ended.
                    self.training.problem.finalize_epoch(self.app_state.epoch)

                    # Aggregate training statistics for the epoch.
                    self.aggregate_all_statistics(self.training, self.pipeline, self.training_stat_col, self.training_stat_agg)
                    self.export_all_statistics( self.training_stat_agg,  '[Full Training]')

                    # Apply curriculum learning - change some of the Problem parameters
                    self.curric_done = self.training.problem.curriculum_learning_update_params(self.app_state.episode)

                    # IV. Epoch limit has been reached.
                    if self.app_state.epoch+1 >= self.epoch_limit:
                        training_status = "Not converged: Epoch Limit reached"
                        # "Finish" the training.
                        break

                    # Next epoch!
                    self.app_state.epoch += 1
                    self.logger.info('Starting next epoch: {}'.format(self.app_state.epoch))
                    # Inform the training problem class that epoch has started.
                    self.training.problem.initialize_epoch(self.app_state.epoch)
                    # Empty the statistics collector.
                    self.training_stat_col.empty()

                # Move on to next episode.
                self.app_state.episode += 1

            '''
            End of main training and validation loop. Perform final full validation.
            '''
            # Eventually perform "last" validation on batch.
            if self.validation_stat_col["episode"][-1] != self.app_state.episode:
                # We still must validate and try to save the model as it may perform better during this episode.

                # Clear the validation batch from all items aside of the ones originally returned by the problem.
                self.validation_dict.reinitialize(self.validation.problem.output_data_definitions())
                # Perform validation.
                self.validate_on_batch(self.validation_dict)

                # Try to save the model using the latest validation statistics.
                self.pipeline.save(self.checkpoint_dir, training_status, validation_loss)

            self.logger.info('\n' + '='*80)
            self.logger.info('Training finished because {}'.format(training_status))

            # Validate over the entire validation set.
            self.validate_on_set()

            # Do not save the model, as we tried it already on "last" validation batch.

            self.logger.info('Experiment finished!')

        except SystemExit as e:
            # the training did not end properly
            self.logger.error('Experiment interrupted because {}'.format(e))
        except ConfigurationError as e:
            # the training did not end properly
            self.logger.error('Experiment interrupted because {}'.format(e))
        except KeyboardInterrupt:
            # the training did not end properly
            self.logger.error('Experiment interrupted!')
        finally:
            # Finalize statistics collection.
            self.finalize_statistics_collection()
            self.finalize_tensorboard()
            self.logger.info("Experiment logged to: {}".format(self.log_dir))


def main():
    """
    Entry point function for the ``OnlineTrainer``.
    """
    trainer = OnlineTrainer()
    # parse args, load configuration and create all required objects.
    trainer.setup_experiment()
    # GO!
    trainer.run_experiment()

if __name__ == '__main__':
    main()