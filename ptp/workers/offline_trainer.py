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

__author__ = "Tomasz Kornuta, Vincent Marois"

import torch
import numpy as np

from ptp.workers.trainer import Trainer
import ptp.configuration.config_parsing as config_parsing
from ptp.configuration.configuration_error import ConfigurationError
from ptp.utils.termination_condition import TerminationCondition

class OfflineTrainer(Trainer):
    """
    Implementation for the epoch-based ``OfflineTrainer``.

    ..note::

        The default ``OfflineTrainer`` is based on epochs. \
        An epoch is defined as passing through all samples of a finite-size dataset.\
        The ``OfflineTrainer`` allows to loop over all samples from the training set many times i.e. in many epochs. \
        When an epochs finishes, it performs a similar step for the validation set and collects the statistics.

    """

    def __init__(self):
        """
        Constructor. It on calls the ``Trainer`` constructor as the initialization phase is identical to the one from ``Trainer``.
        """ 
        # Call base constructor to set up app state, registry and add default config.
        super(OfflineTrainer, self).__init__("OfflineTrainer", OfflineTrainer)

    def setup_experiment(self):
        """
        Sets up experiment for episode trainer:

            - Calls base class setup_experiment to parse the command line arguments,
            - Sets up the terminal conditions (loss threshold, episodes & epochs (optional) limits).

        """
        # Call base method to parse all command line arguments, load configuration, create tasks and model etc.
        super(OfflineTrainer, self).setup_experiment()

        # In this trainer Partial Validation is mandatory, hence interval must be > 0.
        self.partial_validation_interval = self.config['validation']['partial_validation_interval']
        if self.partial_validation_interval <= 0:
            self.logger.info("Partial Validation deactivated")
        else:
            self.logger.info("Partial Validation activated with interval equal to {} episodes".format(self.partial_validation_interval))

        ################# TERMINAL CONDITIONS ################# 
        log_str = 'Terminal conditions:\n' + '='*80 + "\n"

        # Terminal condition I: loss. 
        self.loss_stop_threshold = self.config_training['terminal_conditions']['loss_stop_threshold']
        log_str += "    I: Setting Loss Stop Threshold to {}\n".format(self.loss_stop_threshold)

        # Terminal condition II: early stopping. 
        self.early_stop_validations = self.config_training['terminal_conditions']['early_stop_validations']
        if self.early_stop_validations <= 0:
            log_str += "   II: Termination based on Early Stopping is disabled\n"
            # Set to infinity.
            self.early_stop_validations = np.Inf
        else:
            log_str += "   II: Setting the Number of Validations in Early Stopping to: {}\n".format(self.early_stop_validations)

        # Terminal condition III: max epochs. Mandatory.
        self.epoch_limit = self.config_training["terminal_conditions"]["epoch_limit"]
        if self.epoch_limit <= 0:
            self.logger.error("OffLine Trainer relies on epochs, thus Epoch Limit must be a positive number!")
            exit(-5)
        else:
            log_str += "  III: Setting the Epoch Limit to: {}\n".format(self.epoch_limit)

        # Log the epoch size in terms of episodes.
        self.epoch_size = self.training.get_epoch_size()
        log_str += "       Epoch size in terms of training episodes: {}\n".format(self.epoch_size)

        # Terminal condition IV: max episodes. Optional.
        self.episode_limit = self.config_training['terminal_conditions']['episode_limit']
        if self.episode_limit < 0:
            log_str += "   IV: Termination based on Episode Limit is disabled\n"
            # Set to infinity.
            self.episode_limit = np.Inf
        else:
            log_str += "   IV: Setting the Episode Limit to: {}\n".format(self.episode_limit)

        # Ok, finally print it.
        log_str += '='*80          
        self.logger.info(log_str)

        # Export and log configuration, optionally asking the user for confirmation.
        config_parsing.display_parsing_results(self.logger, self.app_state.args, self.unparsed)
        config_parsing.display_globals(self.logger, self.app_state.globalitems())
        config_parsing.export_experiment_configuration_to_yml(self.logger, self.app_state.log_dir, "training_configuration.yml", self.config, self.app_state.args.confirm)

    def run_experiment(self):
        """
        Main function of the ``OfflineTrainer``, runs the experiment.

        Iterates over the (cycled) DataLoader (one iteration = one episode).

        .. note::

            The test for terminal conditions (e.g. convergence) is done at the end of each episode. \
            The terminal conditions are as follows:

                - I. The loss is below the specified threshold (using the partial validation loss),
                - II. Early stopping is set and the full validation loss did went down \
                    for the indicated number of validation steps,
                - III. The maximum number of episodes has been met (OPTIONAL),
                - IV. The maximum number of epochs has been met.
            
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

        try:
            '''
            Main training and validation loop.
            '''
            # Reset the counters.
            self.app_state.episode = -1
            self.app_state.epoch = -1
            # Set initial status.
            training_status = "Not Converged"

            ################################################################################################
            # Beginning of external "epic loop".
            ################################################################################################
            while(True):
                self.app_state.epoch += 1
                self.logger.info('Starting next epoch: {}\n{}'.format(self.app_state.epoch, '='*80))

                # Inform the task managers that epoch has started.
                self.training.initialize_epoch()
                self.validation.initialize_epoch()

                # Apply curriculum learning - change Task parameters.
                self.curric_done = self.training.task.curriculum_learning_update_params(
                    0 if self.app_state.episode < 0 else self.app_state.episode,
                    self.app_state.epoch)
                    

                # Empty the statistics collector.
                self.training_stat_col.empty()
            
                ############################################################################################
                # Beginning of internal "episodic loop".
                ############################################################################################
                for training_batch in self.training.dataloader:
                    # Next episode.
                    self.app_state.episode += 1

                    # reset all gradients
                    self.optimizer.zero_grad()

                    # Turn on training mode for the model.
                    self.pipeline.train()

                    # 1. Perform forward step.
                    self.pipeline.forward(training_batch)

                    # 2. Calculate statistics.
                    self.collect_all_statistics(self.training, self.pipeline, training_batch, self.training_stat_col)

                    # 3. Backward gradient flow.
                    self.pipeline.backward(training_batch)

                    # Check the presence of the 'gradient_clipping'  parameter.
                    try:
                        # if present - clip gradients to a range (-gradient_clipping, gradient_clipping)
                        val = self.config_training['gradient_clipping']
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
                    if self.partial_validation_interval > 0 and (self.app_state.episode % self.partial_validation_interval) == 0:
                        # Clear the validation batch from all items aside of the ones originally returned by the task.
                        self.validation.batch.reinitialize(self.validation.task.output_data_definitions())
                        # Perform validation.
                        self.validate_on_batch(self.validation.batch)
                        # Do not save the model: OfflineTrainer uses the full set to determine whether to save or not.

                    # III. The episodes number limit has been reached.
                    if self.app_state.episode+1 >= self.episode_limit: # = np.Inf when inactive.
                        # If we reach this condition, then it is possible that the model didn't converge correctly
                        # but it currently might get better since last validation.
                        training_status = "Not converged: Episode Limit reached"
                        raise TerminationCondition(training_status)
                    
                ############################################################################################
                # End of internal "episodic loop".
                ############################################################################################

                # Epoch just ended!
                self.logger.info('End of epoch: {}\n{}'.format(self.app_state.epoch, '='*80))

                # Aggregate training statistics for the epoch.
                self.aggregate_all_statistics(self.training, self.pipeline, self.training_stat_col, self.training_stat_agg)
                self.export_all_statistics( self.training_stat_agg,  '[Full Training]')

                # Inform the training task manager that the epoch has ended.
                self.training.finalize_epoch()

                # Validate over the entire validation set.
                self.validate_on_set()

                # Get loss.
                validation_set_loss = self.pipeline.return_loss_on_set(self.validation_stat_agg)

                # Save the pipeline using the latest validation statistics.
                self.pipeline.save(self.checkpoint_dir, training_status, validation_set_loss)

                # Inform the validation task manager that the epoch has ended.
                self.validation.finalize_epoch()

                # Terminal conditions.
                # I. The loss is < threshold (only when curriculum learning is finished if set).
                # We check that condition only in validation step!
                if self.curric_done or not self.must_finish_curriculum:

                    # Check the Partial Validation loss.
                    if (validation_set_loss < self.loss_stop_threshold):
                        # Change the status.
                        training_status = "Converged (Full Validation Loss went below " \
                            "Loss Stop threshold of {})".format(self.loss_stop_threshold)

                        # Save the pipeline (update its statistics).
                        self.pipeline.save(self.checkpoint_dir, training_status, validation_set_loss)
                        # And leave both loops.
                        raise TerminationCondition(training_status)

                # II. Early stopping is set and loss hasn't improved by delta in n epochs.
                if self.pipeline.validation_loss_down_counter >= self.early_stop_validations:
                    training_status = "Not converged: reached limit of validations without improvement (Early Stopping)"
                    raise TerminationCondition(training_status)

                # IV. Epoch limit has been reached.
                if self.app_state.epoch+1 >= self.epoch_limit:
                    training_status = "Not converged: Epoch Limit reached"
                    # "Finish" the training.
                    raise TerminationCondition(training_status)

            ################################################################################################
            # End of external "epic loop".
            ################################################################################################

        except TerminationCondition as e:
            # End of main training and validation loop. Perform final full validation.

            self.logger.info('\n' + '='*80)
            self.logger.info('Training finished because {}'.format(training_status))

            # If episode limit was reached - perform last validation on the full set.
            if training_status == "Not converged: Episode Limit reached":
                # Validate over the entire validation set.
                self.validate_on_set()
                # Get loss.
                validation_set_loss = self.pipeline.return_loss_on_set(self.validation_stat_agg)
                # Save the pipeline using the latest validation statistics.
                self.pipeline.save(self.checkpoint_dir, training_status, validation_set_loss)


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
            self.logger.info("Experiment logged to: {}".format(self.app_state.log_dir))


def main():
    """
    Entry point function for the ``OfflineTrainer``.
    """
    # Create trainer.
    trainer = OfflineTrainer()
    # Parse args, load configuration and create all required objects.
    trainer.setup_experiment()
    # GO!
    trainer.run_experiment()

if __name__ == '__main__':
    main()