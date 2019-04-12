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

import torch
import torch.nn as nn

from ptp.components.losses.loss import Loss
from ptp.data_types.data_definition import DataDefinition


class NLLLoss(Loss):
    """
    Component calculating the negative log likelihood loss.
    """

    def __init__(self, name, config):
        """
        Initializes object, calls base constructor, initializes names of input and output ports.

        :param config: Dictionary of parameters (read from configuration ``.yaml`` file).

        """
        # Call constructors of parent classes.
        Loss.__init__(self, name, NLLLoss, config)

        # Get stream key mappnigs.
        self.key_masks = self.stream_keys["masks"]

        # Get masking flag.
        self.use_masking = self.config["use_masking"]

        # Get number of targets dimensions.
        self.num_targets_dims = self.config["num_targets_dims"]

        # Set loss.
        self.loss_function = nn.NLLLoss()


    def input_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of input data that are required by the component.

        :return: dictionary containing input data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        input_defs = {
            self.key_targets: DataDefinition([-1]*self.num_targets_dims, [torch.Tensor], "Batch of targets (indices) [DIM 1 x DIM 2 x ... ]"),
            self.key_predictions: DataDefinition([-1]*(self.num_targets_dims+1), [torch.Tensor], "Batch of predictions, represented as tensor with probability distribution over classes [DIM 1 x DIM x ... x NUM_CLASSES]")
            }
        if self.use_masking:
            input_defs[self.key_masks] = DataDefinition([-1], [torch.Tensor], "Batch of masks [BATCH_SIZE]")
        return input_defs

    def output_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of output data produced the component.

        :return: dictionary containing output data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {
            self.key_loss: DataDefinition([1], [torch.Tensor], "Loss value (single value for the whole batch - a scalar)")
            }


    def __call__(self, data_dict):
        """
        Calculates loss (negative log-likelihood) and adds it to data dict.

        :param data_dict: DataDict containing:
            - "targets": input batch of targets (class indices) [BATCH_SIZE x NUM_CLASSES]

            - "predictions": input batch of predictions (log_probs) being outputs of the model [BATCH_SIZE x 1]

            - "loss": output scalar representing the loss.

        """
        # Load inputs.
        targets = data_dict[self.key_targets]
        predictions = data_dict[self.key_predictions]

        #print("targets = ",targets)
        #print("predictions = ",predictions)

        #if isinstance(targets, (list,)):
        #    # Change to long tensor, as expected by nllloss.
        #    targets = torch.LongTensor(targets)

        # Mask predictions if option set.

        if self.use_masking:
            masks = data_dict[self.key_masks]
            targets = targets * masks.type(self.app_state.LongTensor)
            #print("unsqueezed masks = ", masks.unsqueeze(1))
            predictions = predictions * masks.unsqueeze(1).type(self.app_state.FloatTensor)

        #print("masked targets = ",targets)
        #print("masked predictions = ",predictions)
        
        # reshape.
        last_dim = predictions.size(-1)

        #print("\nTarget: {}\n Prediction: {}".format(targets.view(-1), predictions.view(-1, last_dim)))

        # Calculate loss.
        loss = self.loss_function(predictions.view(-1, last_dim), targets.view(-1))
        # Add it to datadict.
        data_dict.extend({self.key_loss: loss})
