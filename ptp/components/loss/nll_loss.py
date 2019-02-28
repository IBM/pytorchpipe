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

from ptp.core_types.loss import Loss


class NLLLoss(Loss):
    """
    Component calculating the negative log likelihood loss.
    """

    def __init__(self, name, params):
        """
        Initializes object, calls base constructor, initializes names of input and output ports.

        :param params: Dictionary of parameters (read from configuration ``.yaml`` file).

        """
        # Call constructors of parent classes.
        Loss.__init__(self, name, params)

        # Set loss.
        self.loss_function = nn.NLLLoss()

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

        if isinstance(targets, (list,)):
            # Change to long tensor, as expected by nllloss.
            targets = torch.LongTensor(targets)

        # Calculate loss.
        loss = self.loss_function(predictions, targets)
        # Add it to datadict.
        data_dict.extend({self.key_loss: loss})
