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
import torch.nn.functional as F

from ptp.configuration.configuration_error import ConfigurationError
from ptp.components.models.model import Model
from ptp.data_types.data_definition import DataDefinition


class SoftmaxClassifier(Model): 
    """
    Simple Classifier consisting of fully connected layer with log softmax non-linearity.
    """
    def __init__(self, name, params):
        """
        Initializes the classifier.

        :param params: Dictionary of parameters (read from configuration ``.yaml`` file).
        """
        # Call constructors of parent classes.
        #super(Model, self).__init__(name, params))
        Model.__init__(self, name, params)

        # Set key mappings.
        self.key_inputs = self.mapkey("inputs")
        self.key_predictions = self.mapkey("predictions")

        # Retrieve input and output (prediction) sizes from global params.
        self.key_input_size = self.mapkey("input_size")
        self.input_size = self.app_state[self.key_input_size]
        if type(self.input_size) == list:
            if len(self.input_size) == 1:
                self.input_size = self.input_size[0]
            else:
                raise ConfigurationError("SoftmaxClassifier input size '{}' must be a single dimension (current {})".format(self.key_input_size, self.input_size))


        self.key_prediction_size = self.mapkey("prediction_size")
        self.prediction_size = self.app_state[self.key_prediction_size]
        if type(self.prediction_size) == list:
            if len(self.prediction_size) == 1:
                self.prediction_size = self.prediction_size[0]
            else:
                raise ConfigurationError("SoftmaxClassifier prediction size '{}' must be a single dimension (current {})".format(self.key_prediction_size, self.prediction_size))
        
        print("INPUTS: ", self.input_size)
        print("PREDICTIONS: ", self.prediction_size)

        # Simple classifier.
        self.linear = torch.nn.Linear(self.input_size, self.prediction_size)
        

    def input_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of input data that are required by the component.

        :return: dictionary containing input data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {
            self.key_inputs: DataDefinition([-1, self.input_size], [torch.Tensor], "Batch of inputs, each represented as index [BATCH_SIZE x INPUT_SIZE]"),
            }


    def output_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of output data produced the component.

        :return: dictionary containing output data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {
            self.key_predictions: DataDefinition([-1, self.prediction_size], [torch.Tensor], "Batch of predictions, each represented as probability distribution over classes [BATCH_SIZE x PREDICTION_SIZE]")
            }

    def forward(self, data_dict):
        """
        Forward pass of the model.

        :param data_dict: DataDict({'inputs', 'predictions ...}), where:

            - inputs: expected inputs [BATCH_SIZE x INPUT_SIZE],
            - predictions: returned output with predictions (log_probs) [BATCH_SIZE x NUM_CLASSES]
        """
        # Add noise to weights
        #for _, param in self.linear.named_parameters():
        #    if param.requires_grad:
        #        #print (name, param.data)
        #        noise = torch.randn(param.data.shape)*0.3
        #        param.data = param.data * (1 + noise)
        #        #print (name, param.data)

        inputs = data_dict[self.key_inputs]
        predictions = F.log_softmax(self.linear(inputs), dim=1)
        # Add them to datadict.
        data_dict.extend({self.key_predictions: predictions})
