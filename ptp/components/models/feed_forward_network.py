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

from ptp.configuration.configuration_error import ConfigurationError
from ptp.components.models.model import Model
from ptp.data_types.data_definition import DataDefinition


class FeedForwardNetwork(Model): 
    """
    Simple model consisting of several stacked fully connected layers with ReLU non-linearities and dropout between them.
    Additionally, applies log softmax non-linearity to the output.
    """
    def __init__(self, name, config):
        """
        Initializes the classifier.

        :param config: Dictionary of parameters (read from configuration ``.yaml`` file).
        :type config: ``ptp.configuration.ConfigInterface``
        """
        # Call constructors of parent classes.
        Model.__init__(self, name, FeedForwardNetwork, config)

        # Get key mappings.
        self.key_inputs = self.stream_keys["inputs"]
        self.key_predictions = self.stream_keys["predictions"]

        self.dimensions = self.config["dimensions"]

        # Retrieve input size from global variables.
        self.input_size = self.globals["input_size"]
        if type(self.input_size) == list:
            if len(self.input_size) == 1:
                self.input_size = self.input_size[0]
            else:
                raise ConfigurationError("Input size '{}' must be a single dimension (current {})".format(self.global_keys["input_size"], self.input_size))

        # Retrieve output (prediction) size from global params.
        self.prediction_size = self.globals["prediction_size"]
        if type(self.prediction_size) == list:
            if len(self.prediction_size) == 1:
                self.prediction_size = self.prediction_size[0]
            else:
                raise ConfigurationError("Prediction size '{}' must be a single dimension (current {})".format(self.global_keys["prediction_size"], self.prediction_size))
        
        self.logger.info("Initializing network with input size = {} and prediction size = {}".format(self.input_size, self.prediction_size))

        # Create the module list.
        modules = []
        # Retrieve dropout rate value - if set, will put dropout between every layer.
        dropout_rate = self.config["dropout_rate"]

        # Retrieve number of hidden layers, along with their sizes (numbers of hidden neurons from configuration).
        try:
            hidden_sizes = self.config["hidden_sizes"]
            if type(hidden_sizes) == list:
                # Stack linear layers.
                input_dim = self.input_size
                for hidden_dim in hidden_sizes:
                    # Add linear layer.
                    modules.append( torch.nn.Linear(input_dim, hidden_dim) )
                    # Add activation and dropout.
                    modules.append( torch.nn.ReLU() )
                    if (dropout_rate > 0):
                        modules.append( torch.nn.Dropout(dropout_rate) )
                    # Remember size.
                    input_dim = hidden_dim

                # Add output layer.
                modules.append( torch.nn.Linear(input_dim, self.prediction_size) )

                self.logger.info("Created {} hidden layers".format(len(hidden_sizes)))

            else:
                raise ConfigurationError("'hidden_sizes' must contain a list with numbers of neurons in hidden layers (currently {})".format(self.hidden_sizes))

        except KeyError:
            # Not present, in that case create a simple classifier with 1 linear layer.
            modules.append( torch.nn.Linear(self.input_size, self.prediction_size) )
        
        # Create the final non-linearity.
        self.use_logsoftmax = self.config["use_logsoftmax"]
        if self.use_logsoftmax:
            modules.append( torch.nn.LogSoftmax(dim=1) )

        # Finally create the sequential model out of those modules.
        self.layers = torch.nn.Sequential(*modules)



    def input_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of input data that are required by the component.

        :return: dictionary containing input data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {
            self.key_inputs: DataDefinition(([-1] * (self.dimensions -1)) + [self.input_size], [torch.Tensor], "Batch of inputs, each represented as index [BATCH_SIZE x ... x INPUT_SIZE]"),
            }


    def output_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of output data produced the component.

        :return: dictionary containing output data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {
            self.key_predictions: DataDefinition(([-1] * (self.dimensions -1)) + [self.prediction_size], [torch.Tensor], "Batch of predictions, each represented as probability distribution over classes [BATCH_SIZE x ... x PREDICTION_SIZE]")
            }

    def forward(self, data_dict):
        """
        Forward pass of the model.

        :param data_dict: DataDict({'inputs', 'predictions ...}), where:

            - inputs: expected inputs [BATCH_SIZE x ... x INPUT_SIZE],
            - predictions: returned output with predictions (log_probs) [BATCH_SIZE x ... x NUM_CLASSES]
        """
        # Get inputs.
        x = data_dict[self.key_inputs]
        
        #print("{}: input shape: {}, device: {}\n".format(self.name, x.shape, x.device))

        # Check that the input has the number of dimensions that we expect
        assert len(x.shape) == self.dimensions, \
            "Expected " + str(self.dimensions) + " dimensions for input, got " + str(len(x.shape))\
                 + " instead. Check number of dimensions in the config."

        # Reshape such that we do a broadcast over the last dimension
        origin_shape = x.shape
        x = x.contiguous().view(-1, origin_shape[-1])

        # Propagate inputs through all layers and activations.
        x = self.layers(x)

        # Restore the input dimensions but the last one (as it's been resized by the FFN)
        x = x.view(*origin_shape[0:self.dimensions-1], -1)

        # Add predictions to datadict.
        data_dict.extend({self.key_predictions: x})
