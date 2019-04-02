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


class RNN(Model): 
    """
    Simple Classifier consisting of fully connected layer with log softmax non-linearity.
    """
    def __init__(self, name, config):
        """
        Initializes the model.

        :param config: Dictionary of parameters (read from configuration ``.yaml`` file).
        :type config: ``ptp.configuration.ConfigInterface``
        """
        # Call constructors of parent classes.
        Model.__init__(self, name, RNN, config)

        # Get key mappings.
        self.key_inputs = self.stream_keys["inputs"]
        self.key_predictions = self.stream_keys["predictions"]

        # Retrieve input size from global variables.
        self.key_input_size = self.global_keys["input_size"]
        self.input_size = self.globals["input_size"]
        if type(self.input_size) == list:
            if len(self.input_size) == 1:
                self.input_size = self.input_size[0]
            else:
                raise ConfigurationError("RNN input size '{}' must be a single dimension (current {})".format(self.key_input_size, self.input_size))

        # Retrieve output (prediction) size from global params.
        self.prediction_size = self.globals["prediction_size"]
        if type(self.prediction_size) == list:
            if len(self.prediction_size) == 1:
                self.prediction_size = self.prediction_size[0]
            else:
                raise ConfigurationError("RNN prediction size '{}' must be a single dimension (current {})".format(self.key_prediction_size, self.prediction_size))
        
        # Get prediction mode from configuration.
        self.prediction_mode = self.config["prediction_mode"]
        if self.prediction_mode not in ['Dense','Last']:
            raise ConfigurationError("Invalid 'prediction_mode' (current {}, available {})".format(self.prediction_mode, ['Dense','Last']))

        # Retrieve hidden size from configuration.
        self.hidden_size = self.config["hidden_size"]
        if type(self.hidden_size) == list:
            if len(self.hidden_size) == 1:
                self.hidden_size = self.hidden_size[0]
            else:
                raise ConfigurationError("RNN hidden_size must be a single dimension (current {})".format(self.hidden_size))
        
        self.logger.info("Initializing RNN with input size = {}, hidden size = {} and prediction size = {}".format(self.input_size, self.hidden_size, self.prediction_size))

        # Get dropout value from config.
        dropout_rate = self.config["dropout_rate"]
        # Create dropout layer.
        self.dropout = torch.nn.Dropout(dropout_rate)

        # Get number of layers from config.
        self.num_layers = self.config["num_layers"]

        # Create RNN depending on the configuration
        self.rnn_type = self.config["rnn_type"]
        if self.rnn_type in ['LSTM', 'GRU']:
            # Create rnn cell.
            self.rnn_cell = getattr(torch.nn, self.rnn_type)(self.input_size, self.hidden_size, self.num_layers, dropout=dropout_rate, batch_first=True)
        else:
            try:
                # Retrieve the non-linearity.
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[self.rnn_type]
                # Create rnn cell.
                self.rnn_cell = torch.nn.RNN(self.input_size, self.hidden_size, self.num_layers, nonlinearity=nonlinearity, dropout=dropout_rate, batch_first=True)

            except KeyError:
                raise ConfigurationError( "Invalid RNN type, available options for 'rnn_type' are ['LSTM', 'GRU', 'RNN_TANH', 'RNN_RELU'] (currently '{}')".format(self.rnn_type))
        
        # Create the output layer.
        self.activation2output = torch.nn.Linear(self.hidden_size, self.prediction_size)
        
        # Check if initial state (h0/c0) are trainable or not.
        self.initial_state_trainable = self.config["initial_state_trainable"]

        # Parameters - for a single sample.        
        h0 = torch.zeros(self.num_layers, 1, self.hidden_size)
        c0 = torch.zeros(self.num_layers, 1, self.hidden_size)

        if self.initial_state_trainable:
            self.logger.info("Using trainable initial (h0/c0) state")
            # Initialize a single vector used as hidden state.
            # Initialize it using xavier initialization.
            torch.nn.init.xavier_uniform(h0)
            # It will be trainable, i.e. the system will learn what should be the right initialization state.
            self.init_hidden = torch.nn.Parameter(h0, requires_grad=True)
            # Initilize memory cell in a similar way.
            if self.rnn_type == 'LSTM':
                torch.nn.init.xavier_uniform(c0)
                self.init_memory = torch.nn.Parameter(c0, requires_grad=True)
        else:
            self.logger.info("Using zero initial (h0/c0) state")
            # We will still embedd it into parameter to enable storing/loading of both types of models by each other.
            self.init_hidden = torch.nn.Parameter(h0, requires_grad=False)
            if self.rnn_type == 'LSTM':
                self.init_memory = torch.nn.Parameter(c0, requires_grad=False)


    def initialize_hiddens_state(self, batch_size):

        if self.rnn_type == 'LSTM':
            # Return tuple (hidden_state, memory_cell).
            return (self.init_hidden.expand(self.num_layers, batch_size, self.hidden_size).contiguous(),
                self.init_memory.expand(self.num_layers, batch_size, self.hidden_size).contiguous() )

        else:
            # Return hidden_state.
            return self.init_hidden.expand(self.num_layers, batch_size, self.hidden_size).contiguous()


    def input_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of input data that are required by the component.

        :return: dictionary containing input data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {
            self.key_inputs: DataDefinition([-1, -1, self.input_size], [torch.Tensor], "Batch of inputs, each represented as index [BATCH_SIZE x SEQ_LEN x INPUT_SIZE]"),
            }


    def output_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of output data produced the component.

        :return: dictionary containing output data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
    
        if self.prediction_mode == "Dense":
            return {
                self.key_predictions: DataDefinition([-1, -1, self.prediction_size], [torch.Tensor], "Batch of predictions, each represented as probability distribution over classes [BATCH_SIZE x SEQ_LEN x PREDICTION_SIZE]")
                }
        else: # "Last"
            return {
                # Only last prediction.
                self.key_predictions: DataDefinition([-1, self.prediction_size], [torch.Tensor], "Batch of predictions, each represented as probability distribution over classes [BATCH_SIZE x SEQ_LEN x PREDICTION_SIZE]")
                }


    def forward(self, data_dict):
        """
        Forward pass of the model.

        :param data_dict: DataDict({'inputs', 'predictions ...}), where:

            - inputs: expected inputs [BATCH_SIZE x SEQ_LEN x INPUT_SIZE],
            - predictions: returned output with predictions (log_probs) [BATCH_SIZE x SEQ_LEN x PREDICTION_SIZE]
        """
        
        # Get inputs [BATCH_SIZE x SEQ_LEN x INPUT_SIZE]
        inputs = data_dict[self.key_inputs]
        batch_size = inputs.shape[0]

        # Initialize hidden state.
        hidden = self.initialize_hiddens_state(batch_size)

        # Propagate inputs through rnn cell.
        activations, hidden = self.rnn_cell(inputs, hidden)
        
        # Propagate activations through dropout layer.
        activations = self.dropout(activations)

        if self.prediction_mode == "Dense":
            # Pass every activation through the output layer.
            # Reshape to 2D tensor [BATCH_SIZE * SEQ_LEN x HIDDEN_SIZE]
            outputs = activations.contiguous().view(-1, self.hidden_size)

            # Propagate data through the output layer [BATCH_SIZE * SEQ_LEN x PREDICTION_SIZE]
            outputs = self.activation2output(outputs)

            # Reshape back to 3D tensor [BATCH_SIZE x SEQ_LEN x PREDICTION_SIZE]
            outputs = outputs.view(activations.size(0), activations.size(1), outputs.size(1))

            # Log softmax - along PREDICTION dim.
            predictions = F.log_softmax(outputs, dim=2)
        else:
            # Pass only the last activation through the output layer.
            outputs = activations.contiguous()[:, -1, :].squeeze()
            # Propagate data through the output layer [BATCH_SIZE x PREDICTION_SIZE]
            outputs = self.activation2output(outputs)
            # Log softmax - along PREDICTION dim.
            predictions = F.log_softmax(outputs, dim=1)

        # Add predictions to datadict.
        data_dict.extend({self.key_predictions: predictions})
