# Copyright (C) aasseman, IBM Corporation 2019
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

__author__ = "Alexis Asseman"

import torch

from ptp.configuration.configuration_error import ConfigurationError
from ptp.components.models.model import Model
from ptp.data_types.data_definition import DataDefinition


class Seq2Seq_RNN(Model): 
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
        Model.__init__(self, name, Seq2Seq_RNN, config)

        # Get input/output mode
        self.input_mode = self.config["input_mode"]

        self.autoregression_length = self.config["autoregression_length"]
        
        # Check if initial state (h0/c0) is zero, trainable, or coming from input stream.
        self.initial_state = self.config["initial_state"]

        # Get number of layers from config.
        self.num_layers = self.config["num_layers"]

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
        
        # Retrieve hidden size from configuration.
        self.hidden_size = self.config["hidden_size"]
        if type(self.hidden_size) == list:
            if len(self.hidden_size) == 1:
                self.hidden_size = self.hidden_size[0]
            else:
                raise ConfigurationError("RNN hidden_size must be a single dimension (current {})".format(self.hidden_size))

        # Create RNN depending on the configuration
        self.cell_type = self.config["cell_type"]
        if self.cell_type in ['LSTM', 'GRU']:
            # Create rnn cell.
            self.rnn_cell_enc = getattr(torch.nn, self.cell_type)(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
            self.rnn_cell_dec = getattr(torch.nn, self.cell_type)(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        else:
            try:
                # Retrieve the non-linearity.
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[self.cell_type]
                # Create rnn cell.
                self.rnn_cell_enc = torch.nn.RNN(self.input_size, self.hidden_size, self.num_layers, nonlinearity=nonlinearity, batch_first=True)
                self.rnn_cell_dec = torch.nn.RNN(self.input_size, self.hidden_size, self.num_layers, nonlinearity=nonlinearity, batch_first=True)
            except KeyError:
                raise ConfigurationError( "Invalid RNN type, available options for 'cell_type' are ['LSTM', 'GRU', 'RNN_TANH', 'RNN_RELU'] (currently '{}')".format(self.cell_type))
        
        # Parameters - for a single sample.
        h0 = torch.zeros(self.num_layers, 1, self.hidden_size)
        c0 = torch.zeros(self.num_layers, 1, self.hidden_size)

        self.init_hidden = None

        if self.initial_state == "Trainable":
            self.logger.info("Using trainable initial (h0/c0) state")
            # Initialize a single vector used as hidden state.
            # Initialize it using xavier initialization.
            torch.nn.init.xavier_uniform(h0)
            # It will be trainable, i.e. the system will learn what should be the right initialization state.
            self.init_hidden = torch.nn.Parameter(h0, requires_grad=True)
            # Initilize memory cell in a similar way.
            if self.cell_type == 'LSTM':
                torch.nn.init.xavier_uniform(c0)
                self.init_memory = torch.nn.Parameter(c0, requires_grad=True)
        elif self.initial_state == "Zero":
            self.logger.info("Using zero initial (h0/c0) state")
            # We will still embedd it into parameter to enable storing/loading of both types of models by each other.
            self.init_hidden = torch.nn.Parameter(h0, requires_grad=False)
            if self.cell_type == 'LSTM':
                self.init_memory = torch.nn.Parameter(c0, requires_grad=False)

        # Get key mappings.
        self.key_inputs = self.stream_keys["inputs"]
        self.key_predictions = self.stream_keys["predictions"]
        
        self.logger.info("Initializing RNN with input size = {}, hidden size = {} and prediction size = {}".format(self.input_size, self.hidden_size, self.prediction_size))

        # Create the output layer.
        self.activation2output = torch.nn.Linear(self.hidden_size, self.prediction_size)
        
        # Create the final non-linearity.
        self.use_logsoftmax = self.config["use_logsoftmax"]
        if self.use_logsoftmax:
            # Used then returning dense prediction, i.e. every output of unfolded model.
            self.log_softmax = torch.nn.LogSoftmax(dim=2)

    def initialize_hiddens_state(self, batch_size):

        if self.cell_type == 'LSTM':
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
        d = {}

        d[self.key_inputs] = DataDefinition([-1, -1, self.input_size], [torch.Tensor], "Batch of inputs, each represented as index [BATCH_SIZE x SEQ_LEN x INPUT_SIZE]")

        return d

    def output_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of output data produced the component.

        :return: dictionary containing output data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        d = {}
    
        d[self.key_predictions] = DataDefinition([-1, -1, self.prediction_size], [torch.Tensor], "Batch of predictions, each represented as probability distribution over classes [BATCH_SIZE x SEQ_LEN x PREDICTION_SIZE]")

        return d

    def forward(self, data_dict):
        """
        Forward pass of the model.

        :param data_dict: DataDict({'inputs', 'predictions ...}), where:

            - inputs: expected inputs [BATCH_SIZE x SEQ_LEN x INPUT_SIZE],
            - predictions: returned output with predictions (log_probs) [BATCH_SIZE x SEQ_LEN x PREDICTION_SIZE]
        """
        
        # Get inputs [BATCH_SIZE x SEQ_LEN x INPUT_SIZE]
        inputs = data_dict[self.key_inputs]
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(1)
        batch_size = inputs.shape[0]


        # Initialize hidden state.
        hidden = self.initialize_hiddens_state(batch_size)


        # Encoder
        activations, hidden = self.rnn_cell_enc(inputs, hidden)
        activations_partial = self.activation2output(activations[:, -1, :])

        # Propagate inputs through rnn cell.
        activations_partial, hidden = self.rnn_cell_dec(activations_partial.unsqueeze(1), hidden)
        activations_partial = activations_partial.squeeze(1)
        activations_partial = self.activation2output(activations_partial)
        activations = [activations_partial]
        for i in range(self.autoregression_length - 1):
            activations_partial, hidden = self.rnn_cell_dec(activations_partial.unsqueeze(1), hidden)
            activations_partial = activations_partial.squeeze(1)
            activations_partial = self.activation2output(activations_partial)
            activations += [activations_partial]
        outputs = torch.stack(activations, 1)

        # Log softmax - along PREDICTION dim.
        if self.use_logsoftmax:
            outputs = self.log_softmax(outputs)

        # Add predictions to datadict.
        data_dict.extend({self.key_predictions: outputs})

