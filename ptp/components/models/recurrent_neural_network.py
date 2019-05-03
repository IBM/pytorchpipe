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


class RecurrentNeuralNetwork(Model): 
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
        Model.__init__(self, name, RecurrentNeuralNetwork, config)

        # Get input mode from the configuration.
        self.input_mode = self.config["input_mode"]
        if self.input_mode not in ['Dense','Autoregression_First', 'Autoregression_None']:
            raise ConfigurationError("Invalid 'input_mode' (current '{}', available {})".format(self.input_mode, ['Dense','Autoregression_First', 'Autoregression_None']))

        # Get prediction mode from configuration.
        self.prediction_mode = self.config["prediction_mode"]
        if self.prediction_mode not in ['Dense','Last', 'None']:
            raise ConfigurationError("Invalid 'prediction_mode' (current '{}', available {})".format(self.prediction_mode, ['Dense','Last', 'None']))

        # Get source of initial hidden state from configuration.
        self.initial_state = self.config["initial_state"]
        if self.initial_state not in ["Zero", "Trainable", "Input"]:
            raise ConfigurationError("Invalid 'initial_state' of the hidden state (current '{}', available {})".format(self.initial_state, ["Zero", "Trainable", "Input"]))

        # Make sure that the input-output combination is valid.
        if self.prediction_mode == 'None' and 'Autoregression' in self.input_mode:
            raise ConfigurationError("Invalid combination of 'input_mode' and prediction_mode' (current '{}' and '{}')".format(self.input_mode, self.prediction_mode))
        # TODO: Any others?

        # If we are returning any predictions, set up the right stream and variables.
        if self.prediction_mode != "None":
            self.key_predictions = self.stream_keys["predictions"]
            # Retrieve output (prediction) size from global params.
            self.prediction_size = self.globals["prediction_size"]
            # Check whether it is ok.
            if type(self.prediction_size) == list:
                if len(self.prediction_size) == 1:
                    self.prediction_size = self.prediction_size[0]
                else:
                    raise ConfigurationError("RNN prediction size '{}' must be a single dimension (current '{}')".format(self.key_prediction_size, self.prediction_size))

        # If we are accepting any inputs, set up the right stream and variables.
        if "None" not in self.input_mode:
            # Retrieve stream key.
            self.key_inputs = self.stream_keys["inputs"]
            # Retrieve input size from global variables.
            self.key_input_size = self.global_keys["input_size"]
            self.input_size = self.globals["input_size"]
            if type(self.input_size) == list:
                if len(self.input_size) == 1:
                    self.input_size = self.input_size[0]
                else:
                    raise ConfigurationError("RNN input size '{}' must be a single dimension (current {})".format(self.key_input_size, self.input_size))
        else: 
            # If there are no inputs, do we really need input_size.
            # Because it is autoregression mode and we can use prediction size instead.
            self.input_size = self.prediction_size

        # Setup options for autoregression.
        if "Autoregression" in self.input_mode:
            assert self.input_size == self.prediction_size, "In autoregression mode, needs input_size == prediction_size."
            # Get max length from configuration.
            self.max_autoregression_length = self.config["max_autoregression_length"]

        # Get number of layers from config.
        self.num_layers = self.config["num_layers"]

        # Retrieve hidden size from configuration.
        self.hidden_size = self.config["hidden_size"]
        if type(self.hidden_size) == list:
            if len(self.hidden_size) == 1:
                self.hidden_size = self.hidden_size[0]
            else:
                raise ConfigurationError("RNN hidden_size must be a single dimension (current {})".format(self.hidden_size))
        
        # Get dropout rate value from config.
        dropout_rate = self.config["dropout_rate"]

        # Create RNN depending on the configuration
        self.cell_type = self.config["cell_type"]
        if self.cell_type in ['LSTM', 'GRU']:
            # Create rnn cell.
            self.rnn_cell = getattr(torch.nn, self.cell_type)(self.input_size, self.hidden_size, self.num_layers, dropout=dropout_rate, batch_first=True)
        else:
            try:
                # Retrieve the non-linearity.
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[self.cell_type]
                # Create rnn cell.
                self.rnn_cell = torch.nn.RNN(self.input_size, self.hidden_size, self.num_layers, nonlinearity=nonlinearity, dropout=dropout_rate, batch_first=True)
            except KeyError:
                raise ConfigurationError( "Invalid RNN type, available options for 'cell_type' are ['LSTM', 'GRU', 'RNN_TANH', 'RNN_RELU'] (currently '{}')".format(self.cell_type))
        
        # Parameters - for a single sample.
        h0 = torch.zeros(self.num_layers, 1, self.hidden_size)
        c0 = torch.zeros(self.num_layers, 1, self.hidden_size)

        # Check if initial state (h0/c0) is zero, trainable, or coming from input stream.
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
        else: # "Input" means that it will be taken from the "input_state" stream.
            # Get adequate key mappings.
            self.key_input_state = self.stream_keys["input_state"]
            self.logger.info("Will read initial (h0/c0) state from stream '{}".format(self.key_input_state))


        # Setup for outputs.
        # Last state.
        self.output_last_state = self.config["output_last_state"]
        if self.output_last_state:
            self.key_output_state = self.stream_keys["output_state"]
        
        self.logger.info("Initializing RNN with input size = {}, hidden size = {} and prediction size = {}".format(self.input_size, self.hidden_size, self.prediction_size))

        # Setup for the output layer (and associated non-linearities).
        self.use_output_layer = self.config["use_output_layer"]
        if(self.use_output_layer):
            # Create dropout layer.
            self.dropout = torch.nn.Dropout(dropout_rate)
            # Create the layer.
            self.activation2output = torch.nn.Linear(self.hidden_size, self.prediction_size)
        
        # Setup for the final non-linearity.
        self.use_logsoftmax = self.config["use_logsoftmax"]
        if self.use_logsoftmax:
            if self.prediction_mode == "Dense":
                # Used then returning dense prediction, i.e. every output of unfolded model.
                self.log_softmax = torch.nn.LogSoftmax(dim=2)
            else:
                # Used when returning only the last output.
                self.log_softmax = torch.nn.LogSoftmax(dim=1)


    def initialize_hiddens_state(self, batch_size):
        """
        Function initializes hidden states, depending on the cell type.
        """
        if self.cell_type == 'LSTM':
            # Return tuple (hidden_state, memory_cell).
            return (self.init_hidden.expand(self.num_layers, batch_size, self.hidden_size).contiguous(),
                self.init_memory.expand(self.num_layers, batch_size, self.hidden_size).contiguous() )
        else:
            # Return hidden_state.
            return self.init_hidden.expand(self.num_layers, batch_size, self.hidden_size).contiguous()


    def activation_to_output_pass(self, activations):
        """
        Function propagates hidden state "activations" through output layer (that pass can be optionally turned off).
        """
        if(self.use_output_layer):
            # Use dropout when using output layer.
            output = self.dropout(activations)

            #output = activations.squeeze(1)
            shape = activations.shape

            # Reshape to 2D tensor [BATCH_SIZE * SEQ_LEN x HIDDEN_SIZE]
            output = output.contiguous().view(-1, shape[2])

            # Propagate data through the output layer [BATCH_SIZE * SEQ_LEN x PREDICTION_SIZE]
            output = self.activation2output(output)
            #output = output.unsqueeze(1)

            # Reshape back to 3D tensor [BATCH_SIZE x SEQ_LEN x PREDICTION_SIZE]
            output = output.view(shape[0], shape[1], output.size(1))

        return output


    def input_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of input data that are required by the component.

        :return: dictionary containing input data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        d = {}
        # Input depending on the input_mode
        if self.input_mode == "Dense":
            d[self.key_inputs] = DataDefinition([-1, -1, self.input_size], [torch.Tensor], "Batch of inputs, each being a sequence of items [BATCH_SIZE x SEQ_LEN x INPUT_SIZE]")
        elif self.input_mode == "Autoregression_First":
            d[self.key_inputs] = DataDefinition([-1, self.input_size], [torch.Tensor], "Batch of inputs, each being a single item [BATCH_SIZE x SEQ_LEN x INPUT_SIZE]")
        #else: Autoregression_None: no inputs.

        # Input hidden state
        if self.initial_state == "Input":
            if self.cell_type == "LSTM":
                d[self.key_input_state] = DataDefinition([2, self.num_layers, -1, self.hidden_size], [torch.Tensor], "Batch of LSTM initial hidden states (h0/c0) passed from another LSTM [2 x NUM_LAYERS x SEQ_LEN x HIDDEN_SIZE]")
            else:
                d[self.key_input_state] = DataDefinition([self.num_layers, -1, self.hidden_size], [torch.Tensor], "Batch of RNN initial hidden states passed from another RNN [NUM_LAYERS x SEQ_LEN x HIDDEN_SIZE]")

        return d

    def output_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of output data produced the component.

        :return: dictionary containing output data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        d = {}
    
        # Output: predictions stream.
        if self.prediction_mode == "Dense":
            d[self.key_predictions] = DataDefinition([-1, -1, self.prediction_size], [torch.Tensor], "Batch of predictions, each represented as sequence of probability distributions over classes [BATCH_SIZE x SEQ_LEN x PREDICTION_SIZE]")
        elif self.prediction_mode == "Last": # "Last"
            # Only last prediction.
            d[self.key_predictions] = DataDefinition([-1, self.prediction_size], [torch.Tensor], "Batch of predictions, each represented as a single probability distribution over classes [BATCH_SIZE x PREDICTION_SIZE]")
        # Else: no predictions.

        # Output: hidden state stream.
        if self.output_last_state:
            if self.cell_type == "LSTM":
                d[self.key_output_state] = DataDefinition([2, self.num_layers, -1, self.hidden_size], [torch.Tensor], "Batch of LSTM final hidden states (h0/c0) [2 x NUM_LAYERS x SEQ_LEN x HIDDEN_SIZE]")
            else:
                d[self.key_output_state] = DataDefinition([self.num_layers, -1, self.hidden_size], [torch.Tensor], "Batch of RNN final hidden states [NUM_LAYERS x SEQ_LEN x HIDDEN_SIZE]")

        return d

    def forward(self, data_dict):
        """
        Forward pass of the model.

        :param data_dict: DataDict({'inputs', 'predictions ...}), where:

            - inputs: expected inputs [BATCH_SIZE x SEQ_LEN x INPUT_SIZE],
            - predictions: returned output with predictions (log_probs) [BATCH_SIZE x SEQ_LEN x PREDICTION_SIZE]
        """
        
        inputs = None
        batch_size = None

        # Get inputs
        if "None" in self.input_mode:
            # Must be in autoregressive mode - retrieve batch_size from initial hidden state from encoder.
            batch_size = data_dict[self.key_input_state][0].shape[1]
            # Set zero inputs [BATCH_SIZE x SEQ_LEN x INPUT_SIZE].
            inputs = torch.zeros(batch_size, self.hidden_size, requires_grad=False).type(self.app_state.FloatTensor)

        else:
            # Get inputs [BATCH_SIZE x SEQ_LEN x INPUT_SIZE]
            inputs = data_dict[self.key_inputs]
            if inputs.dim() == 2:
                inputs = inputs.unsqueeze(1)
            batch_size = inputs.shape[0]
        

        # Get initial state, depending on the settings.
        if self.initial_state == "Input":
            # Initialize hidden state.
            hidden = data_dict[self.key_input_state]
        else:
            hidden = self.initialize_hiddens_state(batch_size)

        activations = []

        # Check out operation mode.
        if "Autoregression" in self.input_mode: 
            # Autoregressive mode - feed back outputs in the input
            activations_partial, hidden = self.rnn_cell(inputs, hidden)
            activations_partial = self.activation_to_output_pass(activations_partial)
            activations += [activations_partial]

            # Feed back the outputs iteratively
            for i in range(self.max_autoregression_length - 1):
                activations_partial, hidden = self.rnn_cell(activations_partial, hidden)
                activations_partial = self.activation_to_output_pass(activations_partial)
                # Add the single step output into list
                if self.prediction_mode == "Dense":
                    activations += [activations_partial]
            # Reassemble all the outputs from list into an output sequence
            if self.prediction_mode == "Dense":
                outputs = torch.cat(activations, 1)
                # Log softmax - along PREDICTION dim.
                if self.use_logsoftmax:
                    outputs = self.log_softmax(outputs)
                # Add predictions to datadict.
                data_dict.extend({self.key_predictions: outputs})
            elif self.prediction_mode == "Last":
                # Take only the last activations.
                outputs = activations_partial.squeeze(1)
                if self.use_logsoftmax:
                    outputs = self.log_softmax(outputs)
                # Add predictions to datadict.
                data_dict.extend({self.key_predictions: outputs})

        else:
            # Normal mode - feed the entire input sequence at once
            activations, hidden = self.rnn_cell(inputs, hidden)

            if self.prediction_mode == "Dense":
                # Pass every activation through the output layer.
                outputs = self.activation_to_output_pass(activations)
                
                # Log softmax - along PREDICTION dim.
                if self.use_logsoftmax:
                    outputs = self.log_softmax(outputs)

                # Add predictions to datadict.
                data_dict.extend({self.key_predictions: outputs})
            elif self.prediction_mode == "Last":
                outputs = self.activation_to_output_pass(activations.contiguous()[:, -1, :].unsqueeze(1))
                outputs = outputs.squeeze(1)

                # Log softmax - along PREDICTION dim.
                if self.use_logsoftmax:
                    outputs = self.log_softmax(outputs)
                    
                # Add predictions to datadict.
                data_dict.extend({self.key_predictions: outputs})
            elif self.prediction_mode == "None":
                # Nothing, since we don't want to keep the RNN's outputs
                pass

        if self.output_last_state:
            data_dict.extend({self.key_output_state: hidden})
