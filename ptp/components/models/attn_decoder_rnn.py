# Copyright (C) Alexis Asseman, IBM Corporation 2019
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


class Attn_Decoder_RNN(Model): 
    """
    Single layer GRU decoder with attention:
    Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473.
    
    Needs the full sequence of hidden states from the encoder as input, as well as the last hidden state from the encoder as input state.

    Code is based on https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html.
    """
    def __init__(self, name, config):
        """
        Initializes the model.

        :param config: Dictionary of parameters (read from configuration ``.yaml`` file).
        :type config: ``ptp.configuration.ConfigInterface``
        """
        # Call constructors of parent classes.
        Model.__init__(self, name, Attn_Decoder_RNN, config)

        # Get input/output mode
        self.output_last_state = self.config["output_last_state"]
        self.ffn_output = self.config["ffn_output"]

        # Get prediction mode from configuration.
        self.prediction_mode = self.config["prediction_mode"]
        if self.prediction_mode not in ['Dense','Last', 'None']:
            raise ConfigurationError("Invalid 'prediction_mode' (current {}, available {})".format(self.prediction_mode, ['Dense','Last', 'None']))

        self.autoregression_length = self.config["autoregression_length"]

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
        
        # Get dropout rate value from config.
        dropout_rate = self.config["dropout_rate"]

        # Create dropout layer.
        self.dropout = torch.nn.Dropout(dropout_rate)

        # Create rnn cell: hardcoded one layer GRU.
        self.rnn_cell = getattr(torch.nn, "GRU")(self.input_size, self.hidden_size, 1, dropout=dropout_rate, batch_first=True)

        # Create layers for the attention
        self.attn = torch.nn.Linear(self.hidden_size * 2, self.autoregression_length)
        self.attn_combine = torch.nn.Linear(self.hidden_size * 2, self.hidden_size)

        # Create the trainable initial input for the decoder (A trained <SOS> token of sorts)
        self.sos_token = torch.zeros(1, self.input_size)
        torch.nn.init.xavier_uniform(self.sos_token)
        self.sos_token = torch.nn.Parameter(self.sos_token, requires_grad=True)

        # Get key mappings.
        self.key_inputs = self.stream_keys["inputs"]
        self.key_predictions = self.stream_keys["predictions"]
        self.key_input_state = self.stream_keys["input_state"]
        if self.output_last_state:
            self.key_output_state = self.stream_keys["output_state"]
        
        self.logger.info("Initializing RNN with input size = {}, hidden size = {} and prediction size = {}".format(self.input_size, self.hidden_size, self.prediction_size))

        # Create the output layer.
        self.activation2output_layer = None
        if(self.ffn_output):
            self.activation2output_layer = torch.nn.Linear(self.hidden_size, self.prediction_size)
        
        # Create the final non-linearity.
        self.use_logsoftmax = self.config["use_logsoftmax"]
        if self.use_logsoftmax:
            if self.prediction_mode == "Dense":
                # Used then returning dense prediction, i.e. every output of unfolded model.
                self.log_softmax = torch.nn.LogSoftmax(dim=2)
            else:
                # Used when returning only the last output.
                self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def activation2output(self, activations):
        output = self.dropout(activations)

        if(self.ffn_output):
            #output = activations.squeeze(1)
            shape = activations.shape

            # Reshape to 2D tensor [BATCH_SIZE * SEQ_LEN x HIDDEN_SIZE]
            output = output.contiguous().view(-1, shape[2])

            # Propagate data through the output layer [BATCH_SIZE * SEQ_LEN x PREDICTION_SIZE]
            output = self.activation2output_layer(output)
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

        d[self.key_inputs] = DataDefinition([-1, -1, self.hidden_size], [torch.Tensor], "Batch of encoder outputs [BATCH_SIZE x SEQ_LEN x INPUT_SIZE]")

        # Input hidden state
        d[self.key_input_state] = DataDefinition([-1, 1, self.hidden_size], [torch.Tensor], "Batch of RNN last hidden states passed from another RNN that will be used as initial [BATCH_SIZE x NUM_LAYERS x SEQ_LEN x HIDDEN_SIZE]")

        return d

    def output_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of output data produced the component.

        :return: dictionary containing output data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        d = {}
    
        if self.prediction_mode == "Dense":
            d[self.key_predictions] = DataDefinition([-1, -1, self.prediction_size], [torch.Tensor], "Batch of predictions, each represented as probability distribution over classes [BATCH_SIZE x SEQ_LEN x PREDICTION_SIZE]")
        elif self.prediction_mode == "Last": # "Last"
            # Only last prediction.
            d[self.key_predictions] = DataDefinition([-1, self.prediction_size], [torch.Tensor], "Batch of predictions, each represented as probability distribution over classes [BATCH_SIZE x SEQ_LEN x PREDICTION_SIZE]")

        # Output hidden state stream TODO: why do we need that?
        if self.output_last_state:
            d[self.key_output_state] = DataDefinition([-1, 1, self.hidden_size], [torch.Tensor], "Batch of RNN final hidden states [BATCH_SIZE x NUM_LAYERS x SEQ_LEN x HIDDEN_SIZE]")
        
        return d

    def forward(self, data_dict):
        """
        Forward pass of the model.

        :param data_dict: DataDict({'inputs', 'predictions ...}), where:

            - inputs: expected inputs [BATCH_SIZE x SEQ_LEN x INPUT_SIZE],
            - predictions: returned output with predictions (log_probs) [BATCH_SIZE x SEQ_LEN x PREDICTION_SIZE]
        """
        
        inputs = data_dict[self.key_inputs]
        batch_size = inputs.shape[0]
        print("{}: input shape: {}, device: {}\n".format(self.name, inputs.shape, inputs.device))

        # Initialize hidden state from inputs - as last hidden state from external component.
        hidden = data_dict[self.key_input_state]
        # For RNNs (aside of LSTM): [BATCH_SIZE x NUM_LAYERS x HIDDEN_SIZE] -> [NUM_LAYERS x BATCH_SIZE x HIDDEN_SIZE]
        hidden = hidden.transpose(0,1)
        print("{}: hidden shape: {}, device: {}\n".format(self.name, hidden.shape, hidden.device))


        # List that will contain the output sequence
        activations = []

        # First input to the decoder - trainable "start of sequence" token
        activations_partial = self.sos_token.expand(batch_size, -1).unsqueeze(1)

        # Feed back the outputs iteratively
        for i in range(self.autoregression_length):

            # Do the attention thing
            attn_weights = torch.nn.functional.softmax(
                self.attn(torch.cat((activations_partial.transpose(0, 1), hidden), 2)),
                dim=2
            )
            attn_applied = torch.bmm(attn_weights.transpose(0, 1), inputs)
            activations_partial = torch.cat((activations_partial, attn_applied), 2)
            activations_partial = self.attn_combine(activations_partial)
            activations_partial = torch.nn.functional.relu(activations_partial)

            # Feed through the RNN
            activations_partial, hidden = self.rnn_cell(activations_partial, hidden)
            activations_partial = self.activation2output(activations_partial)

            # Add the single step output into list
            if self.prediction_mode == "Dense":
                activations += [activations_partial]

        # Reassemble all the outputs from list into an output tensor
        if self.prediction_mode == "Dense":
            outputs = torch.cat(activations, 1)
            # Log softmax - along PREDICTION dim.
            if self.use_logsoftmax:
                outputs = self.log_softmax(outputs)
            # Add predictions to datadict.
            data_dict.extend({self.key_predictions: outputs})
        elif self.prediction_mode == "Last":
            if self.use_logsoftmax:
                outputs = self.log_softmax(activations_partial.squeeze(1))
            # Add predictions to datadict.
            data_dict.extend({self.key_predictions: outputs})

        # Output last hidden state, if requested
        if self.output_last_state:
            # For others: [NUM_LAYERS x BATCH_SIZE x HIDDEN_SIZE] -> [BATCH_SIZE x NUM_LAYERS x HIDDEN_SIZE] 
            hidden = hidden.transpose(0,1)
            # Export last hidden state.
            data_dict.extend({self.key_output_state: hidden})
