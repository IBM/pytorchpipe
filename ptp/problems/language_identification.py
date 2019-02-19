#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) IBM Corporation 2019
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


# Thoma, Martin. "The WiLI benchmark dataset for written language identification." arXiv preprint arXiv:1801.07779 (2018).
# https://arxiv.org/abs/1801.07779
# https://zenodo.org/record/841984/files/wili-2018.zip?download=1
# Author: Robert Guthrie

import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ptp.problems import Problem
from ptp.utils.data_dict import DataDict

#torch.manual_seed(1)
class Component(object):
    def __init__(self, name, params):
        """
        Initializes the component. Saves names and params.

        :param name: Name of the component.

        :param params: Dictionary of parameters (read from configuration ``.yaml`` file).
        """
        self.name = name
        self.params = params
        # Set default (empty) data definitions and default_values.
        self.data_definitions = {}
        self.default_values =  {}
        # Initialize the "name mapping facility".
        params.add_default_params({"mappings": {}})
        self.mappings = params["mappings"]

    def get_name(self, stream_name):
        """
        Method responsible for checking whether name exists in the mappings.
        
        :raturn: Mapped name or stream name (if it does not exist in mappings list).
        """
        return self.mappings.get(stream_name, stream_name)

    def create_data_dict(self, data_definitions = None):
        """
        Returns a :py:class:`miprometheus.utils.DataDict` object with keys created on the \
        problem data_definitions and empty values (None).

        :param data_definitions: Data definitions that will be used (DEFAULT: None, meaninng that self.data_definitions will be used)

        :return: new :py:class:`miprometheus.utils.DataDict` object.
        """
        # Use self.data_definitions as default.
        data_definitions = data_definitions if data_definitions is not None else self.data_definitions

        return DataDict({key: None for key in data_definitions.keys()})

    def extend_data_dict(self, data_dict, data_definitions): #= None):
        """
        Copies and optionally extends a :py:class:`miprometheus.utils.DataDict` object by adding keys created on the \
        problem data_definitions and empty values (None).

        .. warning::
            This is in-place operation, i.e. extends existing object, does not return a new one.

        :param data_dict: :py:class:`miprometheus.utils.DataDict` object.

        :param data_definitions: Data definitions that will be used (DEFAULT: None, meaninng that self.data_definitions will be used)

        """
        # Use self.data_definitions as default.
        #data_definitions = data_definitions if data_definitions is not None else self.data_definitions
        for key in data_definitions.keys():
            assert key not in data_dict.keys(), "Cannot extend DataDict, as {} already present in its keys!".format(key)
            data_dict[key] = None

    def extend_default_values(self, default_values):
        """
        Extends the input list of default values by component's own list.

        .. warning::
            This is in-place operation, i.e. extends existing object, does not return a new one.

        .. warning::
            Value will be overwritten for the existing keys!

        :param default_values: Dictionary containing default values (that will be extended).
        """
        for key, value in self.default_values.items():
            default_values[key] = value


class SoftmaxClassifier(nn.Module, Component): 
    """
    Simple Classifier consisting of fully connected layer with log softmax non-linearity.
    """
    def __init__(self, params, input_default_values):
        """
        Initializes the classifier.

        :param params: Dictionary of parameters (read from configuration ``.yaml`` file).

        :param input_default_values: Dictionary containing default_input_values.
        """
        # Call constructors of parent classes.
        Component.__init__(self, "SoftmaxClassifier", params)
        nn.Module.__init__(self)

        # Name mappings.
        self.name_inputs = self.get_name("inputs")
        self.name_predictions = self.get_name("predictions")

        # Retrieve input (vocabulary) size and number of classes from default params.
        self.input_size = input_default_values['encoded_input_size']
        self.num_classes = input_default_values['num_classes']

        # Simple classifier.
        self.linear = nn.Linear(self.input_size, self.num_classes)
        
        # Set default data_definitions dict.
        # Encoded with BoW its is [BATCH_SIZE x NUM_CLASSES] !
        self.data_definitions = {self.name_predictions: {'size': [-1, self.num_classes], 'type': [torch.Tensor]} }


    def forward(self, data_dict):
        """
        Forward pass of the  model.

        :param data_dict: DataDict({'inputs', 'predictions ...}), where:

            - inputs: expected inputs [BATCH_SIZE x INPUT_SIZE],
            - predictions: returned output with predictions (log_probs) [BATCH_SIZE x NUM_CLASSES]
        """
        inputs = data_dict[self.name_inputs]
        predictions = F.log_softmax(self.linear(inputs), dim=1)
        # Add them to datadict.
        self.extend_data_dict(data_dict, {self.name_predictions: None})
        data_dict[self.name_predictions] = predictions
        # For compatibility. TODO: remove return!
        return data_dict

class TokenEncoder(object):
    """
    Class representing 1hot (index) word encoder.
    """
    def __init__(self, params):
        # Set default parameters.
        params.add_default_params({
            'data_folder': '~/data/language_identification',
            'source_files': '', # Source files
            'encodings_file': 'encodings.csv', # File containing encodings
            'decode': False, # Mode - False: word to index, True: index to word
            'regenerate': False # True means that it will be regenerated despite the existence of file.
            })
        # Read the actual configuration.
        self.data_folder = params['data_folder']
        self.source_files = params['source_files']
        self.encodings_file = params['encodings_file']
        self.mode_decode = params['decode']
        self.mode_regenerate = params['regenerate']

        # Encodings file.
        encodings_file = os.path.expanduser(self.data_folder) + "/" + self.encodings_file

        # Try to load dictionary from a file.
        if self.mode_regenerate or not os.path.exists(encodings_file):
            # Generate new encodings.
            self.word_to_ix = self.create_encodings(self.data_folder, self.source_files)
            assert (len(self.word_to_ix) > 0), "The encodings list cannot be empty!"
            # Ok, save necodings, so next time we will simply load them.
            self.save_encodings(self.encodings_file, self.word_to_ix)
        else:
            # Load encodings.
            self.word_to_ix = self.load_encodings(self.encodings_file)
        
        # If dictionary is supposed to map indices to keys.
        if self.mode_decode:
            self.ix_to_word = dict((v,k) for k,v in self.word_to_ix.items())
        
        # Ok, we are ready to go!


    def create_encodings(self, data_folder, source_files):
        """
        Load list of files (containing raw text) and creates a dictionary from all words (tokens).
        Indexing starts from 0.

        :return: Dictionary with mapping "word-to-index".
        """
        assert len(source_files) > 0, 'Cannot create dictionary: "source_files" is empty, please provide comma separated list of files to be processed'
        # Get absolute path.
        data_folder = os.path.expanduser(data_folder)

        # Dictionary word_to_ix maps each word in the vocab to a unique integer.
        word_to_ix = {}

        for filename in source_files(','):
            content = eval(open(data_folder+ '/' + filename).read())
            # Parse tokens.
            for word in content.split():
                # If new token.
                if word not in self.word_to_ix:
                    word_to_ix[word] = len(word_to_ix)
        return word_to_ix


    def load_encodings(self, encodings_file):
        """
        Loads encodings from csv file.

        .. warning::
             There is an assumption that file will contain key:value pairs (no content checking for now!)

        :param encodings_file: File with encodings (absolute path + filename).
        :return: dictionary with word:index keys
        """        
        with open(encodings_file, mode='rb') as file:
            # Check the presence of the header.
            has_header = csv.Sniffer().has_header(file.read(1024))
            file.seek(0)  # Rewind.
            reader = csv.reader(file)
            # Skip the header row.
            if has_header:
                next(reader)  
            # Read the remaining rows.
            encodings_dict = {rows[0]:rows[1] for rows in reader}
        return encodings_dict


    def save_encodings(self, encodings_file, word_to_ix):
        """
        Saves encodings (dictionary) to a file.

        :param encodings_file: File with encodings (absolute path + filename).
        :param word_to_ix: dictionary with word:index keys
        """
        with open(encodings_file, mode='w') as outfile:
            # Create header.
            fieldnames = ['word', 'index']
            writer = csv.writer(outfile, fieldnames=fieldnames)
            writer.writeheader()
            # Write word-index pairs.
            for (k,v) in word_to_ix:
                writer.writerow({k : v})


class LanguageIdentification(Problem, Component):
    """
    Language identification (classification) problem.
    """

    def __init__(self, params):
        """
        Initializes problem object. Calls base constructor.

        :param params: Dictionary of parameters (read from configuration ``.yaml`` file).

        """
        # Call constructors of parent classes.
        Component.__init__(self, "LanguageIdentification", params)
        Problem.__init__(self, params, "LanguageIdentification")

        # Name mappings.
        self.name_inputs = self.get_name("inputs")
        self.name_targets = self.get_name("targets")
        self.name_encoded_targets = self.get_name("encoded_targets")
        self.name_encoded_predictions = self.get_name("encoded_predictions")

        # Set default parameters.
        self.params.add_default_params({'data_folder': '~/data/language_identification',
                                        'use_train_data': True
                                        })
        # Get absolute path.
        #data_folder = os.path.expanduser(self.params['data_folder'])
        # Retrieve parameters from the dictionary.
        self.use_train_data = self.params['use_train_data']

        # Set default data_definitions dict.
        self.data_definitions = {self.name_inputs: {'size': [-1, 1], 'type': [list, str]}, 
                                # [BATCH x SENTENCE (list of words as a single string)]
                                self.name_targets: {'size': [-1, 1], 'type': [list, str]}
                                # [BATCH x WORD (word as a single string)]
                                }

        self.data = [("me gusta comer en la cafeteria", "SPANISH"),
                ("Give it to me", "ENGLISH"),
                ("No creo que sea una buena idea", "SPANISH"),
                ("No it is not a good idea to get lost at sea", "ENGLISH")]

        self.test_data = [("Yo creo que si", "SPANISH"),
                    ("it is lost on me", "ENGLISH")]

        # Set length.
        self.length = len(self.data)

        # Set loss.
        self.loss_function = nn.NLLLoss()

    def generate_dummy_data(self, mode):
        """
        Method generates dummy files (sentence-language) pairs (in two files).
        Data taken from the _example.

        .. _example: https://pytorch.org/tutorials/beginner/nlp/deep_learning_tutorial.html
        """
        # Dummy data.
        pass
        
        

    def __getitem__(self, index):
        """
        Getter method to access the dataset and return a sample.

        :param index: index of the sample to return.
        :type index: int

        :return: ``DataDict({'inputs','targets'})``

        """
        # Get sentence and language.
        (sentence, language) = self.data[index]

        # Return data_dict.
        data_dict = self.create_data_dict()
        data_dict[self.name_inputs] = sentence
        data_dict[self.name_targets] = language
        return data_dict

    def evaluate_loss(self, data_dict):
        """ Calculates accuracy equal to mean number of correct predictions in a given batch.

        :param data_dict: DataDict containing:
            - "encoded_targets": batch of targets (class indices) [BATCH_SIZE x NUM_CLASSES]

            - "encoded_predictions": batch of predictions (log_probs) being outputs of the model [BATCH_SIZE x 1]

        :return: loss calculated useing the loss function (negative log-likelihood).
        """
        targets = data_dict[self.name_encoded_targets]
        predictions = data_dict[self.name_encoded_predictions]
        loss = self.loss_function(predictions, targets.squeeze(dim=1))
        return loss


class Encoder(Component):
    """
    Default encoder class. Creates interface and provides generic methods for batch processing.
    """
    def __init__(self, name, params, default_input_values):
        """
        Initializes encoder object.

        :param name: Name of the encoder.

        :param params: Dictionary of parameters (read from configuration ``.yaml`` file).

        :param default_input_values: Dictionary containing default_input_values.
        """
        # Call constructor of parent class.
        Component.__init__(self, name, params)

        # Default name mappings for all encoders.
        # decoded inputs -> encoded inputs
        self.name_decoded_inputs = self.get_name("decoded_inputs")
        self.name_encoded_inputs = self.get_name("encoded_inputs")
        # encoded outputs <- encoded outputs
        self.name_encoded_outputs = self.get_name("encoded_outputs")
        self.name_decoded_outputs = self.get_name("decoded_outputs")


    def encode_sample(self, sample):
        """
        Method responsible for encoding of a single sample (interface).
        """
        pass

    def encode_batch(self, data_dict):
        """
        Method responsible for encoding of a single sample (interface).

        :param data_dict: :py:class:`miprometheus.utils.DataDict` object containing data to encode and that will be extended with encoded results.
        """
        pass

    def decode_sample(self, encoded_sample):
        """
        Method responsible for decoding of a single encoded sample (interface).
        """
        pass    

    def decode_batch(self, data_dict):
        """
        Method responsible for decoding of a single encoded sample (interface).

        :param data_dict: :py:class:`miprometheus.utils.DataDict` object containing data to decode and that will be extended with decoded results.
        """
        pass    


class BOWSentenceEncoder(Encoder):
    """
    Simple Bag-of-word type encoder that encodes the sentence into a vector.
    
    .. warning::
        BoW transformation is inreversible, thus decode-related methods in fact return original inputs.
    """
    def  __init__(self, params, default_input_values):
        """t
        Initializes the bag-of-word encoded by creating dictionary mapping ALL words from training, validation and test sets into unique indices.

        :param params: Dictionary of parameters (read from configuration ``.yaml`` file).

        :param default_input_values: Dictionary containing default_input_values.
        """
        # Call base class constructor.
        super(BOWSentenceEncoder, self).__init__('BOWSentenceEncoder', params, default_input_values)

        # Dummy data.
        self.data = [("me gusta comer en la cafeteria", "SPANISH"),
                ("Give it to me", "ENGLISH"),
                ("No creo que sea una buena idea", "SPANISH"),
                ("No it is not a good idea to get lost at sea", "ENGLISH")]

        self.test_data = [("Yo creo que si", "SPANISH"),
                    ("it is lost on me", "ENGLISH")]

        # Dictionary word_to_ix maps each word in the vocab to a unique integer.
        # It will later used as word index during encoding into the Bag of words vector.
        self.word_to_ix = {}
        for sent, _ in self.data + self.test_data:
            for word in sent.split():
                if word not in self.word_to_ix:
                    self.word_to_ix[word] = len(self.word_to_ix)
        #print(word_to_ix)

        # Size of a single encoded item.
        self.item_size = len(self.word_to_ix)

        # Define the default_values dict: holds parameters values that a model may need.
        self.default_values = {'encoded_input_size': self.item_size}
        # Set default data_definitions dict.
        # Encoded with BoW its is [BATCH_SIZE x VOCAB_SIZE] !
        self.data_definitions = {self.name_encoded_inputs: {'size': [-1, -1], 'type': [torch.Tensor]} }

    def encode_batch(self, data_dict):
        """
        Encodes batch, or, in fact, only one field of batch ("inputs").
        Stores result in "encoded_inputs" field of data_dict.

        :param data_dict: :py:class:`miprometheus.utils.DataDict` object containing (among others):

            - "inputs": expected input containing list of samples [BATCH SIZE] x [string]
            - "encoded_inputs": added output tensor with encoded samples [BATCH_SIZE x INPUT_SIZE]
        """
        # Get inputs to be encoded.
        inputs = data_dict[self.name_decoded_inputs]
        encoded_inputs_list = []
        # Process samples 1 by one.
        for sample in inputs:
            # Encode sample
            encoded_sample = self.encode_sample(sample)
            # Add to list plus unsqueeze inputs dimension(!)
            encoded_inputs_list.append( encoded_sample.unsqueeze(0) )
        # Concatenate inputs.
        encoded_inputs = torch.cat(encoded_inputs_list, dim=0)
        # Create the returned dict.
        self.extend_data_dict(data_dict, {self.name_encoded_inputs: None})
        data_dict[self.name_encoded_inputs] = encoded_inputs


    def encode_sample(self, sentence):
        """
        Generates a bag-of-word vector of length `encoded_input_size`.

        :return: torch.LongTensor [INPUT_SIZE]
        """
        # Create empty vector.
        vector = torch.zeros(self.item_size)
        # Encode each word and add its "representation" to vector.
        for word in sentence.split():
            vector[self.word_to_ix[word]] += 1
        return vector


    def decode_batch(self, data_dict):
        """ 
        Method DOES NOT change data dict.
        """ 
        pass

    def decode_sample(self, vector):
        """
        Method DOES NOT anything, as bow transformation cannot be reverted.
        """
        pass

class WordEncoder(Encoder):
    """
    Simple word encoder. Encodes a given input word into a unique index.
    """
    def  __init__(self, params, default_input_values):
        """
        Initializes the simple word encoded by creating dictionary mapping ALL words from training, validation and test sets into unique indices.

        :param params: Dictionary of parameters (read from configuration ``.yaml`` file).

        :param default_input_values: Dictionary containing default_input_values.
        """
        # Call base class constructor.
        super(WordEncoder, self).__init__('WordEncoder', params, default_input_values)

        # Dummy data.
        self.word_to_ix = {"SPANISH": 0, "ENGLISH": 1}
        self.ix_to_word = ["SPANISH", "ENGLISH"]
        
        self.num_classes = len(self.word_to_ix)

        # Define the default_values dict: holds parameters values that a model may need.
        self.default_values = {'num_classes': self.num_classes}
        # Set default data_definitions dict.
        self.data_definitions = {
            self.name_encoded_inputs: {'size': [-1, -1], 'type': [torch.Tensor]}, # 'encoded_targets'
            self.name_decoded_outputs: {'size': [-1, 1], 'type': [list, str]} # "encoded_targets"
            }

    def encode_batch(self, data_dict):
        """
        Encodes batch, or, in fact, only one field of bach ("inputs").
        Stores result in "encoded_inputs" field of in data_dict.

        :param data_dict: :py:class:`miprometheus.utils.DataDict` object containing (among others):

            - "inputs": expected input field containing list of words

            - "encoded_targets": added field containing output, tensor with encoded samples [BATCH_SIZE x 1] 
        """
        # Get inputs to be encoded.
        inputs = data_dict[self.name_decoded_inputs] # "languages"
        encoded_targets_list = []
        # Process samples 1 by one.
        for sample in inputs:
            # Encode sample
            encoded_sample = self.encode_sample(sample)
            # Add to list plus unsqueeze inputs dimension(!)
            encoded_targets_list.append( encoded_sample.unsqueeze(0) )
        # Concatenate inputs.
        encoded_targets = torch.cat(encoded_targets_list, dim=0)
        # Create the returned dict.
        self.extend_data_dict(data_dict, {self.name_encoded_inputs: None})
        data_dict[self.name_encoded_inputs] = encoded_targets

    def encode_sample(self, word):
        """
        Encodes a single word.

        :param word: A single word (string).

        :return: torch.LongTensor [1] (i.e. tensor of size 1)
        """
        return torch.LongTensor([self.word_to_ix[word]])

    def decode_batch(self, data_dict):
        """ 
        Method adds decoded predictions to data dict.

        :param data_dict: :py:class:`miprometheus.utils.DataDict` object containing (among others):

            - "outputs": expected input field containing (encoded) outputs, tensor [BATCH_SIZE x NUM_CLASSES]
            - "decoded_outputs": returned list of words [BATCH_SIZE] x [string] 
        """ 
        # Get inputs to be encoded.
        outputs = data_dict[self.name_encoded_outputs] # "predictions"
        decoded_outputs_list = []
        # Process samples 1 by one.
        for sample in outputs.chunk(outputs.size(0), 0):
            # Decode sample
            decoded_sample = self.decode_sample(sample)
            # Add to list plus unsqueeze batch dimension(!)
            decoded_outputs_list.append( decoded_sample )

        # Create the returned dict.
        self.extend_data_dict(data_dict, {self.name_decoded_outputs: None}) # "decoded_predictions"
        data_dict[self.name_decoded_outputs] = decoded_outputs_list 


    def decode_sample(self, vector):
        """
        Decodes vector into a single word.
        Handles with two types of inputs:

            - a single index: returns the associated word.
         
            - a vector containing a probability distribution: returns word associated with index with with max probability.

        :param vector: Single index or vector containing a probability distribution.

        :return: torch.LongTensor [1] (i.e. tensor of size 1)
        """
        return self.ix_to_word[vector.argmax(dim=1)]


if __name__ == "__main__":
    """ Tests sequence generator - generates and displays a random sample"""

    from miprometheus.utils.param_interface import ParamInterface
    # "Simulate" configuration.
    params = ParamInterface()
    params.add_config_params({
        'problem': {
            'data_folder': '~/data/language_identification',
            'use_train_data': True,
            'mappings' : {'inputs': 'sentences', 'targets': 'languages', 'encoded_targets': 'encoded_languages'}
        },
        'input_encoder': {
            'data_folder': '~/data/language_identification',
            'mappings' : {'decoded_inputs': 'sentences', 'encoded_inputs': 'encoded_sentences' }
        },
        'output_encoder': {
            'data_folder': '~/data/language_identification',
            'mappings' : {'decoded_inputs': 'languages', 'encoded_inputs': 'encoded_languages', 'decoded_outputs': 'decoded_predictions', 'encoded_outputs': 'encoded_predictions' }
        },
        'model': {
            'mappings' : {'inputs': 'encoded_sentences', 'predictions': 'encoded_predictions'}
        }})

    batch_size = 2

    # Create problem and model.
    problem  = LanguageIdentification(params["problem"])
    default_values = problem.default_values
    # Input (sentence) encoder.
    input_encoder = BOWSentenceEncoder(params["input_encoder"], default_values)
    input_encoder.extend_default_values(default_values)
    # Output (word) encoder.
    output_encoder = WordEncoder(params["output_encoder"], default_values)
    output_encoder.extend_default_values(default_values)
    # Model.
    model = SoftmaxClassifier(params["model"], default_values)

    # wrap DataLoader on top of this Dataset subclass
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset=problem, collate_fn=problem.collate_fn,
                            batch_size=batch_size, shuffle=True, num_workers=0)

    # Print the matrix column corresponding to "creo"
    #print(next(model.parameters())[:, word_to_ix["creo"]])

    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # Usually you want to pass over the training data several times.
    # 100 is much bigger than on a real data set, but real datasets have more than
    # two instances.  Usually, somewhere between 5 and 30 epochs is reasonable.
    for epoch in range(1000):
        for i, batch in enumerate(dataloader):
            # Step 1. Remember that PyTorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Encode inputs and targets.
            input_encoder.encode_batch(batch)
            output_encoder.encode_batch(batch)

            # Step 3. Run our forward pass.
            model(batch)
            
            # Decode predictions.
            output_encoder.decode_batch(batch)
            print("sequences: {} targets: {} \t\t  -> model predictions: {}".format(batch["sentences"], batch["languages"], batch["decoded_predictions"]))

            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss = problem.evaluate_loss(batch)
            #print("Loss = ", loss)

            loss.backward()
            optimizer.step()
