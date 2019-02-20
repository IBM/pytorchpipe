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
import errno
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ptp.utils.component import Component
from ptp.utils.problem import Problem


#torch.manual_seed(1)


class SoftmaxClassifier(nn.Module, Component): 
    """
    Simple Classifier consisting of fully connected layer with log softmax non-linearity.
    """
    def __init__(self, name, params):
        """
        Initializes the classifier.

        :param params: Dictionary of parameters (read from configuration ``.yaml`` file).
        """
        # Call constructors of parent classes.
        Component.__init__(self, name, params)
        nn.Module.__init__(self)

        # Set key mappings.
        self.key_inputs = self.mapkey("inputs")
        self.key_predictions = self.mapkey("predictions")

        # Retrieve input (vocabulary) size and number of classes from default params.
        self.input_size = 26 # TODO!   input_default_values['encoded_input_size']
        self.num_classes = 2 # TODO! input_default_values['num_classes']

        # Simple classifier.
        self.linear = nn.Linear(self.input_size, self.num_classes)
        
        # Set default data_definitions dict.
        # Encoded with BoW its is [BATCH_SIZE x NUM_CLASSES] !
        self.data_definitions = {self.key_predictions: {'size': [-1, self.num_classes], 'type': [torch.Tensor]} }

    def forward(self, data_dict):
        """
        Forward pass of the model.

        :param data_dict: DataDict({'inputs', 'predictions ...}), where:

            - inputs: expected inputs [BATCH_SIZE x INPUT_SIZE],
            - predictions: returned output with predictions (log_probs) [BATCH_SIZE x NUM_CLASSES]
        """
        inputs = data_dict[self.key_inputs]
        predictions = F.log_softmax(self.linear(inputs), dim=1)
        # Add them to datadict.
        data_dict.extend({self.key_predictions: predictions})


def save_list_to_txt_file(folder, filename, data):
    """ 
    Writes data to txt file.
    """
    # Check directory existence.
    if not os.path.exists(os.path.dirname(folder)):
        try:
            os.makedirs(os.path.dirname(folder))
        except OSError as exc:
            # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise        
    # Write elements in separate lines.        
    with open(folder+'/'+filename, mode='w+') as txtfile:
        txtfile.write('\n'.join(data))


def load_list_from_txt_file(folder, filename):
    """
    Loads data from txt file.
    """
    data = []
    with open(folder+'/'+filename, mode='rt') as txtfile:
        for line in txtfile:
            if line[-1] == '\n':
                line = line[:-1]
            data.append(line)
    return data


def load_dict_from_csv_file(folder, filename):
    """
    Loads data from csv file.

    .. warning::
            There is an assumption that file will contain key:value pairs (no content checking for now!)

    :param filename: File with encodings (absolute path + filename).
    :return: dictionary with word:index keys
    """        
    file_path = os.path.expanduser(folder) + "/" + filename

    with open(file_path, mode='rt') as csvfile:
        # Check the presence of the header.
        sniffer = csv.Sniffer()
        first_bytes = str(csvfile.read(256))
        has_header = sniffer.has_header(first_bytes)
        # Rewind.
        csvfile.seek(0)  
        reader = csv.reader(csvfile)
        # Skip the header row.
        if has_header:
            next(reader)  
        # Read the remaining rows.
        ret_dict = {rows[0]:int(rows[1]) for rows in reader}
    return ret_dict


def save_dict_to_csv_file(folder, filename, word_to_ix, fieldnames = []):
    """
    Saves dictionary to a file.

    :param filename: File with encodings (absolute path + filename).
    :param word_to_ix: dictionary with word:index keys
    """
    file_path = os.path.expanduser(folder) + "/" + filename

    # Check directory existence.
    if not os.path.exists(os.path.dirname(folder)):
        try:
            os.makedirs(os.path.dirname(folder))
        except OSError as exc:
            # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise        

    with open(file_path, mode='w+') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # Create header.
        writer.writeheader()

        # Write word-index pairs.
        for (k,v) in word_to_ix.items():
            #print("{} : {}".format(k,v))
            writer.writerow({fieldnames[0]:k, fieldnames[1]: v})


class LanguageIdentification(Problem):
    """
    Language identification (classification) problem.
    """

    def __init__(self, name, params):
        """
        Initializes problem object. Calls base constructor.

        :param name: Name of the component.

        :param params: Dictionary of parameters (read from configuration ``.yaml`` file).
        """
        # Call constructors of parent classes.
        Component.__init__(self, name, params)

        # Set key mappings.
        self.key_inputs = self.mapkey("inputs")
        self.key_targets = self.mapkey("targets")

        # Set default parameters.
        self.params.add_default_params({'data_folder': '~/data/language_identification',
                                        'use_train_data': True
                                        })
        # Get absolute path.
        self.data_folder = os.path.expanduser(self.params['data_folder'])
        
        # Retrieve parameters from the dictionary.
        self.use_train_data = self.params['use_train_data']

        # Set default data_definitions dict.
        self.data_definitions = {self.key_inputs: {'size': [-1, 1], 'type': [list, str]}, 
                                # [BATCH x SENTENCE (list of words as a single string)]
                                self.key_targets: {'size': [-1, 1], 'type': [list, str]}
                                # [BATCH x WORD (word as a single string)]
                                }

    def __getitem__(self, index):
        """
        Getter method to access the dataset and return a sample.

        :param index: index of the sample to return.
        :type index: int

        :return: ``DataDict({'inputs','targets'})``

        """
        # Return data_dict.
        data_dict = self.create_data_dict(index)
        data_dict[self.key_inputs] = self.inputs[index]
        data_dict[self.key_targets] = self.targets[index]
        return data_dict


class DummyLanguageIdentification(LanguageIdentification):
    """
    Simple Language identification (classification) problem.
    Data taken from the _example.

    .. _example: https://pytorch.org/tutorials/beginner/nlp/deep_learning_tutorial.html
    """
    def __init__(self, name, params):
        # Call constructors of parent classes.
        LanguageIdentification.__init__(self, name, params) 

        # Set default parameters.
        self.params.add_default_params({
            'generate': True
            })
        # Generate the dataset (can be turned off).    
        if self.params['generate']:
            self.generate_dummy_dataset()

        if self.use_train_data:
            inputs_file = "x_training.txt"
            targets_file = "y_training.txt"
        else:
            inputs_file = "x_test.txt"
            targets_file = "y_test.txt"

        # Load files.
        self.inputs = load_list_from_txt_file(self.data_folder, inputs_file)
        self.targets = load_list_from_txt_file(self.data_folder, targets_file)

        print("Inputs =", self.inputs)
        print("Targets =", self.targets)

        # Assert that they are equal in size!
        assert len(self.inputs) == len(self.targets), "Number of inputs loaded from {} not equal to number of targets loaded from {}!".format(inputs_file, targets_file)

    def __len__(self):
        """
        Returns the "size" of the "problem" (total number of samples).

        :return: The size of the problem.
        """
        return len(self.inputs)


    def generate_dummy_dataset(self):
        """
        Method generates dummy dataset for language identification, few (sentence-language) pairs, training and text sets.
        """
        self.logger.info("Generating dummy dataset in {}".format(self.data_folder))
        # "Training" set.
        x_training_data = [
            "me gusta comer en la cafeteria",
            "Give it to me", 
            "No creo que sea una buena idea",
            "No it is not a good idea to get lost at sea"]
        save_list_to_txt_file(self.data_folder, 'x_training.txt', x_training_data)

        y_training_data = [
            "SPANISH",
            "ENGLISH",
            "SPANISH",
            "ENGLISH"]
        save_list_to_txt_file(self.data_folder, 'y_training.txt', y_training_data)

        # "Test" set.
        x_test_data = [
            "Yo creo que si",
            "it is lost on me"]
        save_list_to_txt_file(self.data_folder, 'x_test.txt', x_test_data)

        y_test_data = [
            "SPANISH",
            "ENGLISH"]
        save_list_to_txt_file(self.data_folder, 'y_test.txt', y_test_data)


class NLLLoss(Component):
    """
    Component calculating the negative log likelihood loss.
    """

    def __init__(self, name, params):
        """
        Initializes object, calls base constructor, initializes names of input and output ports.

        :param params: Dictionary of parameters (read from configuration ``.yaml`` file).

        """
        # Call constructors of parent classes.
        Component.__init__(self, name, params)

        # Set key mappings.
        self.key_targets = self.mapkey("targets")
        self.key_predictions = self.mapkey("predictions")
        self.key_loss = self.mapkey("loss")

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


class SentenceTokenizer(Component):
    """
    Class responsible for tokenizing the sentence.
    """
    def __init__(self, name, params):
        # Call constructors of parent classes.
        Component.__init__(self, name, params)

        # Set default parameters.
        params.add_default_params({
            'detokenize': False, # Mode - False: sentence -> list of strings, True: list of strings -> sentence.
            })
        # Read the actual configuration.
        self.mode_detokenize = params['detokenize']

        # Set key mappings.
        self.key_inputs = self.mapkey("inputs")
        self.key_outputs = self.mapkey("outputs")

        if self.mode_detokenize:
            # list of strings -> sentence.
            self.processor = self.detokenize_sample
        else:
            # sentence -> list of strings.
            self.processor = self.tokenize_sample
        # Ok, we are ready to go!

    def tokenize_sample(self, sample):
        """
        Changes sample (sentence) into list of tokens (words).

        :param sample: sentence (string).

        :return: list of words (strings).
        """
        return sample.split()

    def detokenize_sample(self, sample):
        """
        Changes list of tokens (words) into sentence.

        :param sample: list of words (strings).

        :return: sentence (string).
        """
        return ''.join([str(x) for x in sample])

    def __call__(self, data_dict):
        """
        Encodes batch, or, in fact, only one field of bach ("inputs").
        Stores result in "encoded_inputs" field of in data_dict.

        :param data_dict: :py:class:`ptp.utils.DataDict` object containing (among others):

            - "inputs": expected input field containing list of words

            - "encoded_targets": added field containing output, tensor with encoded samples [BATCH_SIZE x 1] 
        """
        # Get inputs to be encoded.
        inputs = data_dict[self.key_inputs]
        outputs_list = []
        # Process samples 1 by one.
        for sample in inputs:
            output = self.processor(sample)
            # Add to outputs.
            outputs_list.append( output )
        # Create the returned dict.
        data_dict.extend({self.key_outputs: outputs_list})



class TokenEncoder(Component):
    """
    Abstract class responsible for encoding tokens. Please use derrived classes.
    """
    def __init__(self, name, params):
        # Call constructors of parent classes.
        Component.__init__(self, name, params)

        # Set default parameters.
        params.add_default_params({
            'data_folder': '~/data/',
            'source_files': '', # Source files
            'encodings_file': 'default_encodings.csv', # File containing encodings
            'regenerate': False # True means that it will be regenerated despite the existence of file.
            })

        # Read the actual configuration.
        self.data_folder = params['data_folder']
        self.source_files = params['source_files']
        self.encodings_file = params['encodings_file']
        self.mode_regenerate = params['regenerate']

        # Default name mappings for all encoders.
        self.key_inputs = self.mapkey("inputs")
        self.key_outputs = self.mapkey("outputs")

        # Encodings file.
        encodings_file_path = os.path.expanduser(self.data_folder) + "/" + self.encodings_file

        # Check whether we want to (re)generate new  or load existing encodings.
        if self.mode_regenerate or not os.path.exists(encodings_file_path):
            # Generate new encodings.
            self.word_to_ix = self.create_encodings(self.data_folder, self.source_files)
            assert (len(self.word_to_ix) > 0), "The created encodings list is empty!"
            # Ok, save necodings, so next time we will simply load them.
            save_dict_to_csv_file(self.data_folder, self.encodings_file, self.word_to_ix, ['word', 'index'])
        else:
            # Load encodings.
            self.word_to_ix = load_dict_from_csv_file(self.data_folder, self.encodings_file)
            assert (len(self.word_to_ix) > 0), "The loaded encodings list is empty!"
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

        for filename in source_files.split(','):
            # filename + path.
            fn = data_folder+ '/' + filename
            if not os.path.exists(fn):
                self.logger.warning("Cannot load tokens files from {} because file does not exist".format(fn))
                continue
            # File exists, try to parse.
            content = open(fn).read()
            # Parse tokens.
            for word in content.split():
                # If new token.
                if word not in word_to_ix:
                    word_to_ix[word] = len(word_to_ix)
        return word_to_ix


class WordEncoder(TokenEncoder):
    """
    class responsible for encoding of samples consisting of single words.
    """
    def __init__(self, name, params):
        # Call constructors of parent classes.
        TokenEncoder.__init__(self, name, params)

    def __call__(self, data_dict):
        """
        Encodes "inputs" in the format of a single word.
        Stores result in "outputs" field of in data_dict.

        :param data_dict: :py:class:`ptp.utils.DataDict` object containing (among others):

            - "inputs": expected input field containing list of words [BATCH_SIZE] x x [string]

            - "outputs": added output field containing list of indices  [BATCH_SIZE] x [1] 
        """
        # Get inputs to be encoded.
        inputs = data_dict[self.key_inputs]
        outputs_list = []
        # Process samples 1 by 1.
        for sample in inputs:
            assert not isinstance(sample, (list,)), 'This encoder requires input sample to contain a single word'
            # Process single token.
            output_sample = self.word_to_ix[sample]
            outputs_list.append(output_sample)
        # Create the returned dict.
        data_dict.extend({self.key_outputs: outputs_list})


class SentenceEncoder(TokenEncoder):
    """
    class responsible for encoding of samples being sequences of words.
    """
    def __init__(self, name, params):
        # Call constructors of parent classes.
        TokenEncoder.__init__(self, name, params)

    def __call__(self, data_dict):
        """
        Encodes "inputs" in the format of list of tokens (for a single sample)
        Stores result in "encoded_inputs" field of in data_dict.

        :param data_dict: :py:class:`ptp.utils.DataDict` object containing (among others):

            - "inputs": expected input field containing list of words [BATCH_SIZE] x [SEQ_SIZE] x [string]

            - "encoded_targets": added output field containing list of indices [BATCH_SIZE] x [SEQ_SIZE] x [1] 
        """
        # Get inputs to be encoded.
        inputs = data_dict[self.key_inputs]
        outputs_list = []
        # Process samples 1 by one.
        for sample in inputs:
            assert isinstance(sample, (list,)), 'This encoder requires input sample to contain a list of words'
            # Process list.
            output_sample = []
            # Encode sample (list of words)
            for token in sample:
                output_token = self.word_to_ix[token]
                # Add to outputs.
                output_sample.append( output_token )

            outputs_list.append(output_sample)
        # Create the returned dict.
        data_dict.extend({self.key_outputs: outputs_list})


class WordDecoder(TokenEncoder):
    """
    class responsible for decoding of samples encoded in the form of vectors ("probability distributions").
    """
    def __init__(self, name, params):
        # Call constructors of parent classes.
        TokenEncoder.__init__(self, name, params)
        # Construct reverse mapping for faster processing.
        self.ix_to_word = dict((v,k) for k,v in self.word_to_ix.items())

    def __call__(self, data_dict):
        """
        Encodes "inputs" in the format of a single word.
        Stores result in "outputs" field of in data_dict.

        :param data_dict: :py:class:`ptp.utils.DataDict` object containing (among others):

            - "inputs": expected input field containing tensor [BATCH_SIZE x NUM_CLASSES]

            - "outputs": added output field containing list of words [BATCH_SIZE] x [string] 
        """
        # Get inputs to be encoded.
        inputs = data_dict[self.key_inputs]
        outputs_list = []
        # Process samples 1 by 1.
        for sample in inputs.chunk(inputs.size(0), 0):
            # Process single token.
            max_index = sample.squeeze(0).argmax(dim=0).item() 
            output_sample = self.ix_to_word[max_index]
            outputs_list.append(output_sample)
        # Create the returned dict.
        data_dict.extend({self.key_outputs: outputs_list})


class BOWEncoder(Component):
    """
    Simple Bag-of-word type encoder that encodes the sentence (in the form of a list of encoded words) into a vector.
    
    .. warning::
        BoW transformation is inreversible, thus decode-related methods in fact return original inputs.
    """
    def  __init__(self, name, params):
        """
        Initializes the bag-of-word encoded by creating dictionary mapping ALL words from training, validation and test sets into unique indices.

        :param params: Dictionary of parameters (read from configuration ``.yaml`` file).
        """
        # Call constructors of parent classes.
        Component.__init__(self, name, params)

        # Default name mappings for all encoders.
        self.key_inputs = self.mapkey("inputs")
        self.key_outputs = self.mapkey("outputs")

        # Size of a single encoded item.
        self.item_size = 26#len(self.word_to_ix)

        # Define the default_values dict: holds parameters values that a model may need.
        #self.default_values = {'encoded_input_size': self.item_size}
        # Set default data_definitions dict.
        # Encoded with BoW its is [BATCH_SIZE x VOCAB_SIZE] !
        #self.data_definitions = {self.key_encoded_inputs: {'size': [-1, -1], 'type': [torch.Tensor]} }

    def __call__(self, data_dict):
        """
        Encodes batch, or, in fact, only one field of batch ("inputs").
        Stores result in "outputs" field of data_dict.

        :param data_dict: :py:class:`ptp.utils.DataDict` object containing (among others):

            - "inputs": expected input containing list of (list of tokens) [BATCH SIZE] x [SEQ_LEN] x [ITEM_SIZE]
            - "outputs": added output tensor with encoded words [BATCH_SIZE x ITEM_SIZE]
        """
        # Get inputs to be encoded.
        inputs = data_dict[self.key_inputs]
        outputs_list = []
        # Process samples 1 by one.
        for sample in inputs:
            # Encode sample
            output = self.encode_sample(sample)
            # Add to list plus unsqueeze inputs dimension(!)
            outputs_list.append( output.unsqueeze(0) )
        # Concatenate output tensors.
        outputs = torch.cat(outputs_list, dim=0)
        # Add result to the data dict.
        data_dict.extend({self.key_outputs: outputs})

    def encode_sample(self, list_of_tokens):
        """
        Generates a bag-of-word vector of length `output_size`.

        :param list_of_tokens: List of tokens [SEQ_LENGTH] x [ITEM_SIZE]
        :return: torch.LongTensor [ITEM_SIZE]
        """
        # Create empty vector.
        output = torch.zeros(self.item_size)
        # "Adds" tokens.
        for token in list_of_tokens:
            output[token] += 1
        return output


if __name__ == "__main__":
    """ Tests sequence generator - generates and displays a random sample"""
    # Set logging.
    import logging
    logging.basicConfig(level=logging.INFO)

    from ptp.utils.param_interface import ParamInterface
    # "Simulate" configuration.
    params = ParamInterface()
    params.add_config_params({
        'problem': {
            'name': 'LanguageIdentification',
            'data_folder': '~/data/language_identification/dummy',
            'use_train_data': True,
            'keymappings' : {'inputs': 'sentences', 'targets': 'languages'}
        },
        # Sentences encoding.
        'sentence_tokenizer': {
            'name': 'SentenceTokenizer',
            'keymappings' : {'inputs': 'sentences', 'outputs': 'tokenized_sentences'}
        },
        'sentence_encoder': {
            'name': 'SentenceEncoder',
            'data_folder': '~/data/language_identification/dummy',
            'source_files': 'x_training.txt,x_test.txt',
            'encodings_file': 'word_encodings.csv',
            'keymappings' : {'inputs': 'tokenized_sentences', 'outputs': 'encoded_sentences'}
        },
        'bow_encoder': {
            'name': 'BOWEncoder',
            'keymappings' : {'inputs': 'encoded_sentences', 'outputs': 'bow_sencentes'}
        },
        # Targets encoding.
        'target_encoder': {
            'name': 'WordEncoder',
            'data_folder': '~/data/language_identification/dummy',
            'source_files': 'y_training.txt,y_test.txt',
            'encodings_file': 'language_name_encodings.csv',
            'keymappings' : {'inputs': 'languages', 'outputs': 'encoded_languages'}
        },
        # Model
        'model': {
            'keymappings' : {'inputs': 'bow_sencentes', 'predictions': 'encoded_predictions'}
        },
        # Loss
        'nllloss': {
            'name': 'NLLLoss',
            'keymappings' : {'targets': 'encoded_languages', 'predictions': 'encoded_predictions', 'loss': 'loss' }
        },
        # Predictions decoder.
        'prediction_decoder': {
            'name': 'WordDecoder',
            'data_folder': '~/data/language_identification/dummy',
            'source_files': 'y_training.txt,y_test.txt',
            'encodings_file': 'language_name_encodings.csv',
            'decode': True, # Decoding mode!
            'keymappings' : {'inputs': 'encoded_predictions', 'outputs': 'predictions'}
        }

        })

    batch_size = 2

    # Create problem.
    problem  = DummyLanguageIdentification("problem", params["problem"])

    # Input (sentence) encoder.
    sentence_tokenizer = SentenceTokenizer("sentence_tokenizer", params["sentence_tokenizer"])
    sentence_encoder = SentenceEncoder("sentence_encoder", params["sentence_encoder"])
    bow_encoder = BOWEncoder("bow_encoder", params["bow_encoder"])

    # Target encoder.
    target_encoder = WordEncoder("target_encoder", params["target_encoder"])

    # Model.
    model = SoftmaxClassifier("model", params["model"])

    # Loss.
    loss = NLLLoss("nllloss", params["nllloss"])

    # Decoder.
    prediction_decoder  = WordDecoder("prediction_decoder", params["prediction_decoder"])

    # Constructd dataloader.
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset=problem, collate_fn=problem.collate_fn,
                            batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # Usually you want to pass over the training data several times.
    # 100 is much bigger than on a real data set, but real datasets have more than
    # two instances.  Usually, somewhere between 5 and 30 epochs is reasonable.
    for epoch in range(100):
        for i, batch in enumerate(dataloader):
            # Step 1. Remember that PyTorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Input (sentence) encoder.
            sentence_tokenizer(batch)
            sentence_encoder(batch)
            bow_encoder(batch)

            # Target encoder.
            target_encoder(batch)

            # Model.
            model(batch)

            # Loss.
            loss(batch)

            # Decoder.
            prediction_decoder(batch)

            print("sequences: {} targets: {} \t\t  -> model predictions: {}".format(batch["sentences"], batch["languages"], batch["predictions"]))

            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss_value = batch["loss"]
            #print("Loss = ", loss)

            loss_value.backward()
            optimizer.step()
