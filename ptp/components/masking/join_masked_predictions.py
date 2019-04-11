# -*- coding: utf-8 -*-
#
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
import numpy as np

from ptp.components.component import Component
from ptp.data_types.data_definition import DataDefinition


class JoinMaskedPredictions(Component):
    """
    Class responsible joining several prediction streams using the associated masks.
    Additionally, it returns the associated string indices.

    .. warning:
        As performed operations are not differentiable, the returned  'output_indices' cannot be used for e.g. calculation of loss!!

    """

    def __init__(self, name, config):
        """
        Initializes the object. Loads keys, word mappings and vocabularies.

        :param name: Name of the component read from the configuration file
        :type name: str

        :param config: Dictionary of parameters (read from the configuration ``.yaml`` file).
        :type config: :py:class:`ptp.configuration.ConfigInterface`

        """
        # Call constructors of parent classes.
        Component.__init__(self, name, JoinMaskedPredictions, config)

        # Get input key mappings.
        # Load list of prediction streams names (keys).
        self.input_prediction_stream_keys = self.config["input_prediction_streams"]
        if type(self.input_prediction_stream_keys) == str:
            self.input_prediction_stream_keys = self.input_prediction_stream_keys.replace(" ", "").split(",")
        #assert(self.input_prediction_stream_keys != ""), "ooo"

        # Load list of mask streams names (keys).
        self.input_mask_stream_keys = self.config["input_mask_streams"]
        if type(self.input_mask_stream_keys) == str:
            self.input_mask_stream_keys = self.input_mask_stream_keys.replace(" ", "").split(",")

        # Load list of word mappings names (keys).
        input_word_mappings_keys = self.config["input_word_mappings"]
        if type(input_word_mappings_keys) == str:
            input_word_mappings_keys = input_word_mappings_keys.replace(" ", "").split(",")

        # Retrieve input word mappings from globals.
        self.input_ix_to_word = []
        for wmk in input_word_mappings_keys:
            # Get word mappings.
            word_to_ix = self.globals[wmk]
            # Create inverse transformation.
            ix_to_word = {value: key for (key, value) in word_to_ix.items()}
            self.input_ix_to_word.append(ix_to_word)


        # Get output key mappings.
        self.key_output_indices = self.stream_keys["output_indices"]
        self.key_output_strings = self.stream_keys["output_strings"]

        # Retrieve output word mappings from globals.
        self.output_word_to_ix = self.globals["output_word_mappings"]


    def input_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of input data that are required by the component.

        :return: dictionary containing input data definitions (each of type :py:class:`ptp.data_types.DataDefinition`).
        """
        input_defs = {}
        # Add input prediction streams.
        for i, ipsk in enumerate(self.input_prediction_stream_keys):
            # Use input prediction stream key along with the length of the associated word mappings (i.e. size of the vocabulary = NUM_CLASSES)
            input_defs[ipsk] = DataDefinition([-1, len(self.input_ix_to_word[i])], [torch.Tensor], "Batch of predictions, represented as tensor with probability distribution over classes [BATCH_SIZE x NUM_CLASSES]")
        # Add mask streams.
        for imsk in self.input_mask_stream_keys:
            # Every mask has the same definition, but different stream key.
            input_defs[imsk] = DataDefinition([-1], [torch.Tensor], "Batch of masks [BATCH_SIZE]")

        return input_defs

    def output_data_definitions(self):
        """ 
        Function returns a empty dictionary with definitions of output data produced the component.

        :return: Empty dictionary.
        """
        return {
            self.key_output_indices: DataDefinition([-1], [torch.Tensor], "Batch of merged (output) indices [BATCH_SIZE]"),
            self.key_output_strings: DataDefinition([-1], [torch.Tensor], "Batch of merged strings, corresponging to indices when using the provided word mappings [BATCH_SIZE]")
            }


    def __call__(self, data_dict):
        """
        Encodes "inputs" in the format of a single tensor.
        Stores reshaped tensor in "outputs" field of in data_dict.

        :param data_dict: :py:class:`ptp.utils.DataDict` object containing (among others):

            - "inputs": expected input field containing tensor [BATCH_SIZE x ...]

            - "outputs": added output field containing tensor [BATCH_SIZE x ...] 
        """
        # Get inputs masks
        masks = []
        for imsk in self.input_mask_stream_keys:
            masks.append(data_dict[imsk].data.cpu().numpy())
        
        # Sum all masks and make sure that they are complementary.
        masks_sum = np.sum(masks, axis=0)
        batch_size = masks_sum.shape[0]
        sum_ones = sum(filter(lambda x: x == 1, masks_sum))
        if sum_ones != batch_size:
            self.logger.error("Masks received from the {} streams are not complementary!".format(self.input_mask_stream_keys))
            exit(-1)

        # Create mapping indicating from which input prediction/mask/dictionary stream we will take data associated with given "sample".
        weights = np.array(range(len(masks)))
        masks = np.array(masks).transpose()
        mapping = np.dot(masks, weights)
        #print("Mapping = \n",mapping)

        # "Translate". 
        output_answers = []
        output_indices = []
        output_predictions_lst = []
        # Iterate through samples.
        for sample in range(batch_size):
            # Get the right dictionary.
            ix_to_word = self.input_ix_to_word[mapping[sample]]
            #print(ix_to_word)

            # Get the right sample from the right prediction stream.
            sample_prediction = data_dict[self.input_prediction_stream_keys[mapping[sample]]][sample]
            #print(sample_prediction)
            output_predictions_lst.append(sample_prediction)

            # Get the index of max log-probabilities.
            index = sample_prediction.max(0)[1].data.cpu().item()
            #print(index)
            
            # Get the right word.
            word = ix_to_word[index]
            output_answers.append(word)

            # Get original index using output dictionary.
            output_indices.append(self.output_word_to_ix[word])

        #print(output_predictions_lst)
        #targets = data_dict["targets"].data.cpu().numpy()
        #print("targets = \n",targets.tolist())
        #print("joined answers = \n",output_indices)

        # Change to tensor.
        output_indices_tensor = torch.tensor(output_indices)

        # Extend the dict by returned output streams.
        data_dict.extend({
            self.key_output_indices: output_indices_tensor,
            self.key_output_strings: output_answers
            })
        
