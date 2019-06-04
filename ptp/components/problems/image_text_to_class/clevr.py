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

__author__ = "Tomasz Kornuta, Vincent Marois"

import os
import string
import tqdm

#import pandas as pd
#from PIL import Image
#import numpy as np
#import nltk
import json

import torch
#from torchvision import transforms

from ptp.components.problems.problem import Problem
from ptp.data_types.data_definition import DataDefinition

#from ptp.components.utils.io import save_nparray_to_csv_file
from ptp.configuration.config_parsing import get_value_from_dictionary
from ptp.configuration.configuration_error import ConfigurationError

class CLEVR(Problem):
    """
    Problem providing data associated with CLEVR (Compositional Language andElementary Visual Reasoning) diagnostics dataset

    The dataset consists of three splits:
        - A training set of 70,000 images and 699,989 questions
        - A validation set of 15,000 images and 149,991 questions
        - A test set of 15,000 images and 14,988 questions
        - Answers for all train and val questions
        - Scene graph annotations for train and val images giving ground-truth locations, attributes, and relationships for objects
        - Functional program representations for all training and validation images

    CLEVR contains a total of 90 question families, eachwith a single program template and an average of four texttemplates.
    Those are further aggregated into 13 Question Types:
        - Querying attributes (Size, Color, Material, Shape)
        - Comparing attributes (Size, Color, Material, Shape)
        - Existence
        - Counting
        - Integer comparison (Equal, Less, More)

    For more details please refer to the associated _website or _paper for more details.
    Test set with answers can be downloaded from a separate repository _repo.

    .. _website: https://cs.stanford.edu/people/jcjohns/clevr/

    .._paper: https://arxiv.org/pdf/1612.06890

    """
    def __init__(self, name, config):
        """
        Initializes problem object. Calls base constructor. Downloads the dataset if not present and loads the adequate files depending on the mode.

        :param name: Name of the component.

        :param class_type: Class type of the component.

        :param config: Dictionary of parameters (read from configuration ``.yaml`` file).
        """
        # Call constructors of parent classes.
        Problem.__init__(self, name, CLEVR, config)

        # (Eventually) download required packages.
        #nltk.download('punkt')
        #nltk.download('stopwords')

        # Get key mappings of all output streams.
        self.key_images = self.stream_keys["images"]
        self.key_image_ids = self.stream_keys["image_ids"]
        self.key_questions = self.stream_keys["questions"]
        self.key_answers = self.stream_keys["answers"]
        self.key_question_type_ids = self.stream_keys["question_type_ids"]
        self.key_question_type_names = self.stream_keys["question_type_names"]

        # Get flag informing whether we want to stream images or not.
        self.stream_images = self.config['stream_images']

        # Output image dimensions.
        self.height = 480 # self.config['resize_image'][0]
        self.width = 320 #self.config['resize_image'][1]
        self.depth = 3

        # Set global variables - all dimensions ASIDE OF BATCH.
        self.globals["image_height"] = self.height
        self.globals["image_width"] = self.width
        self.globals["image_depth"] = self.depth

        # Mapping of question subtypes to types.
        self.question_type_subtype_mapping = {
            'query_size': 'query_attribute',
            'equal_size': 'compare_attribute',
            'query_shape': 'query_attribute',
            'query_color': 'query_attribute',
            'greater_than': 'compare_integer',
            'equal_material': 'compare_attribute',
            'equal_color': 'compare_attribute',
            'equal_shape': 'compare_attribute',
            'less_than': 'compare_integer',
            'count': 'count',
            'exist': 'exist',
            'equal_integer': 'compare_integer',
            'query_material': 'query_attribute'}



        # Get the absolute path.
        self.data_folder = os.path.expanduser(self.config['data_folder'])

        # Get split.
        split = get_value_from_dictionary('split', self.config, "training | validation | test | cogent_a_training | cogent_a_validation | cogent_b_validation".split(" | "))

        # Set split-dependent data.
        if split == 'training':
            # Training split folder and file with data question.
            split_image_folder = os.path.join(self.data_folder, "images", "train")
            data_file = os.path.join(self.data_folder, "questions", 'CLEVR_train_questions.json')

        elif split == 'validation':
            # Validation split folder and file with data question.
            split_image_folder = os.path.join(self.data_folder, "images", "val")
            data_file = os.path.join(self.data_folder, "questions", 'CLEVR_val_questions.json')

        elif split == 'test':
            # Test split folder and file with data question.
            split_image_folder = os.path.join(self.data_folder, "images", "test")
            data_file = os.path.join(self.data_folder, "questions", 'CLEVR_test_questions.json')

        else: # cogent
            raise ConfigurationError("Split {} not supported yet".format(split))

        # Load dataset.
        self.dataset = self.load_dataset(data_file, split_image_folder)

        # Display exemplary sample.
        #self.logger.info("Exemplary sample 0 ({}):\n [ category: {}\t image_ids: {}\t question: {}\t answer: {} ]".format(
        #    self.ix[0],
        #    self.category_idx_to_word[self.dataset[self.ix[0]][self.key_question_type_ids]],
        #    self.dataset[self.ix[0]][self.key_image_ids],
        #    self.dataset[self.ix[0]][self.key_questions],
        #    self.dataset[self.ix[0]][self.key_answers]
        #    ))



    def output_data_definitions(self):
        """
        Function returns a dictionary with definitions of output data produced the component.

        :return: dictionary containing output data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        # Add all "standard" streams.
        d = {
            self.key_indices: DataDefinition([-1, 1], [list, int], "Batch of sample indices [BATCH_SIZE] x [1]"),
            self.key_image_ids: DataDefinition([-1, 1], [list, str], "Batch of image names, each being a single word [BATCH_SIZE] x [STRING]"),
            self.key_question_type_ids: DataDefinition([-1], [torch.Tensor], "Batch of target question type indices, each being a single index [BATCH_SIZE]"),
            self.key_question_type_names: DataDefinition([-1, 1], [list, str], "Batch of target question type names, each being a single word [BATCH_SIZE] x [STRING]"),
            }
        
        # Return images only when required.
        if self.stream_images:
            d[self.key_images] = DataDefinition([-1, self.depth, self.height, self.width], [torch.Tensor], "Batch of images [BATCH_SIZE x IMAGE_DEPTH x IMAGE_HEIGHT x IMAGE_WIDTH]")

        # Add stream with questions.
        d[self.key_questions] = DataDefinition([-1, 1], [list, str], "Batch of questions, each being a string consisting of many words [BATCH_SIZE] x [STRING]")

        # Add stream with answers.
        d[self.key_answers]= DataDefinition([-1, 1], [list, str], "Batch of target answers, each being a string consisting of many words [BATCH_SIZE] x [STRING]")

        return d


    def __len__(self):
        """
        Returns the "size" of the "problem" (total number of samples).

        :return: The size of the problem.
        """
        return len(self.dataset)


    def load_dataset(self, source_data_file, source_image_folder):
        """
        Loads the dataset from source file

        :param source_data_file: jSON file with image ids, questions, answers, scene graphs, etc.

        :param source_image_folder: Folder containing image files.

        """
        self.logger.info("Loading dataset from:\n {}".format(source_data_file))
        # Set containing list of tuples.
        dataset = []

        with open(source_data_file) as f:
            self.logger.info('Loading samples from {} ...'.format(source_data_file))
            dataset = json.load(f)
        self.logger.info('Loaded {} samples'.format(len(dataset['questions'])))
        print(dataset["questions"][0])
        exit(1)


        self.logger.info("Loaded dataset consisting of {} samples".format(len(dataset)))
        # Return the created list.
        return dataset


    def __getitem__(self, index):
        """
        Getter method to access the dataset and return a single sample.

        :param index: index of the sample to return.
        :type index: int

        :return: DataDict({'indices', 'images', 'images_ids','questions', 'answers', 'category_ids', 'image_sizes'})
        """
        # Get item.
        item = self.dataset[self.ix[index]]

        # Create the resulting sample (data dict).
        data_dict = self.create_data_dict(index)

        # Load and stream the image ids.
        img_id = item[self.key_image_ids]
        data_dict[self.key_image_ids] = img_id

        # Load the adequate image - only when required.
        if self.stream_images:

            # Image related variables.
            data_dict[self.key_images] = item[self.key_images]

        # Return question.
        data_dict[self.key_questions] = item[self.key_questions]

        # Return answer. 
        data_dict[self.key_answers] = item[self.key_answers]

        # Question type related variables.
        data_dict[self.key_question_type_ids] = item[self.key_question_type_ids]
        data_dict[self.key_question_type_names] = self.category_idx_to_word[item[self.key_question_type_ids]]

        # Return sample.
        return data_dict


    def collate_fn(self, batch):
        """
        Combines a list of DataDict (retrieved with :py:func:`__getitem__`) into a batch.

        :param batch: list of individual samples to combine
        :type batch: list

        :return: DataDict({'indices', 'images', 'images_ids','questions', 'answers', 'category_ids', 'image_sizes'})

        """
        # Collate indices.
        data_dict = self.create_data_dict([sample[self.key_indices] for sample in batch])

        # Stack images.
        data_dict[self.key_image_ids] = [item[self.key_image_ids] for item in batch]
        if self.stream_images:
            data_dict[self.key_images] = torch.stack([item[self.key_images] for item in batch]).type(torch.FloatTensor)

        # Collate lists/lists of lists.
        data_dict[self.key_questions] = [item[self.key_questions] for item in batch]
        data_dict[self.key_answers] = [item[self.key_answers] for item in batch]

        # Stack categories.
        data_dict[self.key_question_type_ids] = torch.tensor([item[self.key_question_type_ids] for item in batch])
        data_dict[self.key_question_type_names] = [item[self.key_question_type_names] for item in batch]

        # Return collated dict.
        return data_dict
