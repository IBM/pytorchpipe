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

__author__ = "Tomasz Kornuta"

import os
import json
from PIL import Image

import torch
from torchvision import transforms

from ptp.components.problems.problem import Problem
from ptp.data_types.data_definition import DataDefinition

#from ptp.components.utils.io import save_nparray_to_csv_file
from ptp.configuration.config_parsing import get_value_from_dictionary, get_value_list_from_dictionary
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

        # Check the resize image option.
        if "resize_image" in self.config:
            if len(self.config['resize_image']) != 2:
                self.logger.error("'resize_image' field must contain 2 values: the desired height and width")
                exit(-1)

            # Output image dimensions.
            self.height = self.config['resize_image'][0]
            self.width = self.config['resize_image'][1]
            self.depth = 3
            resize = True
        else:
            # Use original image dimensions.
            self.height = 480 
            self.width = 320 
            self.depth = 3
            resize = False
        self.logger.info("Setting image size to [D  x H x W]: {} x {} x {}".format(self.depth,  self.height, self.width))

        # Set global variables - all dimensions ASIDE OF BATCH.
        self.globals["image_height"] = self.height
        self.globals["image_width"] = self.width
        self.globals["image_depth"] = self.depth

        # Get image preprocessing.
        self.image_preprocessing = get_value_list_from_dictionary(
            "image_preprocessing", self.config,
            'none | normalize | all'.split(" | ")
            )
        if 'none' in self.image_preprocessing:
            self.image_preprocessing = []
        if 'all' in self.image_preprocessing:
            self.image_preprocessing = ['normalize']

        if resize:
            # Add resize as transformation.
                self.image_preprocessing = ["resize"] + self.image_preprocessing
        self.logger.info("Applied image preprocessing: {}".format(self.image_preprocessing))

        # Mapping of question subtypes to types (not used, but keeping it just in case).
        #self.question_subtype_to_type_mapping = {
        #    'query_size': 'query_attribute',
        #    'equal_size': 'compare_attribute',
        #    'query_shape': 'query_attribute',
        #    'query_color': 'query_attribute',
        #    'greater_than': 'compare_integer',
        #    'equal_material': 'compare_attribute',
        #    'equal_color': 'compare_attribute',
        #    'equal_shape': 'compare_attribute',
        #    'less_than': 'compare_integer',
        #    'count': 'count',
        #    'exist': 'exist',
        #    'equal_integer': 'compare_integer',
        #    'query_material': 'query_attribute'}

        # Mapping of question subtypes to types.
        self.question_subtype_to_id_mapping = {
            'query_size': 0,
            'equal_size': 1,
            'query_shape': 2,
            'query_color': 3,
            'greater_than': 4,
            'equal_material': 5,
            'equal_color': 6,
            'equal_shape': 7,
            'less_than': 8,
            'count': 9,
            'exist': 10,
            'equal_integer': 11,
            'query_material': 12}

        # Mapping of question families to subtypes.
        self.question_family_id_to_subtype_mapping = {
            0: "equal_integer", 1: "less_than", 2: "greater_than", 3: "equal_integer", 4: "less_than", 5: "greater_than", 6: "equal_integer", 7: "less_than", 8: "greater_than", 9: "equal_size",
            10: "equal_color", 11: "equal_material", 12: "equal_shape", 13: "equal_size", 14: "equal_size", 15: "equal_size", 16: "equal_color", 17: "equal_color", 18: "equal_color", 19: "equal_material",
            20: "equal_material", 21: "equal_material", 22: "equal_shape", 23: "equal_shape", 24: "equal_shape", 25: "count", 26: "exist", 27: "query_size", 28: "query_shape", 29: "query_color",
            30: "query_material", 31: "count", 32: "query_size", 33: "query_color", 34: "query_material", 35: "query_shape", 36: "exist", 37: "exist", 38: "exist", 39: "exist",
            40: "count", 41: "count", 42: "count", 43: "count", 44: "exist", 45: "exist", 46: "exist", 47: "exist", 48: "count", 49: "count",
            50: "count", 51: "count", 52: "query_color", 53: "query_material", 54: "query_shape", 55: "query_size", 56: "query_material", 57: "query_shape", 58: "query_size", 59: "query_color",
            60: "query_shape", 61: "query_size", 62: "query_color", 63: "query_material", 64: "count", 65: "count", 66: "count", 67: "count", 68: "count", 69: "count",
            70: "count", 71: "count", 72: "count", 73: "exist", 74: "query_size", 75: "query_color", 76: "query_material", 77: "query_shape", 78: "count", 79: "exist",
            80: "query_size", 81: "query_color", 82: "query_material", 83: "query_shape", 84: "count", 85: "exist", 86: "query_shape", 87: "query_material", 88: "query_color", 89: "query_size"}

        # Finally, "merge" those two.
        self.question_family_id_to_subtype_id_mapping = { key: self.question_subtype_to_id_mapping[value] for key, value in self.question_family_id_to_subtype_mapping.items() }


        # Get the absolute path.
        self.data_folder = os.path.expanduser(self.config['data_folder'])

        # Get split.
        split = get_value_from_dictionary('split', self.config, "training | validation | test | cogent_a_training | cogent_a_validation | cogent_b_validation".split(" | "))

        # Set split-dependent data.
        if split == 'training':
            # Training split folder and file with data question.
            data_file = os.path.join(self.data_folder, "questions", 'CLEVR_train_questions.json')
            self.split_image_folder = os.path.join(self.data_folder, "images", "train")

        elif split == 'validation':
            # Validation split folder and file with data question.
            data_file = os.path.join(self.data_folder, "questions", 'CLEVR_val_questions.json')
            self.split_image_folder = os.path.join(self.data_folder, "images", "val")

        elif split == 'test':
            # Test split folder and file with data question.
            data_file = os.path.join(self.data_folder, "questions", 'CLEVR_test_questions.json')
            self.split_image_folder = os.path.join(self.data_folder, "images", "test")

        else: # cogent
            raise ConfigurationError("Split {} not supported yet".format(split))

        # Load dataset.
        self.dataset = self.load_dataset(data_file)
        
        # Display exemplary sample.
        i = 0
        self.logger.info("Exemplary sample {} ({}):\n  question_type: {} ({})\n  image_ids: {}\n  question: {}\n  answer: {}".format(
            i, self.dataset[i]["question_index"],
            self.question_family_id_to_subtype_mapping[self.dataset[i]["question_family_index"]],
            self.question_family_id_to_subtype_id_mapping[self.dataset[i]["question_family_index"]],
            self.dataset[i]["image_filename"],
            self.dataset[i]["question"],
            self.dataset[i]["answer"]
            ))



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


    def load_dataset(self, source_data_file):
        """
        Loads the dataset from source file

        :param source_data_file: jSON file with image ids, questions, answers, scene graphs, etc.

        """
        self.logger.info("Loading dataset from:\n {}".format(source_data_file))
        dataset = []

        with open(source_data_file) as f:
            self.logger.info('Loading samples from {} ...'.format(source_data_file))
            dataset = json.load(f)['questions']

        self.logger.info("Loaded dataset consisting of {} samples".format(len(dataset)))
        return dataset


    def get_image(self, img_id):
        """
        Function loads and returns image along with its size.
        Additionally, it performs all the required transformations.

        :param img_id: Identifier of the images.
        :param img_folder: Path to the image.

        :return: image (Tensor)
        """

        # Load the image.
        img = Image.open(os.path.join(self.split_image_folder, img_id)).convert('RGB')

        image_transformations_list = []

        # Optional: resize.
        if 'resize' in self.image_preprocessing:
            image_transformations_list.append(transforms.Resize([self.height,self.width]))

        # Add obligatory transformation.
        image_transformations_list.append(transforms.ToTensor())

        # Optional: normalization.
        if 'normalize' in self.image_preprocessing:
            # Use normalization that the pretrained models from TorchVision require.
            image_transformations_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        # Resize the image and transform to Torch Tensor.
        transforms_com = transforms.Compose(image_transformations_list)
        # Apply transformations.
        img = transforms_com(img)

        # Return image.
        return img

    def __getitem__(self, index):
        """
        Getter method to access the dataset and return a single sample.

        :param index: index of the sample to return.
        :type index: int

        :return: DataDict({'indices', 'images', 'images_ids','questions', 'answers', 'question_type_ids', 'question_type_names'})
        """
        # Get item.
        item = self.dataset[index]

        # Create the resulting sample (data dict).
        data_dict = self.create_data_dict(index)

        # Load and stream the image ids.
        img_id = item["image_filename"]
        data_dict[self.key_image_ids] = img_id

        # Load the adequate image - only when required.
        if self.stream_images:
            img = self.get_image(img_id)
            # Image related variables.
            data_dict[self.key_images] = img

        # Return question.
        data_dict[self.key_questions] = item["question"]

        # Return answer. 
        data_dict[self.key_answers] = item["answer"]

        # Question type related variables.
        data_dict[self.key_question_type_ids] = self.question_family_id_to_subtype_id_mapping[item["question_family_index"]]
        data_dict[self.key_question_type_names] = self.question_family_id_to_subtype_mapping[item["question_family_index"]]

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
