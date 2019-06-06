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
import tqdm
from PIL import Image

import torch
from torchvision import transforms

from ptp.components.problems.problem import Problem
from ptp.data_types.data_definition import DataDefinition

from ptp.configuration.config_parsing import get_value_from_dictionary, get_value_list_from_dictionary
from ptp.configuration.configuration_error import ConfigurationError

class GQA(Problem):
    """
    Problem providing data associated with the GQA dataset (Question Answering on Image Scene Graphs).

    The dataset consists of 22M questions about various day-to-day images. 
    Each image is associated with a scene graph of the image's objects, attributes and relations.
    Each question is associated with a structured representation of its semantics, a functional program
    that specifies the reasoning steps have to be taken to answer it.

    For more details please refer to the associated _website or _paper for more details.
    Test set with answers can be downloaded from a separate repository _repo.

    .. _website: https://cs.stanford.edu/people/dorarad/gqa/index.html

    .._paper: http://openaccess.thecvf.com/content_CVPR_2019/html/Hudson_GQA_A_New_Dataset_for_Real-World_Visual_Reasoning_and_Compositional_CVPR_2019_paper.html

    """
    def __init__(self, name, config):
        """
        Initializes problem object. Calls base constructor. Downloads the dataset if not present and loads the adequate files depending on the mode.

        :param name: Name of the component.

        :param class_type: Class type of the component.

        :param config: Dictionary of parameters (read from configuration ``.yaml`` file).
        """
        # Call constructors of parent classes.
        Problem.__init__(self, name, GQA, config)

        # Get key mappings of all output streams.
        self.key_sample_ids = self.stream_keys["sample_ids"]
        self.key_images = self.stream_keys["images"]
        self.key_image_ids = self.stream_keys["image_ids"]
        self.key_questions = self.stream_keys["questions"]
        self.key_answers = self.stream_keys["answers"]
        self.key_full_answers = self.stream_keys["full_answers"]

        # Get flag informing whether we want to stream images or not.
        self.stream_images = self.config['stream_images']

        # Check the resize image option.
        if len(self.config['resize_image']) != 2:
            self.logger.error("'resize_image' field must contain 2 values: the desired height and width")
            exit(-1)

        # Output image dimensions.
        self.height = self.config['resize_image'][0]
        self.width = self.config['resize_image'][1]
        self.depth = 3
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
        # Add resize as transformation.
        self.image_preprocessing = ["resize"] + self.image_preprocessing

        self.logger.info("Applied image preprocessing: {}".format(self.image_preprocessing))

        # Get the absolute path.
        self.data_folder = os.path.expanduser(self.config['data_folder'])

        # Get split.
        split = get_value_from_dictionary('split', self.config, "training_0 | training | validation | test_dev | test".split(" | "))
        self.split_image_folder = os.path.join(self.data_folder, "images")

        # Set split-dependent data.
        if split == 'training':
            # Training split folder and file with data question.
            data_files = []
            for i in range(10):
                data_files.append(os.path.join(self.data_folder, "questions1.2", "train_all_questions", "train_all_questions_{}.json".format(i)))

        elif split == 'training_0':
            # Validation split folder and file with data question.
            data_files = [ os.path.join(self.data_folder, "questions1.2", "train_all_questions", "train_all_questions_0.json") ]
            self.logger.warning("Please remember that this split constitutes only 10 percent of the whole training set!")

        elif split == 'validation':
            # Validation split folder and file with data question.
            data_files = [ os.path.join(self.data_folder, "questions1.2", "val_all_questions.json") ]
            self.logger.warning("Please use 'test_dev' split for validation!")

        elif split == 'test_dev':
            # Validation split folder and file with data question.
            data_files = [ os.path.join(self.data_folder, "questions1.2", "testdev_all_questions.json") ]

        elif split == 'test':
            # Test split folder and file with data question.
            data_files = [ os.path.join(self.data_folder, "questions1.2", "test_all_questions.json") ]

        else:
            raise ConfigurationError("Split {} not supported yet".format(split))

        # Load dataset.
        self.dataset = self.load_dataset(data_files)
        
        # Display exemplary sample.
        i = 0
        sample = self.dataset[i]
        # Check if this is a test set.
        self.logger.info("Exemplary sample {} ({}):\n  image_ids: {}\n  question: {}\n  answer: {} ({})".format(
            i,
            sample[self.key_sample_ids],
            sample[self.key_image_ids],
            sample[self.key_questions],
            sample[self.key_answers],
            sample[self.key_full_answers]
            ))


    def output_data_definitions(self):
        """
        Function returns a dictionary with definitions of output data produced the component.

        :return: dictionary containing output data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        # Add all "standard" streams.
        d = {
            self.key_indices: DataDefinition([-1, 1], [list, int], "Batch of sample indices [BATCH_SIZE] x [1]"),
            self.key_sample_ids: DataDefinition([-1, 1], [list, int], "Batch of sample ids [BATCH_SIZE] x [1]"),
            self.key_image_ids: DataDefinition([-1, 1], [list, str], "Batch of image names, each being a single word [BATCH_SIZE] x [STRING]"),
            }
        
        # Return images only when required.
        if self.stream_images:
            d[self.key_images] = DataDefinition([-1, self.depth, self.height, self.width], [torch.Tensor], "Batch of images [BATCH_SIZE x IMAGE_DEPTH x IMAGE_HEIGHT x IMAGE_WIDTH]")

        # Add stream with questions.
        d[self.key_questions] = DataDefinition([-1, 1], [list, str], "Batch of questions, each being a string consisting of many words [BATCH_SIZE] x [STRING]")

        # Add stream with answers.
        d[self.key_answers]= DataDefinition([-1, 1], [list, str], "Batch of target answers, each being a string consisting of few words (still treated as a single label) [BATCH_SIZE] x [STRING]")
        d[self.key_full_answers]= DataDefinition([-1, 1], [list, str], "Batch of target full (long) answers, each being a string consisting of many words [BATCH_SIZE] x [STRING]")

        return d


    def __len__(self):
        """
        Returns the "size" of the "problem" (total number of samples).

        :return: The size of the problem.
        """
        return len(self.dataset)


    def load_dataset(self, source_files):
        """
        Loads the dataset from source files.

        :param source_files: list of jSON file with image ids, questions, answers, scene graphs, etc.

        """
        self.logger.info("Loading dataset from:\n {}".format(source_files))
        dataset = []

        # Load and process files, one by one.
        for source_file in source_files:
            with open(source_file) as f:
                self.logger.info("Loading samples from '{}'...".format(source_file))
                json_dataset = json.load(f)
                # Process samples.

                # Add tdqm bar.
                t = tqdm.tqdm(total=len(json_dataset))
                for key,value in json_dataset.items():
                    # New sample.
                    sample = {}
                    sample[self.key_sample_ids] = key
                    sample[self.key_image_ids] = value["imageId"]
                    sample[self.key_questions] = value["question"]

                    # Return answer. 
                    if "answer" in value.keys():
                        sample[self.key_answers] = value["answer"]
                        sample[self.key_full_answers] = value["fullAnswer"]
                    else:
                        # Test set.
                        sample[self.key_answers] = "<UNK>"
                        sample[self.key_full_answers] = "<UNK>"

                    # Add to dataset.
                    dataset.append(sample)
                    t.update()
                # Close the bar.
                t.close()

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
        img = Image.open(os.path.join(self.split_image_folder, img_id+".jpg")).convert('RGB')

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

        :return: DataDict({'indices', 'sample_ids', images', 'images_ids','questions', 'answers', 'full_answers'})
        """
        # Get item.
        item = self.dataset[index]

        # Create the resulting sample (data dict).
        data_dict = self.create_data_dict(index)

        # Return sample id.
        data_dict[self.key_sample_ids] = item[self.key_sample_ids]

        # Load and stream the image ids.
        img_id = item[self.key_image_ids]
        data_dict[self.key_image_ids] = img_id

        # Load the adequate image - only when required.
        if self.stream_images:
            data_dict[self.key_images] = self.get_image(img_id)

        # Return question.
        data_dict[self.key_questions] = item[self.key_questions]

        # Return answers.
        data_dict[self.key_answers] = item[self.key_answers]
        data_dict[self.key_full_answers] = item[self.key_full_answers]

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

        # Collate sample ids.
        data_dict[self.key_sample_ids] = [item[self.key_sample_ids] for item in batch]

        # Stack images.
        data_dict[self.key_image_ids] = [item[self.key_image_ids] for item in batch]
        if self.stream_images:
            data_dict[self.key_images] = torch.stack([item[self.key_images] for item in batch]).type(torch.FloatTensor)

        # Collate lists/lists of lists.
        data_dict[self.key_questions] = [item[self.key_questions] for item in batch]
        data_dict[self.key_answers] = [item[self.key_answers] for item in batch]
        data_dict[self.key_full_answers] = [item[self.key_full_answers] for item in batch]

        # Return collated dict.
        return data_dict
