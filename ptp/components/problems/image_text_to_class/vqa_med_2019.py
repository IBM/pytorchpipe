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

__author__ = "Chaitanya Shivade, Tomasz Kornuta"

import os
import string
import tqdm

import pandas as pd
from PIL import Image
import numpy as np
import nltk

import torch
from torchvision import transforms

from ptp.components.problems.problem import Problem
from ptp.data_types.data_definition import DataDefinition

from ptp.configuration.config_parsing import get_value_list_from_dictionary

class VQAMED2019(Problem):
    """
    Problem providing data associated with ImageCLEF VQA 2019 challenge.

    The dataset consists of four splits:

        - a training set of 3,200 medical images with 12,792 Question-Answer (QA) pairs,
        - a validation set of 500 medical images with 2,000 QA pairs, and
        - a test set of 500 medical images with 500 questions.

    Aside of that, there are 4 categories of questions based on:
        - Modality (C1),
        - Plane (C2),
        - Organ System (C3), and
        - Abnormality (C4).

    Please see the readme file of the crowdAI dataset section for more detailed information.
    For more details please refer to the associated _website or _crowdai websites for more details.

    .. _crowdai: https://www.crowdai.org/challenges/imageclef-2019-vqa-med

    .. _website: https://www.imageclef.org/2019/medical/vqa/
    """
    def __init__(self, name, config):
        """
        Initializes problem object. Calls base constructor. Downloads the dataset if not present and loads the adequate files depending on the mode.

        :param name: Name of the component.

        :param class_type: Class type of the component.

        :param config: Dictionary of parameters (read from configuration ``.yaml`` file).
        """
        # Call constructors of parent classes.
        Problem.__init__(self, name, VQAMED2019, config)

        # (Eventually) download required packages.
        nltk.download('punkt')
        nltk.download('stopwords')

        # Get key mappings of all output streams.
        self.key_images = self.stream_keys["images"]
        self.key_image_ids = self.stream_keys["image_ids"]
        self.key_questions = self.stream_keys["questions"]
        self.key_answers = self.stream_keys["answers"]
        self.key_category_ids = self.stream_keys["category_ids"]
        self.key_category_names = self.stream_keys["category_names"]
        self.key_image_sizes = self.stream_keys["image_sizes"]

        # Check the desired image size.
        if len(self.config['resize_image']) != 2:
            self.logger.error("'resize_image' field must contain 2 values: the desired height and width")
            exit(-1)

        # Output image dimensions.
        self.height = self.config['resize_image'][0]
        self.width = self.config['resize_image'][1]
        self.depth = 3

        # Set global variables - all dimensions ASIDE OF BATCH.
        self.globals["image_height"] = self.height
        self.globals["image_width"] = self.width
        self.globals["image_depth"] = self.depth

        # Those values will be used to rescale the image_sizes to range (0, 1).
        self.scale_image_height = self.config['scale_image_size'][0]
        self.scale_image_width = self.config['scale_image_size'][1]

        # Set parameters and globals related to categories.
        self.globals["num_categories"] = 6
        self.globals["category_word_mappings"] = {'C1': 0, 'C2': 1, 'C3': 2, 'C4': 3, 'BINARY': 4, '<UNK>': 5}
        self.category_idx_to_word = {0: 'C1', 1: 'C2', 2: 'C3', 3: 'C4', 4: 'BINARY', 5: '<UNK>'}

        # Get the absolute path.
        self.data_folder = os.path.expanduser(self.config['data_folder'])

        # Set split-dependent data.
        if self.config['split'] == 'training':
            # Training split folder.
            split_folder = os.path.join(self.data_folder, "ImageClef-2019-VQA-Med-Training")
            # Set source files.
            source_files = [
                os.path.join(split_folder,"QAPairsByCategory/C1_Modality_train.txt"),
                os.path.join(split_folder,"QAPairsByCategory/C2_Plane_train.txt"),
                os.path.join(split_folder,"QAPairsByCategory/C3_Organ_train.txt"),
                os.path.join(split_folder,"QAPairsByCategory/C4_Abnormality_train.txt")
                ]
            # Set image folders.
            source_image_folders = [os.path.join(split_folder, 'Train_images')] * 4

            # Set the categories associated with each of those files.
            source_categories = [0, 1, 2, 3]

            # Filter lists taking into account configuration.
            source_files, source_image_folders, source_categories = self.filter_sources(source_files, source_image_folders, source_categories)

        elif self.config['split'] == 'validation':
            # Validation split folder.
            split_folder = os.path.join(self.data_folder, "ImageClef-2019-VQA-Med-Validation")

            # Set source files.
            source_files = [
                os.path.join(split_folder,"QAPairsByCategory/C1_Modality_val.txt"),
                os.path.join(split_folder,"QAPairsByCategory/C2_Plane_val.txt"),
                os.path.join(split_folder,"QAPairsByCategory/C3_Organ_val.txt"),
                os.path.join(split_folder,"QAPairsByCategory/C4_Abnormality_val.txt")
                ]

            # Set image folders.
            source_image_folders = [os.path.join(split_folder, 'Val_images')] * 4

            # Set the categories associated with each of those files.
            source_categories = [0, 1, 2, 3]

            # Filter lists taking into account configuration.
            source_files, source_image_folders, source_categories = self.filter_sources(source_files, source_image_folders, source_categories)

        elif self.config['split'] == 'training_validation':
            # This split takes both training and validation and assumes utilization of kFoldWeightedRandomSampler.

            # 1. Training split folder.
            split_folder = os.path.join(self.data_folder, "ImageClef-2019-VQA-Med-Training")
            # Set source files.
            source_files = [
                os.path.join(split_folder,"QAPairsByCategory/C1_Modality_train.txt"),
                os.path.join(split_folder,"QAPairsByCategory/C2_Plane_train.txt"),
                os.path.join(split_folder,"QAPairsByCategory/C3_Organ_train.txt"),
                os.path.join(split_folder,"QAPairsByCategory/C4_Abnormality_train.txt")
                ]
            # Set image folders.
            source_image_folders = [os.path.join(split_folder, 'Train_images')] * 4

            # Set the categories associated with each of those files.
            source_categories = [0, 1, 2, 3]

            # Filter lists taking into account configuration.
            training_source_files, training_source_image_folders, training_source_categories = self.filter_sources(source_files, source_image_folders, source_categories)

            #2.  Validation split folder.
            split_folder = os.path.join(self.data_folder, "ImageClef-2019-VQA-Med-Validation")

            # Set source files.
            source_files = [
                os.path.join(split_folder,"QAPairsByCategory/C1_Modality_val.txt"),
                os.path.join(split_folder,"QAPairsByCategory/C2_Plane_val.txt"),
                os.path.join(split_folder,"QAPairsByCategory/C3_Organ_val.txt"),
                os.path.join(split_folder,"QAPairsByCategory/C4_Abnormality_val.txt")
                ]

            # Set image folders.
            source_image_folders = [os.path.join(split_folder, 'Val_images')] * 4

            # Set the categories associated with each of those files.
            source_categories = [0, 1, 2, 3]

            # Filter lists taking into account configuration.
            valid_source_files, valid_source_image_folders, valid_source_categories = self.filter_sources(source_files, source_image_folders, source_categories)

            # 3. Merge lists.
            source_files = [*training_source_files, *valid_source_files]
            source_image_folders = [*training_source_image_folders, *valid_source_image_folders]
            source_categories  = [*training_source_categories, *valid_source_categories]
        # else: # Test set. # TODO

        # Get image preprocessing.
        self.image_preprocessing = get_value_list_from_dictionary(
            "image_preprocessing", self.config,
            'none | random_affine | random_horizontal_flip | normalize | all'.split(" | ")
            )
        if 'none' in self.image_preprocessing:
            self.image_preprocessing = []
        if 'all' in self.image_preprocessing:
            self.image_preprocessing = 'random_affine | random_horizontal_flip | normalize'.split(" | ")
        self.logger.info("Applied image preprocessing: {}".format(self.image_preprocessing))


        # Get question preprocessing.
        self.question_preprocessing = get_value_list_from_dictionary(
            "question_preprocessing", self.config,
            'none | lowercase | remove_punctuation | tokenize | random_remove_stop_words | random_shuffle_words | all'.split(" | ")
            )
        if 'none' in self.question_preprocessing:
            self.question_preprocessing = []
        if 'all' in self.question_preprocessing:
            self.question_preprocessing = 'lowercase | remove_punctuation | tokenize | remove_stop_words | shuffle_words'.split(" | ")
        self.logger.info("Applied question preprocessing: {}".format(self.question_preprocessing))

        # Get answer preprocessing.
        self.answer_preprocessing = get_value_list_from_dictionary(
            "answer_preprocessing", self.config,
            'none | lowercase | remove_punctuation | tokenize | all'.split(" | ")
            )
        if 'none' in self.answer_preprocessing:
            self.answer_preprocessing = []
        if 'all' in self.answer_preprocessing:
            self.answer_preprocessing = 'lowercase | remove_punctuation | tokenize '.split(" | ")
        self.logger.info("Applied answer preprocessing: {}".format(self.answer_preprocessing))


        # Load dataset.
        self.logger.info("Loading dataset from files:\n {}".format(source_files))
        self.dataset = self.load_dataset(source_files, source_image_folders, source_categories)
        self.logger.info("Loaded dataset consisting of {} samples".format(len(self.dataset)))

        # Display exemplary sample.
        self.logger.info("Exemplary sample:\n [ category: {}\t image_ids: {}\t question: {}\t answer: {} ]".format(
            self.dataset[0][self.key_category_ids],
            self.dataset[0][self.key_image_ids],
            self.dataset[0][self.key_questions],
            self.dataset[0][self.key_answers]
            ))


    def filter_sources(self, source_files, source_image_folders, source_categories):
        """
        Loads the dataset from one or more files.

        :param source_files: List of source files.

        :param source_image_folders: List of folders containing image files.

        :param source_categories: List of categories associated with each of those files. (<UNK> unknown)

        :return: Tuple consisting of: filtered source_files and filtered source_categories
        """
        # Check categories that user want to use.
        use_files = [False] * 4
        categs = {'C1': 0, 'C2': 1, 'C3': 2, 'C4': 3}
        # Parse categories from configuration list.
        loaded_categs = get_value_list_from_dictionary("categories", self.config, ['C1', 'C2', 'C3', 'C4', 'all'])
        for cat in loaded_categs:
            # "Special" case.
            if cat == "all":
                use_files = [True] * 4
                # Make no sense to continue.
                break
            else:
                use_files[categs[cat]] = True
        # Filter.
        _, source_files, source_image_folders, source_categories = zip(*(filter(lambda x: x[0], zip(use_files, source_files, source_image_folders, source_categories))))
        return source_files, source_image_folders, source_categories


    def __len__(self):
        """
        Returns the "size" of the "problem" (total number of samples).

        :return: The size of the problem.
        """
        return len(self.dataset)


    def output_data_definitions(self):
        """
        Function returns a dictionary with definitions of output data produced the component.

        :return: dictionary containing output data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        # Add all "standard" streams.
        d = {
            self.key_indices: DataDefinition([-1, 1], [list, int], "Batch of sample indices [BATCH_SIZE] x [1]"),
            self.key_images: DataDefinition([-1, self.depth, self.height, self.width], [torch.Tensor], "Batch of images [BATCH_SIZE x IMAGE_DEPTH x IMAGE_HEIGHT x IMAGE_WIDTH]"),
            self.key_image_ids: DataDefinition([-1, 1], [list, str], "Batch of image names, each being a single word [BATCH_SIZE] x [STRING]"),
            self.key_image_sizes: DataDefinition([-1, 2], [torch.Tensor], "Batch of original sizes (height, width) of images [BATCH_SIZE x 2]"),
            self.key_category_ids: DataDefinition([-1], [torch.Tensor], "Batch of target category indices, each being a single index [BATCH_SIZE]"),
            self.key_category_names: DataDefinition([-1, 1], [list, str], "Batch of category target names, each being a single word [BATCH_SIZE] x [STRING]"),
            }

        # Add stream with questions.
        if 'tokenize' in self.question_preprocessing:
            d[self.key_questions] = DataDefinition([-1, -1, 1], [list, list, str], "Batch of questions, each being a list of words [BATCH_SIZE] x [SEQ_LEN] x [STRING]")
        else:
            d[self.key_questions] = DataDefinition([-1, 1], [list, str], "Batch of questions, each being a string consisting of many words [BATCH_SIZE] x [STRING]")

        # Add stream with answers.
        if 'tokenize' in self.answer_preprocessing:
            d[self.key_answers] = DataDefinition([-1, -1, 1], [list, list, str], "Batch of target answers, each being a list of words [BATCH_SIZE] x [SEQ_LEN] x [STRING]")
        else:
            d[self.key_answers]= DataDefinition([-1, 1], [list, str], "Batch of target answers, each being a string consisting of many words [BATCH_SIZE] x [STRING]")
        return d


    def preprocess_text(self, text, lowercase = False, remove_punctuation = False, tokenize = False, remove_stop_words = False):
        """
        Function that preprocesses questions/answers as suggested by ImageCLEF VQA challenge organizers:
            * lowercases all words (optional)
            * removes punctuation (optional)
            * removes stop words (optional)

        :param text: text to be processed.
        :param lowercase: lowercases text (DEFAULT: False)
        :param remove_punctuation: removes punctuation (DEFAULT: False)
        :param tokenize: tokenizes the text (DEFAULT: False)
        :param remove_stop_words: removes stop words (DEFAULT: False)

        :return: Preprocessed and tokenized text (list of strings)
        """
        # Lowercase.
        if lowercase:
            text = text.lower()

        # Remove punctuation.
        if remove_punctuation:
            # Remove '“' and '”' and '’'!!!
            for char in ['“', '”', '’']:
                text = text.replace(char,' ')
            translator = str.maketrans('', '', string.punctuation)
            text = text.translate(translator)

        # If not tokenize - return text.
        if not tokenize:
            return text
        
        # Tokenize.
        text_words = nltk.tokenize.word_tokenize(text)

        # If we do not want to remove stop words - return text.
        if not remove_stop_words:
            return text_words

        # Perform "cleansing".
        stops = set(nltk.corpus.stopwords.words("english"))
        cleansed_words = [word for word in text_words if word not in stops]
        # Return the original text if there are no words left :]
        if len(cleansed_words) == 0:
            return text_words

        # Return cleaned text.
        return cleansed_words

    def random_remove_stop_words(self, words):
        """
        Function removes random stop words, each with 0.5 probability.
        
        :param words: tokenized text
        :return: resulting tokenized text.
        """

        # Find stop words.
        stops = set(nltk.corpus.stopwords.words("english"))
        stop_words = [False]*len(words)
        for i, word in enumerate(words):
            if word in stops:
                stop_words[i] = True
        #print(stop_words)
        if sum(stop_words) > 0:
            remove_probs = np.random.binomial(1, 0.5, len(words))
            #print(remove_probs)
            result = []
            for word,is_stop,rem_prob in zip(words,stop_words,remove_probs):
                if is_stop and rem_prob:
                    # Remove word.
                    continue
                # Else: add word.
                result.append(word)

        return result


    def random_shuffle_words(self, words):
        """
        Function randomly shuffles, with probability of 0.5, two consecutive words in text.
        
        :param words: tokenized text
        :return: resulting tokenized text.
        """
        # Do not shuffle if there are less than 2 words.
        if len(words) < 2:
            return words
        # Shuffle with probability of 0.5.
        if np.random.binomial(1, 0.5, 1):
            return words
        
        # Find words to shuffle - random without replacement.
        shuffled_i = np.random.choice(len(words)-1, )
        indices = [i for i in range(len(words))]
        indices[shuffled_i] = shuffled_i+1
        indices[shuffled_i+1] = shuffled_i
        #print(indices)
        
        # Create resulting table.
        result = [words[indices[i]] for i in range(len(words))]

        return result


    def load_dataset(self, source_files, source_image_folders, source_categories):
        """
        Loads the dataset from one or more files.

        :param source_files: List of source files.

        :param source_image_folders: List of folders containing image files.

        :param source_categories: List of categories associated with each of those files. (<UNK> unknown)
        """
        # Set containing list of tuples.
        dataset = []

        # Process files with categories.
        for data_file, image_folder, category in zip(source_files, source_image_folders, source_categories):
            # Set absolute path to file.
            self.logger.info('Loading dataset from {} (category: {})...'.format(data_file, category))
            # Load file content using '|' separator.
            df = pd.read_csv(filepath_or_buffer=data_file, sep='|',header=None,
                    names=[self.key_image_ids,self.key_questions,self.key_answers])

            # Add tdqm bar.
            t = tqdm.tqdm(total=len(df.index))
            for _, row in df.iterrows():
                # Retrieve question and answer.
                question = row[self.key_questions]
                answer = row[self.key_answers]

                # Process question - if required.
                preprocessed_question = self.preprocess_text(
                    question,
                    'lowercase' in self.question_preprocessing,
                    'remove_punctuation' in self.question_preprocessing,
                    'tokenize' in self.question_preprocessing,
                    'remove_stop_words' in self.question_preprocessing
                    )

                # Process answer - if required.
                preprocessed_answer = self.preprocess_text(
                    answer,
                    'lowercase' in self.answer_preprocessing,
                    'remove_punctuation' in self.answer_preprocessing,
                    'tokenize' in self.answer_preprocessing,
                    False
                    )

                # Add record to dataset.
                dataset.append({
                    # Image name and path leading to it.
                    self.key_image_ids: row[self.key_image_ids],
                    "image_folder": image_folder,
                    self.key_questions: preprocessed_question,
                    self.key_answers: preprocessed_answer,
                    # Add category.
                    self.key_category_ids: category
                    })

                t.update()
            t.close()

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
        item = self.dataset[index]

        # Load the adequate image.
        img_id = item[self.key_image_ids]
        img_folder = item["image_folder"]
        extension = '.jpg'
        # Load the image.
        img = Image.open(os.path.join(img_folder, img_id + extension))
        # Get its width and height.
        width, height = img.size

        image_transformations_list = []
        # Optional.
        if 'random_affine' in self.image_preprocessing:
            rotate = (-45, 80)
            translate = (0.05, 0.25)
            scale = (0.5, 2)
            image_transformations_list.append(transforms.RandomAffine(rotate, translate, scale))
        if 'random_horizontal_flip' in self.image_preprocessing:
            image_transformations_list.append(transforms.RandomHorizontalFlip())
            
        # Add two obligatory transformations.
        image_transformations_list.append(transforms.Resize([self.height,self.width]))
        image_transformations_list.append(transforms.ToTensor())

        # Optional normalizastion.
        if 'normalize' in self.image_preprocessing:
            # Use normalization that the pretrained models from TorchVision require.
            image_transformations_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        # Resize the image and transform to Torch Tensor.
        transforms_com = transforms.Compose(image_transformations_list)
        # Apply transformations.
        img = transforms_com(img)

        # Create the resulting sample (data dict).
        data_dict = self.create_data_dict(index)

        # Image related variables.
        data_dict[self.key_images] = img
        data_dict[self.key_image_ids] = img_id
        # Scale width and height to range (0,1).
        data_dict[self.key_image_sizes] = torch.FloatTensor([float(height/self.scale_image_height), float(width/self.scale_image_width)])

        # Apply question transformations.
        preprocessed_question = item[self.key_questions]
        if 'tokenize' in self.question_preprocessing:
            # Apply them only if text is tokenized.
            if 'random_remove_stop_words' in self.question_preprocessing:
                preprocessed_question = self.random_remove_stop_words(preprocessed_question)

            if 'random_shuffle_words' in self.question_preprocessing:
                preprocessed_question = self.random_shuffle_words(preprocessed_question)
        # Return question.
        data_dict[self.key_questions] = preprocessed_question

        # Return answer. 
        preprocessed_answer = item[self.key_answers]
        data_dict[self.key_answers] = preprocessed_answer

        # Question category related variables.
        # Check if this is binary question.
        if self.predict_yes_no(item[self.key_answers]):
            data_dict[self.key_category_ids] = 4 # Binary.
            data_dict[self.key_category_names] = self.category_idx_to_word[4]
        else:
            data_dict[self.key_category_ids] = item[self.key_category_ids]
            data_dict[self.key_category_names] = self.category_idx_to_word[item[self.key_category_ids]]

        # Return sample.
        return data_dict

    def predict_yes_no(self, qtext):
        """
        Determines whether this is binary (yes/no) type of question.
        """
        yes_no_starters = ['is','was','are','does']
        if 'tokenize' not in self.question_preprocessing:
            qtext = qtext.split(' ')
        first_token = qtext[0]
        if first_token in yes_no_starters and ('or' not in qtext):
            return True
        return False

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
        data_dict[self.key_images] = torch.stack([item[self.key_images] for item in batch]).type(torch.FloatTensor)
        data_dict[self.key_image_ids] = [item[self.key_image_ids] for item in batch]
        data_dict[self.key_image_sizes] = torch.stack([item[self.key_image_sizes] for item in batch]).type(torch.FloatTensor)

        # Collate lists/lists of lists.
        data_dict[self.key_questions] = [item[self.key_questions] for item in batch]
        data_dict[self.key_answers] = [item[self.key_answers] for item in batch]

        # Stack categories.
        data_dict[self.key_category_ids] = torch.tensor([item[self.key_category_ids] for item in batch])
        data_dict[self.key_category_names] = [item[self.key_category_names] for item in batch]

        # Return collated dict.
        return data_dict
