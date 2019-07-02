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

from ptp.components.tasks.task import Task
from ptp.data_types.data_definition import DataDefinition

from ptp.components.utils.io import save_nparray_to_csv_file
from ptp.configuration.config_parsing import get_value_list_from_dictionary, get_value_from_dictionary


class VQAMED2019(Task):
    """
    Task providing data associated with ImageCLEF VQA 2019 challenge.

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
    Test set with answers can be downloaded from a separate repository _repo.

    .. _crowdai: https://www.crowdai.org/challenges/imageclef-2019-vqa-med

    .. _website: https://www.imageclef.org/2019/medical/vqa/

    .._repo: https://github.com/abachaa/VQA-Med-2019
    """
    def __init__(self, name, config):
        """
        Initializes task object. Calls base constructor. Downloads the dataset if not present and loads the adequate files depending on the mode.

        :param name: Name of the component.

        :param class_type: Class type of the component.

        :param config: Dictionary of parameters (read from configuration ``.yaml`` file).
        """
        # Call constructors of parent classes.
        Task.__init__(self, name, VQAMED2019, config)

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

        # Get flag informing whether we want to stream images or not.
        self.stream_images = self.config['stream_images']

        # Get flag indicating whether we want to (pre)aload all images at the start.
        self.preload_images = self.config['preload_images']

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


        # Get the absolute path.
        self.data_folder = os.path.expanduser(self.config['data_folder'])

        # Get split.
        split = get_value_from_dictionary('split', self.config, "training,validation,training_validation,test_answers,test".split(","))

        # Set split-dependent data.
        if split == 'training':
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
            # Load dataset.
            self.dataset = self.load_dataset(source_files, source_image_folders, source_categories)

        elif split == 'validation':
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
            # Load dataset.
            self.dataset = self.load_dataset(source_files, source_image_folders, source_categories)

        elif split == 'training_validation':
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
            # Load dataset.
            self.dataset = self.load_dataset(source_files, source_image_folders, source_categories)

        elif split == 'test_answers':
            # Test set WITH ANSWERS.
            split_folder = os.path.join(self.data_folder, "ImageClef-2019-VQA-Med-Test")
            # Set source file.
            source_file = os.path.join(split_folder,"VQAMed2019_Test_Questions_w_Ref_Answers.txt")
            # Set image folder.
            source_image_folder = os.path.join(split_folder, 'VQAMed2019_Test_Images')
            self.dataset = self.load_testset_with_answers(source_file, source_image_folder)

        else: # "test"
            # Test set WITHOUT ANSWERS.
            split_folder = os.path.join(self.data_folder, "ImageClef-2019-VQA-Med-Test")
            # Set source file.
            source_file = os.path.join(split_folder,"VQAMed2019_Test_Questions.txt")
            # Set image folder.
            source_image_folder = os.path.join(split_folder, 'VQAMed2019_Test_Images')
            self.dataset = self.load_testset_without_answers(source_file, source_image_folder)

        # Ok, now we got the whole dataset (for given "split").
        self.ix = np.arange(len(self.dataset))
        if self.config["import_indices"] != '':
            # Try to load indices from the file.
            self.ix = np.load(os.path.join(self.app_state.log_dir, self.config["import_indices"]))
            self.logger.info("Imported indices from '{}'".format(os.path.join(self.app_state.log_dir, self.config["export_indices"])))
        else:
            # Ok, check whether we want to shuffle.
            if self.config["shuffle_indices"]:
                np.random.shuffle(self.ix)
            # Export if required.
            if self.config["export_indices"] != '':
                # export indices to file.
                np.save(os.path.join(self.app_state.log_dir, self.config["export_indices"]), self.ix)
                self.logger.info("Exported indices to '{}'".format(os.path.join(self.app_state.log_dir, self.config["export_indices"])))

        # Display exemplary sample.
        self.logger.info("Exemplary sample 0 ({}):\n  category: {}\n  image_ids: {}\n  question: {}\n  answer: {}".format(
            self.ix[0],
            self.category_idx_to_word[self.dataset[self.ix[0]][self.key_category_ids]],
            self.dataset[self.ix[0]][self.key_image_ids],
            self.dataset[self.ix[0]][self.key_questions],
            self.dataset[self.ix[0]][self.key_answers]
            ))

        # Check if we want the task to calculate and export the weights.
        self.export_sample_weights = self.config["export_sample_weights"]
        if self.export_sample_weights != '':
            self.calculate_and_export_sample_weights(self.export_sample_weights)


    def output_data_definitions(self):
        """
        Function returns a dictionary with definitions of output data produced the component.

        :return: dictionary containing output data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        # Add all "standard" streams.
        d = {
            self.key_indices: DataDefinition([-1, 1], [list, int], "Batch of sample indices [BATCH_SIZE] x [1]"),
            self.key_image_ids: DataDefinition([-1, 1], [list, str], "Batch of image names, each being a single word [BATCH_SIZE] x [STRING]"),
            self.key_category_ids: DataDefinition([-1], [torch.Tensor], "Batch of target category indices, each being a single index [BATCH_SIZE]"),
            self.key_category_names: DataDefinition([-1, 1], [list, str], "Batch of category target names, each being a single word [BATCH_SIZE] x [STRING]"),
            }
        
        # Return images only when required.
        if self.stream_images:
            d[self.key_images] = DataDefinition([-1, self.depth, self.height, self.width], [torch.Tensor], "Batch of images [BATCH_SIZE x IMAGE_DEPTH x IMAGE_HEIGHT x IMAGE_WIDTH]")
            d[self.key_image_sizes] = DataDefinition([-1, 2], [torch.Tensor], "Batch of original sizes (height, width) of images [BATCH_SIZE x 2]")

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


    def __len__(self):
        """
        Returns the "size" of the "task" (total number of samples).

        :return: The size of the task.
        """
        return len(self.dataset)


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


    def calculate_and_export_sample_weights(self, filename):
        """
        Method calculates and export weights associated with samples by looking at distribution of answers.

        :param filename: Name of the file (optionally with path) that the sample weights will be saved to.
        """
        # 0. Create "answers dataset" object for faster computations.
        answers_dataset = []
        for sample in self.dataset:
            if ('tokenize' in self.answer_preprocessing):
                # Need to create one string.
                answers_dataset.append(' '.join(sample[self.key_answers]))
            else:
                answers_dataset.append(sample[self.key_answers])

        
        # 1. Iterate over all samples in dataset and create "answer" vocabulary.
        answer_to_ix = {}
        for answer in answers_dataset:
            # If new token.
            if answer not in answer_to_ix:
                answer_to_ix[answer] = len(answer_to_ix)
                #print("Adding '{}': {}".format(answer, len(answer_to_ix)-1) )

        # 2. Count the samples having the same answer.
        class_sample_count = [0] * len(answer_to_ix)
        for answer in answers_dataset:
            # Increment the adequate class counter.
            class_sample_count[answer_to_ix[answer]] += 1

        # 3. Calculate the weights.
        weights = np.asarray([1.0 / count if count > 0 else 0.0 for count in class_sample_count], dtype=np.float64)
        # Normalize to 1.
        sum_w = sum(weights)
        weights = weights/sum_w
        #print(weights)

        # 4. Assign weights to samples.
        sample_weights = np.array([weights[answer_to_ix[answer]] for answer in answers_dataset])
        #print(sample_weights)
        #print(len(sample_weights))
        
        # Reorder weights accodring to ix.
        sample_weights_ix = np.array([sample_weights[self.ix[i]] for i in range(len(sample_weights))])
        
        # Process filename.
        (path, name) = os.path.split(filename)
        if path == '':
            # Use default task folder as destination.
            path = self.data_folder
        else:
            path = os.path.expanduser(path)

        # Export "reordered weights" to file.
        save_nparray_to_csv_file(path, name, sample_weights_ix)
        self.logger.info("Generated weights for {} samples and exported them to {}".format(len(sample_weights_ix), os.path.join(path, name)))


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
        self.logger.info("Loading dataset from files:\n {}".format(source_files))
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

                # Create item "dictionary".
                item = {
                    # Image name and path leading to it.
                    self.key_image_ids: row[self.key_image_ids],
                    "image_folder": image_folder,
                    self.key_questions: preprocessed_question,
                    self.key_answers: preprocessed_answer,
                    # Add category.
                    self.key_category_ids: category
                    }

                # Preload image.
                if self.preload_images and self.stream_images:
                    img, img_size = self.get_image(row[self.key_image_ids], image_folder)
                    item[self.key_images] = img
                    item[self.key_image_sizes] = img_size

                # Add item to dataset.
                dataset.append(item)

                t.update()
            t.close()

        self.logger.info("Loaded dataset consisting of {} samples".format(len(dataset)))
        # Return the created list.
        return dataset


    def load_testset_with_answers(self, data_file, image_folder):
        """
        Loads the test set with answers.

        :param data_file: Source file.

        :param image_folder: Folder containing image files.

        """
        # Set containing list of tuples.
        dataset = []
        category_mapping = {'modality': 0, 'plane': 1, 'organ': 2, 'abnormality': 3}

        # Set absolute path to file.
        self.logger.info('Loading test set from {}...'.format(data_file))
        # Load file content using '|' separator.
        df = pd.read_csv(filepath_or_buffer=data_file, sep='|',header=None,
                names=[self.key_image_ids,"category",self.key_questions,self.key_answers])

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

            # Get category id.
            category_id = category_mapping[row["category"]]

            # Create item "dictionary".
            item = {
                # Image name and path leading to it.
                self.key_image_ids: row[self.key_image_ids],
                "image_folder": image_folder,
                self.key_questions: preprocessed_question,
                self.key_answers: preprocessed_answer,
                # Add category.
                self.key_category_ids: category_id
                }

            # Preload image.
            if self.preload_images and self.stream_images:
                img, img_size = self.get_image(row[self.key_image_ids], image_folder)
                item[self.key_images] = img
                item[self.key_image_sizes] = img_size

            # Add item to dataset.
            dataset.append(item)

            t.update()
        t.close()

        self.logger.info("Loaded dataset consisting of {} samples".format(len(dataset)))
        # Return the created list.
        return dataset


    def load_testset_without_answers(self, data_file, image_folder):
        """
        Loads the test set without answers.

        :param data_file: Source file.

        :param image_folder: Folder containing image files.

        """
        # Set containing list of tuples.
        dataset = []
        category_id = 5 # <UNK>
        answer = '<UNK>'

        # Set absolute path to file.
        self.logger.info('Loading test set from {}...'.format(data_file))
        # Load file content using '|' separator.
        df = pd.read_csv(filepath_or_buffer=data_file, sep='|',header=None,
                names=[self.key_image_ids,self.key_questions])

        # Add tdqm bar.
        t = tqdm.tqdm(total=len(df.index))
        for _, row in df.iterrows():
            # Retrieve question.
            question = row[self.key_questions]

            # Process question - if required.
            preprocessed_question = self.preprocess_text(
                question,
                'lowercase' in self.question_preprocessing,
                'remove_punctuation' in self.question_preprocessing,
                'tokenize' in self.question_preprocessing,
                'remove_stop_words' in self.question_preprocessing
                )

            # Process answer - if required.
            if 'tokenize' in self.answer_preprocessing:
                preprocessed_answer = [answer]
            else:
                preprocessed_answer = answer 

            # Create item "dictionary".
            item = {
                # Image name and path leading to it.
                self.key_image_ids: row[self.key_image_ids],
                "image_folder": image_folder,
                self.key_questions: preprocessed_question,
                self.key_answers: preprocessed_answer,
                # Add category.
                self.key_category_ids: category_id
                }

            # Preload image.
            if self.preload_images and self.stream_images:
                img, img_size = self.get_image(row[self.key_image_ids], image_folder)
                item[self.key_images] = img
                item[self.key_image_sizes] = img_size

            # Add item to dataset.
            dataset.append(item)

            t.update()
        t.close()

        self.logger.info("Loaded dataset consisting of {} samples".format(len(dataset)))
        # Return the created list.
        return dataset

    def get_image(self, img_id, img_folder):
        """
        Function loads and returns image along with its size.
        Additionally, it performs all the required transformations.

        :param img_id: Identifier of the images.
        :param img_folder: Path to the image.

        :return: image (Tensor), image size (Tensor, w,h, both scaled to (0,1>)
        """

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

        # Get scaled image size.
        img_size = torch.FloatTensor([float(height/self.scale_image_height), float(width/self.scale_image_width)])

        # Return image and size.
        return img, img_size

    def __getitem__(self, index):
        """
        Getter method to access the dataset and return a single sample.

        :param index: index of the sample to return.
        :type index: int

        :return: DataStreams({'indices', 'images', 'images_ids','questions', 'answers', 'category_ids', 'image_sizes'})
        """
        # Get item.
        item = self.dataset[self.ix[index]]

        # Create the resulting sample (data dict).
        data_streams = self.create_data_streams(index)

        # Load and stream the image ids.
        img_id = item[self.key_image_ids]
        data_streams[self.key_image_ids] = img_id

        # Load the adequate image - only when required.
        if self.stream_images:

            if self.preload_images:
                # Use preloaded values.
                img = item[self.key_images]             
                img_size = item[self.key_image_sizes]             
            else:
                # Load at the very moment.
                img, img_size = self.get_image(img_id, item["image_folder"])

            # Image related variables.
            data_streams[self.key_images] = img

            # Scale width and height to range (0,1).
            data_streams[self.key_image_sizes] = img_size

        # Apply question transformations.
        preprocessed_question = item[self.key_questions]
        if 'tokenize' in self.question_preprocessing:
            # Apply them only if text is tokenized.
            if 'random_remove_stop_words' in self.question_preprocessing:
                preprocessed_question = self.random_remove_stop_words(preprocessed_question)

            if 'random_shuffle_words' in self.question_preprocessing:
                preprocessed_question = self.random_shuffle_words(preprocessed_question)
        # Return question.
        data_streams[self.key_questions] = preprocessed_question

        # Return answer. 
        preprocessed_answer = item[self.key_answers]
        data_streams[self.key_answers] = preprocessed_answer

        # Question category related variables.
        # Check if this is binary question.
        if self.predict_yes_no(item[self.key_questions]):
            data_streams[self.key_category_ids] = 4 # Binary.
            data_streams[self.key_category_names] = self.category_idx_to_word[4]
        else:
            data_streams[self.key_category_ids] = item[self.key_category_ids]
            data_streams[self.key_category_names] = self.category_idx_to_word[item[self.key_category_ids]]

        # Return sample.
        return data_streams

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
        Combines a list of DataStreams (retrieved with :py:func:`__getitem__`) into a batch.

        :param batch: list of individual samples to combine
        :type batch: list

        :return: DataStreams({'indices', 'images', 'images_ids','questions', 'answers', 'category_ids', 'image_sizes'})

        """
        # Collate indices.
        data_streams = self.create_data_streams([sample[self.key_indices] for sample in batch])

        # Stack images.
        data_streams[self.key_image_ids] = [item[self.key_image_ids] for item in batch]
        if self.stream_images:
            data_streams[self.key_images] = torch.stack([item[self.key_images] for item in batch]).type(torch.FloatTensor)
            data_streams[self.key_image_sizes] = torch.stack([item[self.key_image_sizes] for item in batch]).type(torch.FloatTensor)

        # Collate lists/lists of lists.
        data_streams[self.key_questions] = [item[self.key_questions] for item in batch]
        data_streams[self.key_answers] = [item[self.key_answers] for item in batch]

        # Stack categories.
        data_streams[self.key_category_ids] = torch.tensor([item[self.key_category_ids] for item in batch])
        data_streams[self.key_category_names] = [item[self.key_category_names] for item in batch]

        # Return collated dict.
        return data_streams
