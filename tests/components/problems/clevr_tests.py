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

import unittest
from unittest.mock import MagicMock, patch
from os import path

from ptp.components.utils.io import check_file_existence
from ptp.components.problems.image_text_to_class.clevr import CLEVR
from ptp.configuration.config_interface import ConfigInterface


class TestCLEVR(unittest.TestCase):

    def test_training_set(self):
        """
            Tests the training split.

            ..note:
                Test on real data is performed only if json file '~/data/CLEVR_v1.0/questions/CLEVR_train_questions.json' is found.
        """
        # Empty config.
        config = ConfigInterface()
        config.add_config_params({"split": "training"})

        # Check the existence of test set.
        if check_file_existence(path.expanduser('~/data/CLEVR_v1.0/questions'),'CLEVR_train_questions.json'):

            # Create object.
            clevr = CLEVR("CLEVR", config)
            
            # Check dataset size.
            self.assertEqual(len(clevr), 699989)

            # Get sample.
            sample = clevr[0]

        else: 
            dataset_content = [{'image_index': 0, 'program': [{'inputs': [], 'function': 'scene', 'value_inputs': []}, {'inputs': [0], 'function': 'filter_size', 'value_inputs': ['large']}, 
            {'inputs': [1], 'function': 'filter_color', 'value_inputs': ['green']}, {'inputs': [2], 'function': 'count', 'value_inputs': []}, 
            {'inputs': [], 'function': 'scene', 'value_inputs': []}, {'inputs': [4], 'function': 'filter_size', 'value_inputs': ['large']}, 
            {'inputs': [5], 'function': 'filter_color', 'value_inputs': ['purple']}, {'inputs': [6], 'function': 'filter_material', 'value_inputs': ['metal']}, 
            {'inputs': [7], 'function': 'filter_shape', 'value_inputs': ['cube']}, {'inputs': [8], 'function': 'count', 'value_inputs': []}, 
            {'inputs': [3, 9], 'function': 'greater_than', 'value_inputs': []}], 'question_index': 0, 'image_filename': 'CLEVR_train_000000.png', 'question_family_index': 2,
            'split': 'train', 'answer': 'yes', 'question': 'Are there more big green things than large purple shiny cubes?'}]

            # Mock up the load_dataset method.
            with patch( "ptp.components.problems.image_text_to_class.clevr.CLEVR.load_dataset", MagicMock( side_effect = [ dataset_content ] )):
                clevr = CLEVR("CLEVR", config)

            # Mock up the get_image method.
            with patch( "ptp.components.problems.image_text_to_class.clevr.CLEVR.get_image", MagicMock( side_effect = [ "0" ] )):
                sample = clevr[0]

        # Check sample.
        self.assertEqual(sample['indices'], 0)
        self.assertEqual(sample['image_ids'], 'CLEVR_train_000000.png')
        self.assertEqual(sample['question_type_ids'], 4)
        self.assertEqual(sample['question_type_names'], 'greater_than')
        self.assertEqual(sample['questions'], 'Are there more big green things than large purple shiny cubes?')
        self.assertEqual(sample['answers'], 'yes')
        
    
    def test_validation_set(self):
        """
            Tests the validation split.

            ..note:
                Test on real data is performed only if json file '~/data/CLEVR_v1.0/questions/CLEVR_val_questions.json' is found.
        """
        # Empty config.
        config = ConfigInterface()
        config.add_config_params({"split": "validation"})

        # Check the existence of test set.
        if check_file_existence(path.expanduser('~/data/CLEVR_v1.0/questions'),'CLEVR_test_questions.json'):

            # Create object.
            clevr = CLEVR("CLEVR", config)
            
            # Check dataset size.
            self.assertEqual(len(clevr), 149991)

            # Get sample.
            sample = clevr[0]

        else: 
            dataset_content = [{'image_index': 0, 'program': [{'inputs': [], 'function': 'scene', 'value_inputs': []}, {'inputs': [0], 'function': 'filter_size', 'value_inputs': ['large']}, 
                {'inputs': [1], 'function': 'filter_material', 'value_inputs': ['metal']}, {'inputs': [2], 'function': 'unique', 'value_inputs': []}, 
                {'inputs': [3], 'function': 'same_shape', 'value_inputs': []}, {'inputs': [4], 'function': 'exist', 'value_inputs': []}], 
                'question_index': 0, 'image_filename': 'CLEVR_val_000000.png', 'question_family_index': 39, 'split': 'val', 'answer': 'no', 'question': 'Are there any other things that are the same shape as the big metallic object?'}]

            # Mock up the load_dataset method.
            with patch( "ptp.components.problems.image_text_to_class.clevr.CLEVR.load_dataset", MagicMock( side_effect = [ dataset_content ] )):
                clevr = CLEVR("CLEVR", config)

            # Mock up the get_image method.
            with patch( "ptp.components.problems.image_text_to_class.clevr.CLEVR.get_image", MagicMock( side_effect = [ "0" ] )):
                sample = clevr[0]

        # Check sample.
        self.assertEqual(sample['indices'], 0)
        self.assertEqual(sample['image_ids'], 'CLEVR_val_000000.png')
        self.assertEqual(sample['question_type_ids'], 10)
        self.assertEqual(sample['question_type_names'], 'exist')
        self.assertEqual(sample['questions'], 'Are there any other things that are the same shape as the big metallic object?')
        self.assertEqual(sample['answers'], 'no')
        

    def test_test_set(self):
        """
            Tests the test split.

            ..note:
                Test on real data is performed only if json file '~/data/CLEVR_v1.0/questions/CLEVR_test_questions.json' is found.
        """
        # Empty config.
        config = ConfigInterface()
        config.add_config_params({"split": "test"})
    
        # Check the existence of test set.
        if check_file_existence(path.expanduser('~/data/CLEVR_v1.0/questions'),'CLEVR_test_questions.json'):

            # Create object.
            clevr = CLEVR("CLEVR", config)
            
            # Check dataset size.
            self.assertEqual(len(clevr), 149988)

            # Get sample.
            sample = clevr[0]

        else: 
            dataset_content = [{'image_index': 0, 'split': 'test', 'image_filename': 'CLEVR_test_000000.png', 'question_index': 0, 'question': 'Is there anything else that is the same shape as the small brown matte object?'}]

            # Mock up the load_dataset method.
            with patch( "ptp.components.problems.image_text_to_class.clevr.CLEVR.load_dataset", MagicMock( side_effect = [ dataset_content ] )):
                clevr = CLEVR("CLEVR", config)

            # Mock up the get_image method.
            with patch( "ptp.components.problems.image_text_to_class.clevr.CLEVR.get_image", MagicMock( side_effect = [ "0" ] )):
                sample = clevr[0]

        # Check sample.
        self.assertEqual(sample['indices'], 0)
        self.assertEqual(sample['image_ids'], 'CLEVR_test_000000.png')
        self.assertEqual(sample['question_type_ids'], -1)
        self.assertEqual(sample['question_type_names'], '<UNK>')
        self.assertEqual(sample['questions'], 'Is there anything else that is the same shape as the small brown matte object?')
        self.assertEqual(sample['answers'], '<UNK>')
        

#if __name__ == "__main__":
#    unittest.main()