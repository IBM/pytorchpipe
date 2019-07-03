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

from ptp.components.mixins.io import check_file_existence
from ptp.components.tasks.image_text_to_class.gqa import GQA
from ptp.configuration.config_interface import ConfigInterface


class TestGQA(unittest.TestCase):


    def test_training_0_split(self):
        """
            Tests the training_0 split.

            ..note:
                Test on real data is performed only if adequate json source file is found.
        """
        # Empty config.
        config = ConfigInterface()
        config.add_config_params({"gqa_training_0": {"split": "training_0", "globals": {"image_height": "gqa_image_height", "image_width": "gqa_image_width"}}})
    
        # Check the existence of test set.
        if False: #check_file_existence(path.expanduser('~/data/gqa/questions1.2/train_all_questions'),'train_all_questions_0.json'):

            # Create object.
            task = GQA("gqa_training_0", config["gqa_training_0"])
            
            # Check dataset size.
            self.assertEqual(len(task), 1430536)

            # Get sample.
            sample = task[0]

        else: 
            processed_dataset_content = [ {'sample_ids': '07333408', 'image_ids': '2375429', 'questions': 'What is on the white wall?', 'answers': 'pipe', 'full_answers': 'The pipe is on the wall.'} ]

            # Mock up the load_dataset method.
            with patch( "ptp.components.tasks.image_text_to_class.gqa.GQA.load_dataset", MagicMock( side_effect = [ processed_dataset_content ] )):
                task = GQA("gqa_training_0", config["gqa_training_0"])

            # Mock up the get_image method.
            with patch( "ptp.components.tasks.image_text_to_class.gqa.GQA.get_image", MagicMock( side_effect = [ "0" ] )):
                sample = task[0]

        # Check sample.
        self.assertEqual(sample['indices'], 0)
        self.assertEqual(sample['sample_ids'], '07333408')
        self.assertEqual(sample['image_ids'], '2375429')
        self.assertEqual(sample['questions'], 'What is on the white wall?')
        self.assertEqual(sample['answers'], 'pipe')
        self.assertEqual(sample['full_answers'], 'The pipe is on the wall.')


    def test_validation_split(self):
        """
            Tests the validation split.

            ..note:
                Test on real data is performed only if adequate json source file is found.
        """
        # Empty config.
        config = ConfigInterface()
        config.add_config_params({"gqa_validation": {"split": "validation", "globals": {"image_height": "gqa_image_height", "image_width": "gqa_image_width"}}})
    
        # Check the existence of test set.
        if False: #check_file_existence(path.expanduser('~/data/gqa/questions1.2'),'val_all_questions.json'):

            # Create object.
            task = GQA("gqa_validation", config["gqa_validation"])
            
            # Check dataset size.
            self.assertEqual(len(task), 2011853)

            # Get sample.
            sample = task[0]

        else: 
            processed_dataset_content = [ {'sample_ids': '05451384', 'image_ids': '2382986', 'questions': 'Are there blankets under the brown cat?', 'answers': 'no', 'full_answers': 'No, there is a towel under the cat.'} ]

            # Mock up the load_dataset method.
            with patch( "ptp.components.tasks.image_text_to_class.gqa.GQA.load_dataset", MagicMock( side_effect = [ processed_dataset_content ] )):
                task = GQA("gqa_validation", config["gqa_validation"])

            # Mock up the get_image method.
            with patch( "ptp.components.tasks.image_text_to_class.gqa.GQA.get_image", MagicMock( side_effect = [ "0" ] )):
                sample = task[0]

        # Check sample.
        self.assertEqual(sample['indices'], 0)
        self.assertEqual(sample['sample_ids'], '05451384')
        self.assertEqual(sample['image_ids'], '2382986')
        self.assertEqual(sample['questions'], 'Are there blankets under the brown cat?')
        self.assertEqual(sample['answers'], 'no')
        self.assertEqual(sample['full_answers'], 'No, there is a towel under the cat.')


    def test_test_dev_split(self):
        """
            Tests the test_dev split.

            ..note:
                Test on real data is performed only if adequate json source file is found.
        """
        # Empty config.
        config = ConfigInterface()
        config.add_config_params({"gqa_testdev": {"split": "test_dev", "globals": {"image_height": "gqa_image_height", "image_width": "gqa_image_width"}}})
    
        # Check the existence of test set.
        if False: #check_file_existence(path.expanduser('~/data/gqa/questions1.2'),'testdev_all_questions.json'):

            # Create object.
            task = GQA("gqa_testdev", config["gqa_testdev"])
            
            # Check dataset size.
            self.assertEqual(len(task), 172174)

            # Get sample.
            sample = task[0]

        else: 
            processed_dataset_content = [ {'sample_ids': '20968379', 'image_ids': 'n288870', 'questions': 'Do the shorts have dark color?', 'answers': 'yes', 'full_answers': 'Yes, the shorts are dark.'} ]

            # Mock up the load_dataset method.
            with patch( "ptp.components.tasks.image_text_to_class.gqa.GQA.load_dataset", MagicMock( side_effect = [ processed_dataset_content ] )):
                task = GQA("gqa_testdev", config["gqa_testdev"])

            # Mock up the get_image method.
            with patch( "ptp.components.tasks.image_text_to_class.gqa.GQA.get_image", MagicMock( side_effect = [ "0" ] )):
                sample = task[0]

        # Check sample.
        self.assertEqual(sample['indices'], 0)
        self.assertEqual(sample['sample_ids'], '20968379')
        self.assertEqual(sample['image_ids'], 'n288870')
        self.assertEqual(sample['questions'], 'Do the shorts have dark color?')
        self.assertEqual(sample['answers'], 'yes')
        self.assertEqual(sample['full_answers'], 'Yes, the shorts are dark.')


    def test_test_split(self):
        """
            Tests the test split.

            ..note:
                Test on real data is performed only if adequate json source file is found.
        """
        # Empty config.
        config = ConfigInterface()
        config.add_config_params({"gqa_test": {"split": "test", "globals": {"image_height": "gqa_image_height", "image_width": "gqa_image_width"}}})
    
        # Check the existence of test set.
        if False: #check_file_existence(path.expanduser('~/data/gqa/questions1.2'),'test_all_questions.json'):

            # Create object.
            task = GQA("gqa_test", config["gqa_test"])
            
            # Check dataset size.
            self.assertEqual(len(task), 1340048)

            # Get sample.
            sample = task[0]

        else: 
            processed_dataset_content = [ {'sample_ids': '201971873', 'image_ids': 'n15740', 'questions': 'Is the blanket to the right of a pillow?', 'answers': '<UNK>', 'full_answers': '<UNK>'} ]

            # Mock up the load_dataset method.
            with patch( "ptp.components.tasks.image_text_to_class.gqa.GQA.load_dataset", MagicMock( side_effect = [ processed_dataset_content ] )):
                task = GQA("gqa_test", config["gqa_test"])

            # Mock up the get_image method.
            with patch( "ptp.components.tasks.image_text_to_class.gqa.GQA.get_image", MagicMock( side_effect = [ "0" ] )):
                sample = task[0]
        
        # Check sample.
        self.assertEqual(sample['indices'], 0)
        self.assertEqual(sample['sample_ids'], '201971873')
        self.assertEqual(sample['image_ids'], 'n15740')
        self.assertEqual(sample['questions'], 'Is the blanket to the right of a pillow?')
        self.assertEqual(sample['answers'], '<UNK>')
        self.assertEqual(sample['full_answers'], '<UNK>')
        

#if __name__ == "__main__":
#    unittest.main()