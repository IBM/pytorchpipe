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
from ptp.components.problems.image_text_to_class.gqa import GQA
from ptp.configuration.config_interface import ConfigInterface


class TestGQA(unittest.TestCase):



    def test_test_set(self):
        """
            Tests the test split.

            ..note:
                Test on real data is performed only if adequate json source file is found.
        """
        # Empty config.
        config = ConfigInterface()
        config.add_config_params({"split": "test"})
    
        # Check the existence of test set.
        #if check_file_existence(path.expanduser('~/data/gqa/questions1.2'),'test_all_questions.json'):

            # Create object.
            #problem = GQA("GQA", config)
            
            # Check dataset size.
            #self.assertEqual(len(problem), 1340048)

            # Get sample.
            #sample = problem[0]

        #else: 
        processed_dataset_content = [ {'sample_ids': '201971873', 'image_ids': 'n15740', 'questions': 'Is the blanket to the right of a pillow?', 'answers': '<UNK>', 'full_answers': '<UNK>'} ]

        # Mock up the load_dataset method.
        with patch( "ptp.components.problems.image_text_to_class.gqa.GQA.load_dataset", MagicMock( side_effect = [ processed_dataset_content ] )):
            problem = GQA("GQA", config)

        # Mock up the get_image method.
        with patch( "ptp.components.problems.image_text_to_class.gqa.GQA.get_image", MagicMock( side_effect = [ "0" ] )):
            sample = problem[0]
        
        # Check sample.
        self.assertEqual(sample['indices'], 0)
        self.assertEqual(sample['sample_ids'], '201971873')
        self.assertEqual(sample['image_ids'], 'n15740')
        self.assertEqual(sample['questions'], 'Is the blanket to the right of a pillow?')
        self.assertEqual(sample['answers'], '<UNK>')
        self.assertEqual(sample['full_answers'], '<UNK>')
        

if __name__ == "__main__":
    unittest.main()