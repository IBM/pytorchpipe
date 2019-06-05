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
from os import path

from ptp.components.utils.io import check_file_existence
from ptp.components.problems.image_text_to_class.clevr import CLEVR
from ptp.data_types.data_definition import DataDefinition
from ptp.configuration.config_interface import ConfigInterface


class TestCLEVR(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestCLEVR, self).__init__(*args, **kwargs)

        # Check the existence of training set.
        self.unittest_training_set = False # check_file_existence(path.expanduser('~/data/CLEVR_v1.0/questions'),'CLEVR_train_questions.json')
        # Check the existence of validation set.
        self.unittest_validation_set = check_file_existence(path.expanduser('~/data/CLEVR_v1.0/questions'),'CLEVR_val_questions.json')
        # Check the existence of test set.
        self.unittest_test_set = check_file_existence(path.expanduser('~/data/CLEVR_v1.0/questions'),'CLEVR_test_questions.json')
        

    def test_training_set(self):
        """
            Tests the CLEVR training split.

            ..note:
                Test is performed only if json file '~/data/CLEVR_v1.0/questions/CLEVR_train_questions.json' is found.
        """
        if not self.unittest_training_set:
            return
        # Empty config.
        config = ConfigInterface()
        config.add_config_params({"split": "training"})
        clevr = CLEVR("CLEVR", config)

        # Check dataset size.
        self.assertEqual(len(clevr), 699989)

        # Check sample.
        sample = clevr[0]
        self.assertEqual(sample['indices'], 0)
        self.assertEqual(sample['image_ids'], 'CLEVR_train_000000.png')
        self.assertEqual(sample['question_type_ids'], 4)
        self.assertEqual(sample['question_type_names'], 'greater_than')
        self.assertEqual(sample['questions'], 'Are there more big green things than large purple shiny cubes?')
        self.assertEqual(sample['answers'], 'yes')
        
    
    def test_validation_set(self):
        """
            Tests the CLEVR validation split.

            ..note:
                Test is performed only if json file '~/data/CLEVR_v1.0/questions/CLEVR_val_questions.json' is found.
        """
        if not self.unittest_validation_set:
            return
        # Empty config.
        config = ConfigInterface()
        config.add_config_params({"split": "validation"})
        clevr = CLEVR("CLEVR", config)

        # Check dataset size.
        self.assertEqual(len(clevr), 149991)

        # Check sample.
        sample = clevr[0]
        self.assertEqual(sample['indices'], 0)
        self.assertEqual(sample['image_ids'], 'CLEVR_val_000000.png')
        self.assertEqual(sample['question_type_ids'], 10)
        self.assertEqual(sample['question_type_names'], 'exist')
        self.assertEqual(sample['questions'], 'Are there any other things that are the same shape as the big metallic object?')
        self.assertEqual(sample['answers'], 'no')
        

    def test_test_set(self):
        """
            Tests the CLEVR test split.

            ..note:
                Test is performed only if json file '~/data/CLEVR_v1.0/questions/CLEVR_test_questions.json' is found.
        """
        if not self.unittest_test_set:
            return
        # Empty config.
        config = ConfigInterface()
        config.add_config_params({"split": "test"})
        clevr = CLEVR("CLEVR", config)

        # Check dataset size.
        self.assertEqual(len(clevr), 149988)

        # Check sample.
        sample = clevr[0]
        self.assertEqual(sample['indices'], 0)
        self.assertEqual(sample['image_ids'], 'CLEVR_test_000000.png')
        self.assertEqual(sample['question_type_ids'], -1)
        self.assertEqual(sample['question_type_names'], '<UNK>')
        self.assertEqual(sample['questions'], 'Is there anything else that is the same shape as the small brown matte object?')
        self.assertEqual(sample['answers'], '<UNK>')
        



#if __name__ == "__main__":
#    unittest.main()