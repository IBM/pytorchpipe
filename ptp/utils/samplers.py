#!/usr/bin/env python3
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

from math import ceil

import torch
import torch.utils.data.sampler as samplers

class kFoldRandomSampler(samplers.Sampler):
    """
    Samples indices using the k-fold cross validation approach.
    Offers two modes:
        - generate indices for all-but-one folds (for training).
        - generate indices for only one fold (for validation).
    # Every time 
    """

    def __init__(self, dataset_size, num_folds, all_but_current_fold = True):
        """
        Initializes the sampler by generating the indices associated with the fold(s) that are to be used.

        :param dataset_size: Size of the dataset        
        :param num_folds: Number of folds
        :param all_but_current_fold: Operation mode (DEFAULT: True):
            When True, generates indices for all-but-one folds (for training). \
            When False, generates indices for only one fold (for validation). \
        """
        self.dataset_size = dataset_size
        self.num_folds = num_folds
        self.all_but_current_fold = all_but_current_fold
        # Initialize current "fold" as -1, so then dataloder will call next() for the first time 
        # it will return samples for 0-th fold/all-but-0th fold.
        self.current_fold = -1
        # Generate "initial" indices.
        self.indices = []
        self.regenerate_indices()

    def regenerate_indices(self):
        """
        (Re)generates indices used by sampler repending of mode and given fold number.

        :param fold: Fold number
        """
        # Fold size and indices.
        all_indices = range(self.dataset_size)
        fold_size = ceil(self.dataset_size / self.num_folds)

        # Modulo current fold number by total number of folds.
        fold = self.current_fold % self.num_folds

        # Generate indices associated with the given fold / all except the given fold.
        if self.all_but_current_fold:
            if fold == 0:
                first = (fold+1)*fold_size
                # Create indices set.
                self.indices = all_indices[first:]
            else:
                # Concatenate two subsets of indices.
                first_0 = 0
                last_0 = fold*fold_size
                first_1 = (fold+1)*fold_size
                # Assume that the last fold might be "smaller".
                last_1 = min((fold+2)*fold_size,self.dataset_size)
                # Create indices set from two subsets.
                self.indices = [*all_indices[first_0:last_0], *all_indices[first_1:last_1]]
        else:
            # Get first/last indices.
            first = fold*fold_size
            # Assume that the last fold might be "smaller".
            last = min((fold+1)*fold_size, self.dataset_size)
            # Create indices set.
            self.indices = all_indices[first:last]


    def __iter__(self):
        """
        Return "shuffled" indices.
        """
        # Next fold.
        self.current_fold += 1
        # Regenerate indices.
        self.regenerate_indices()
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        """
        Return length of dataset.
        """
        return len(self.indices)
