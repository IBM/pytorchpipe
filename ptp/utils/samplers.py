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
from torch._six import int_classes as _int_classes
from torch.utils.data.sampler import Sampler

class kFoldRandomSampler(Sampler):
    """
    Samples indices using the k-fold cross validation approach.
    Offers two modes:
        - generate indices for all-but-one folds (for training)
        - generate indices for only one fold (for validation)
    
    Every time __iter__() method is called, it moves to next fold/set of folds. 
    """

    def __init__(self, num_samples, num_folds, all_but_current_fold = True):
        """
        Initializes the sampler by generating the indices associated with the fold(s) that are to be used.

        :param num_samples: Size of the dataset        
        :param num_folds: Number of folds
        :param all_but_current_fold: Operation mode (DEFAULT: True):
            When True, generates indices for all-but-one folds (for training). \
            When False, generates indices for only one fold (for validation). \
        """
        # Get number of samples (size of "whole dataset").
        if not isinstance(num_samples, _int_classes) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integeral "
                             "value, but got num_samples={}".format(num_samples))
        self.num_samples = num_samples

        # Get number of folds.
        if not isinstance(num_folds, _int_classes) or isinstance(num_samples, bool) or \
                num_folds <= 0:
            raise ValueError("num_folds should be a positive integeral "
                             "value, but got num_folds={}".format(num_folds))
        self.num_folds = num_folds

        self.all_but_current_fold = all_but_current_fold
        # Initialize current "fold" as -1, so then dataloder will call next() for the first time 
        # it will return samples for 0-th fold/all-but-0th fold.
        self.current_fold = -1
        # Generate "initial" indices.
        self.indices = self.regenerate_indices()

    def regenerate_indices(self):
        """
        (Re)generates indices used by sampler repending of mode and given fold number.

        :return: (re)generated indices.
        """
        # Fold size and indices.
        all_indices = range(self.num_samples)
        fold_size = ceil(self.num_samples / self.num_folds)

        # Modulo current fold number by total number of folds.
        fold = self.current_fold % self.num_folds

        # Generate indices associated with the given fold / all except the given fold.
        if self.all_but_current_fold:
            if fold == 0:
                first = (fold+1)*fold_size
                # Create indices set.
                return all_indices[first:]
            else:
                # Concatenate two subsets of indices.
                first_0 = 0
                last_0 = fold*fold_size
                first_1 = (fold+1)*fold_size
                # Assume that the last fold might be "smaller".
                last_1 = min((fold+2)*fold_size,self.num_samples)
                # Create indices set from two subsets.
                return [*all_indices[first_0:last_0], *all_indices[first_1:last_1]]
        else:
            # Get first/last indices.
            first = fold*fold_size
            # Assume that the last fold might be "smaller".
            last = min((fold+1)*fold_size, self.num_samples)
            # Create indices set.
            return all_indices[first:last]


    def __iter__(self):
        """
        Return "shuffled" indices.
        """
        # Next fold.
        self.current_fold += 1

        # Regenerate indices.
        self.indices = self.regenerate_indices()

        # Return permutated indices.
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        """
        Return length of dataset.
        """
        return len(self.indices)


class kFoldWeightedRandomSampler(kFoldRandomSampler):
    """
    Samples indices using the k-fold cross validation, additionally relying on the provided probabilities (weights).
    Offers two modes:
        - generate indices for all-but-one folds (for training)
        - generate indices for only one fold (for validation)
    
    Every time __iter__() method is called, it moves to next fold/set of folds. 
    """

    def __init__(self, weights, num_samples, num_folds, all_but_current_fold = True, replacement=True):
        """
        Initializes the sampler by generating the indices associated with the fold(s) that are to be used.

        :param num_samples: Size of the dataset        
        :param num_folds: Number of folds
        :param all_but_current_fold: Operation mode (DEFAULT: True):
            When True, generates indices for all-but-one folds (for training). \
            When False, generates indices for only one fold (for validation). \
        :params weights: a sequence of weights, not necessary summing up to one
        :param num_samples: number of samples to draw
        :param replacement: if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.
        """
        # Call k-fold base class constructor.
        super().__init__(num_samples, num_folds, all_but_current_fold)
        # Get replacement flag.
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        self.replacement = replacement

        # Get weights.
        self.weights = torch.tensor(weights, dtype=torch.double)

    def __iter__(self):
        # Next fold.
        self.current_fold += 1

        # Regenerate indices.
        self.indices = self.regenerate_indices()

        # Select the corresponging weights.
        weights = torch.take(self.weights, torch.tensor(self.indices))
        
        # Return indices sampled with multinomial distribution.
        return (self.indices[i] for i in torch.multinomial(weights, len(self.indices), self.replacement).tolist())
