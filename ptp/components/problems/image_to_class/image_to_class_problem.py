#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) IBM Corporation 2018
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

__author__ = "Tomasz Kornuta, Younes Bouhadjar, Vincent Marois"

import numpy as np

from ptp.components.problems.problem import Problem

class ImageToClassProblem(Problem):
    """
    Abstract base class for image classification problems.

    Problem classes like MNIST & CIFAR10 inherits from it.

    Provides some basic features useful in all problems of such type.

    """

    def __init__(self, name, class_type, params):
        """
        Initializes problem.

        :param name: Problem name.
        :type name: str

        :param class_type: Class type of the component.

        :param params: Dictionary of parameters (read from the configuration ``.yaml`` file).
        :type params: :py:class:`ptp.utils.ParamInterface`
        """
        # Call base class constructors.
        super(ImageToClassProblem, self).__init__(name, class_type, params)

        # Set default key mappings.
        self.key_inputs = self.mapkey("inputs")
        self.key_targets = self.mapkey("targets")


    def show_sample(self, data_dict, sample_number=0):
        """
        Shows a sample from the batch.

        :param data_dict: ``DataDict`` containing inputs and targets.
        :type data_dict: DataDict

        :param sample_number: Number of sample in batch (default: 0)
        :type sample_number: int

        """
        import matplotlib.pyplot as plt

        # Unpack dict.
        images, targets, labels = data_dict.values()

        # Get sample.
        image = images[sample_number].cpu().detach().numpy()
        target = targets[sample_number].cpu().detach().numpy()
        label = labels[sample_number]

        # Reshape image.
        if image.shape[0] == 1:
            # This is a single channel image - get rid of this dimension
            image = np.squeeze(image, axis=0)
        else:
            # More channels - move channels to axis2, according to matplotilb documentation.
            # (X : array_like, shape (n, m) or (n, m, 3) or (n, m, 4))
            image = image.transpose(1, 2, 0)

        # Show data.
        plt.title('Target class: {} ({})'.format(label, target))
        plt.imshow(image, interpolation='nearest', aspect='auto')

        # Plot!
        plt.show()
