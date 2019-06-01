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


from ptp.components.problems.problem import Problem

class ImageToClassProblem(Problem):
    """
    Abstract base class for image classification problems.

    Problem classes like MNIST & CIFAR10 inherits from it.

    Provides some basic features useful in all problems of such type.

    """

    def __init__(self, name, class_type, config):
        """
        Initializes problem.

        :param name: Problem name.
        :type name: str

        :param class_type: Class type of the component.

        :param config: Dictionary of parameters (read from the configuration ``.yaml`` file).
        :type config: :py:class:`ptp.configuration.ConfigInterface`
        """
        # Call base class constructors.
        super(ImageToClassProblem, self).__init__(name, class_type, config)

