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

__author__ = "Tomasz Kornuta"

import os
import numpy as np
import logging

import torch.utils.data.sampler

from ptp.configuration.configuration_error import ConfigurationError

class SamplerFactory(object):
    """
    Class returning sampler depending on the name provided in the \
    list of parameters.
    """

    @staticmethod
    def build(problem, config):
        """
        Static method returning particular sampler, depending on the name \
        provided in the list of parameters & the specified problem class.

        :param problem: Instance of an object derived from the Problem class.
        :type problem: ``problems.Problem``

        :param config: Parameters used to instantiate the sampler.
        :type config: :py:class:`ptp.configuration.ConfigInterface`

        ..note::

            ``config`` should contains the exact (case-sensitive) class name of the sampler to instantiate.


        .. warning::

            ``torch.utils.data.sampler.BatchSampler``, \
            ``torch.utils.data.sampler.DistributedSampler`` are not supported yet.

        .. note::

            ``torch.utils.data.sampler.SubsetRandomSampler`` expects 'indices' to index a subset of the dataset. \
             Currently, the user can specify these indices using one of the following options:

            - Option 1: range.
                >>> indices = range(20)

            - Option 2: range as str.
                >>> range_str = '0, 20'

            - Option 3: list of indices.
                >>> yaml_list = yaml.load('[0, 2, 5, 10]')

            - Option 4: name of the file containing indices.
                >>> filename = "~/data/mnist/training_indices.txt"

        .. note::

            ``torch.utils.data.sampler.WeightedRandomSampler`` expercse additional parameter 'weights'.

        :return: Instance of a given sampler or ``None`` if the section not present or couldn't build the sampler.

        """
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger('SamplerFactory')

        # Check if sampler is required, i.e. 'sampler' section is empty.
        if not config:
            logger.info("The sampler configuration section is not present, using default 'random' sampling")
            return None

        try: 
            # Check presence of the name attribute.
            if 'name' not in config:
                raise ConfigurationError("The sampler configuration section does not contain the key 'name'")

            # Get the class name.
            name = config['name']

            # Verify that the specified class is in the samplers package.
            if name not in dir(torch.utils.data.sampler):
                raise ConfigurationError("Could not find the specified class '{}' in the samplers package".format(name))

            # Get the actual class.
            sampler_class = getattr(torch.utils.data.sampler, name)

            # Ok, proceed.
            logger.info('Loading the {} sampler from {}'.format(name, sampler_class.__module__))

            # Handle "special" case.
            if sampler_class.__name__ == 'SubsetRandomSampler':

                # Check presence of the name attribute.
                if 'indices' not in config:
                    raise ConfigurationError("The sampler configuration section does not contain the key 'indices' "
                                    "required by SubsetRandomSampler.")

                indices = config['indices']

                # Analyze the type.
                if type(indices) == str:
                    # Try to open the file.
                    try:
                        # from expanduser()'s doc: If the expansion fails or if the path does not begin
                        # with a tilde, the path is returned unchanged. -> So operation below should be safe.
                        file = open(os.path.expanduser(indices), "r")
                        # Read the file.
                        indices = file.readline() 
                        file.close()

                    except Exception:
                        # Ok, this is not a file.
                        pass
                    finally:
                        # Try to process it as a string.
                        # Get the digits.
                        digits = indices.split(',')
                        indices = [int(x) for x in digits]
                else:
                    # Assume that type(indices) is a list of ints.
                    digits = indices

                # Finally, we got the list of digits.
                if len(digits) == 2:
                    # Create a range.
                    indices = range(int(digits[0]), int(digits[1]))
                # Else: use them as they are

                # Check if indices are within range.
                if max(indices) >= len(problem):
                    logger.error("SubsetRandomSampler cannot work properly when indices are out of range ({}) "
                                 "considering that there are {} samples in the problem!".format(max(indices),
                                                                                                len(problem)))
                    exit(-1)

                # Create the sampler object.
                sampler = sampler_class(indices)

            elif sampler_class.__name__ == 'WeightedRandomSampler':

                # Check presence of the name attribute.
                if 'weights' not in config:
                    raise ConfigurationError("The sampler configuration section does not contain the key 'weights' "
                                    "required by WeightedRandomSampler.")

                # Load weights from file.
                weights = np.fromfile(os.path.expanduser(config['weights']), dtype=float, count=-1, sep=',')
                # Create sampler class.
                sampler = sampler_class(weights, len(problem), replacement=True)

            elif sampler_class.__name__ in ['BatchSampler', 'DistributedSampler']:
                # Sorry, don't support those. Yet;)
                logger.error("Sampler Factory currently does not support {} sampler. Please pick one of the others "
                             "or use defaults random sampling.".format(sampler_class.__name__))
                exit(-2)
            else:
                # Create "regular" sampler.
                sampler = sampler_class(problem)

            # Return sampler.
            return sampler

        except ConfigurationError as e:
            logger.error(e)
            logger.warning("Using default sampling without sampler.")
            return None
