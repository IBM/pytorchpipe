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

import os
import numpy as np

import torch.utils.data.sampler as pt_samplers
import ptp.utils.samplers as ptp_samplers 

import ptp.utils.logger as logging
from ptp.configuration.configuration_error import ConfigurationError


class SamplerFactory(object):
    """
    Class returning sampler depending on the name provided in the \
    list of parameters.
    """

    @staticmethod
    def build(task, config, task_subset_name):
        """
        Static method returning particular sampler, depending on the name \
        provided in the list of parameters & the specified task class.

        :param task: Instance of an object derived from the Task class.
        :type task: ``tasks.Task``

        :param config: Parameters used to instantiate the sampler.
        :type config: :py:class:`ptp.configuration.ConfigInterface`

        :param task_subset_name: Name of task subset (and associated TaskManager object)

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
        # Initialize logger.
        logger = logging.initialize_logger('SamplerFactory')

        try: 
            # Check presence of the typename attribute.
            if 'type' not in config:
                raise ConfigurationError("The sampler configuration section does not contain the key 'type'")

            # Get the class typename.
            typename = config['type']
            logger.info('Trying to instantiate the {} sampler object'.format(typename))

            ###########################################################################
            # Handle first special case: SubsetRandomSampler.
            if typename == 'SubsetRandomSampler':

                # Check presence of the typename attribute.
                if 'indices' not in config:
                    raise ConfigurationError("The sampler configuration section does not contain the key 'indices' "
                                    "required by SubsetRandomSampler")

                # Get and process the indices.
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
                # Else: use them as they are, including single index.

                # Check if indices are within range.
                if max(indices) >= len(task):
                    raise ConfigurationError("SubsetRandomSampler cannot work properly when indices are out of range ({}) "
                        "considering that there are {} samples in the task".format(
                            max(indices), len(task)))

                # Create the sampler object.
                sampler = pt_samplers.SubsetRandomSampler(indices)

            ###########################################################################
            # Handle second special case: WeightedRandomSampler.
            elif typename == 'WeightedRandomSampler':

                # Check presence of the attribute.
                if 'weights' not in config:
                    raise ConfigurationError("The sampler configuration section does not contain the key 'weights' "
                                    "required by WeightedRandomSampler")

                # Load weights from file.
                weights = np.fromfile(os.path.expanduser(config['weights']), dtype=float, count=-1, sep=',')

                # Create sampler class.
                sampler = pt_samplers.WeightedRandomSampler(weights, len(task), replacement=True)

            ###########################################################################
            # Handle third special case: kFoldRandomSampler.
            elif typename == 'kFoldRandomSampler':

                # Check presence of the attribute.
                if 'folds' not in config:
                    raise ConfigurationError("The sampler configuration section does not contain the key 'folds' "
                                    "required by kFoldRandomSampler")

                # Create indices, depending on the fold.
                folds = config["folds"]
                if folds < 2:
                    raise ConfigurationError("kFoldRandomSampler requires  at least two 'folds'")
                # Get epochs per fold (default: 1).
                epochs_per_fold = config.get("epochs_per_fold", 1)

                # Create the sampler object.
                sampler = ptp_samplers.kFoldRandomSampler(len(task), folds, epochs_per_fold, task_subset_name == 'training')

            ###########################################################################
            # Handle fourd special case: kFoldWeightedRandomSampler.
            elif typename == 'kFoldWeightedRandomSampler':

                # Check presence of the attribute.
                if 'weights' not in config:
                    raise ConfigurationError("The sampler configuration section does not contain the key 'weights' "
                                    "required by kFoldWeightedRandomSampler")

                # Load weights from file.
                weights = np.fromfile(os.path.expanduser(config['weights']), dtype=float, count=-1, sep=',')

                # Check presence of the attribute.
                if 'folds' not in config:
                    raise ConfigurationError("The sampler configuration section does not contain the key 'folds' "
                                    "required by kFoldWeightedRandomSampler")

                # Create indices, depending on the fold.
                folds = config["folds"]
                if folds < 2:
                    raise ConfigurationError("kFoldRandomSampler requires  at least two 'folds'")
                # Get epochs per fold (default: 1).
                epochs_per_fold = config.get("epochs_per_fold", 1)

                # Create the sampler object.
                sampler = ptp_samplers.kFoldWeightedRandomSampler(weights, len(task), folds, epochs_per_fold, task_subset_name == 'training')

            elif typename in ['BatchSampler', 'DistributedSampler']:
                # Sorry, don't support those. Yet;)
                raise ConfigurationError("Sampler Factory currently does not support the '{}' sampler. Please pick one of the others "
                             "or use defaults random sampling".format(typename))
            else:
                # Verify that the specified class is in the samplers package.
                if typename not in dir(pt_samplers):
                    raise ConfigurationError("Could not find the specified class '{}' in the samplers package".format(typename))

                # Get the sampler class.
                sampler_class = getattr(pt_samplers, typename)
                # Create "regular" sampler.
                sampler = sampler_class(task)

            # Return sampler.
            return sampler

        except ConfigurationError as e:
            logger.error(e)
            # Do not continue with invalid sampler.
            exit(-1)
