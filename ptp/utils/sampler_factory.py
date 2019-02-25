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

"""
sampler_factory.py: Factory building samplers used by PyTorch's DataLoaders.

"""
__author__ = "Tomasz Kornuta"

import os
import logging
import torch.utils.data.sampler


class SamplerFactory(object):
    """
    Class returning sampler depending on the name provided in the \
    list of parameters.
    """

    @staticmethod
    def build(problem, params):
        """
        Static method returning particular sampler, depending on the name \
        provided in the list of parameters & the specified problem class.

        :param problem: Instance of an object derived from the Problem class.
        :type problem: ``problems.Problem``

        :param params: Parameters used to instantiate the sampler.
        :type params: ``utils.param_interface.ParamInterface``

        ..note::

            ``params`` should contains the exact (case-sensitive) class name of the sampler to instantiate.


        .. warning::

            ``torch.utils.data.sampler.WeightedRandomSampler``, ``torch.utils.data.sampler.BatchSampler``, \
            ``torch.utils.data.sampler.DistributedSampler`` are not yet supported.

        .. note::

            ``torch.utils.data.sampler.SubsetRandomSampler`` expects indices to index a subset of the dataset. \
             Currently, the user can specify these indices using one of the following options:

            - Option 1: range.
                >>> indices = range(20)

            - Option 2: range as str.
                >>> range_str = '0, 20'

            - Option 3: list of indices.
                >>> yaml_list = yaml.load('[0, 2, 5, 10]')

            - Option 4: name of the file containing indices.
                >>> filename = "~/data/mnist/training_indices.txt"


        :return: Instance of a given sampler or ``None`` if the section not present or couldn't build the sampler.

        """
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger('SamplerFactory')

        # Check if sampler is required, i.e. 'sampler' section is empty.
        if not params:
            logger.info('The sampler configuration section is not present.')
            return None

        try: 
            # Check presence of the name attribute.
            if 'name' not in params:
                raise Exception("The sampler configuration section does not contain the key 'name'.")

            # Get the class name.
            name = params['name']

            # Verify that the specified class is in the samplers package.
            if name not in dir(torch.utils.data.sampler):
                raise Exception("Could not find the specified class '{}' in the samplers package".format(name))

            # Get the actual class.
            sampler_class = getattr(torch.utils.data.sampler, name)

            # Ok, proceed.
            logger.info('Loading the {} sampler from {}'.format(name, sampler_class.__module__))

            # Handle "special" case.
            if sampler_class.__name__ == 'SubsetRandomSampler':

                # Check presence of the name attribute.
                if 'indices' not in params:
                    raise Exception("The sampler configuration section does not contain the key 'indices' "
                                    "required by SubsetRandomSampler.")

                indices = params['indices']

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

            elif sampler_class.__name__ in ['WeightedRandomSampler', 'BatchSampler', 'DistributedSampler']:
                # Sorry, don't support those. Yet;)
                logger.error("Sampler Factory currently does not support {} sampler. Please pick one of the others "
                             "or use defaults random sampling.".format(sampler_class.__name__))
                exit(-2)
            else:
                # Create "regular" sampler.
                sampler = sampler_class(problem)

            # Return sampler.
            return sampler

        except Exception as e:
            logger.error(e)
            logger.warning("Using default sampling without sampler.")
            return None


if __name__ == "__main__":
    """
    Tests the factory.
    """
    from miprometheus.utils.param_interface import ParamInterface
    import yaml

    # Problem.
    class TestProblem(object):
        def __len__(self):
            return 50

    # All samplers operate on TestProblem only,
    # whereas SubsetRandomSampler additionally accepts 'indices' with the following options:
    # Option 1: range.
    indices = range(20)
    # Option 2: range as str.
    range_str = '0, 20'
    # Option 3: list of indices.
    yaml_list = yaml.load('[0, 2, 5, 10]')
    # Option 4: name of the file containing indices.
    filename = "~/data/mnist/training_indices.txt"

    params = ParamInterface()
    params.add_default_params({'name': 'SubsetRandomSampler',
                               'indices': yaml_list})

    sampler = SamplerFactory.build(TestProblem(), params)
    print(type(sampler))

    for i, index in enumerate(sampler):
        print('{}: index {}'.format(i, index))
