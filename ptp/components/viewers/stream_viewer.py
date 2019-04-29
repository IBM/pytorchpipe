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

import numpy as np

from ptp.components.component import Component
from ptp.data_types.data_definition import DataDefinition


class StreamViewer(Component):
    """
    Utility for displaying contents of streams of a single sample from the batch.

    """

    def __init__(self, name, config):
        """
        Initializes loss object.

        :param name: Loss name.
        :type name: str

        :param config: Dictionary of parameters (read from the configuration ``.yaml`` file).
        :type config: :py:class:`ptp.configuration.ConfigInterface`

        """
        # Call constructors of parent classes.
        Component.__init__(self, name, StreamViewer, config)

        # Get key mappings for indices.
        self.key_indices = self.stream_keys["indices"]

        # Load list of streams names (keys).
        self.input_stream_keys = self.config["input_streams"]
        if type(self.input_stream_keys) == str:
            self.input_stream_keys = self.input_stream_keys.replace(" ", "").split(",")
        
        # Get sample number.
        self.sample_number = self.config["sample_number"]
        

    def input_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of input data that are required by the component.

        :return: dictionary containing input data definitions (each of type :py:class:`ptp.data_types.DataDefinition`).
        """
        return {
            self.key_indices: DataDefinition([-1, 1], [list, int], "Batch of sample indices [BATCH_SIZE] x [1]"),
            }

    def output_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of output data produced the component.

        :return: dictionary containing output data definitions (each of type :py:class:`ptp.data_types.DataDefinition`).
        """
        return {
            }

    def __call__(self, data_dict):
        """
        Encodes batch, or, in fact, only one field of batch ("inputs").
        Stores result in "outputs" field of data_dict.

        :param data_dict: :py:class:`ptp.utils.DataDict` object containing (among others) "indices".

        """
        # Use worker interval.
        if self.app_state.episode % self.app_state.args.logging_interval == 0:

            # Get indices.
            indices = data_dict[self.key_indices]

            # Get sample number.
            if self.sample_number == -1:
                # Random
                sample_number = np.random.randint(0, len(indices))
            else:
                sample_number = self.sample_number

            # Generate displayed string.
            absent_streams = []
            disp_str = "Showing selected streams for sample {}:\n".format(sample_number)
            for stream_key in self.input_stream_keys:
                if stream_key in data_dict.keys():
                    disp_str += " '{}': {}\n".format(stream_key, data_dict[stream_key][sample_number])
                else:
                    absent_streams.append(stream_key)

            # Log values and inform about missing streams.
            self.logger.info(disp_str)
            if len(absent_streams) > 0:
                self.logger.warning("Coud not display the following (absent) streams: {}".format(absent_streams))
