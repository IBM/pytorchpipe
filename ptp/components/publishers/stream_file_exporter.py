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

from os import path

from ptp.configuration.config_parsing import get_value_list_from_dictionary
from ptp.components.component import Component
from ptp.data_types.data_definition import DataDefinition


class StreamFileExporter(Component):
    """
    Utility for exporting contents of streams of a given batch to file.
    """

    def __init__(self, name, config):
        """
        Initializes the object, retrieves names of input streams and creates the output file in experiment directory.

        :param name: Name of the component.
        :type name: str

        :param config: Dictionary of parameters (read from the configuration ``.yaml`` file).
        :type config: :py:class:`ptp.configuration.ConfigInterface`

        """
        # Call constructors of parent classes.
        Component.__init__(self, name, StreamFileExporter, config)

        # Get key mappings for indices.
        self.key_indices = self.stream_keys["indices"]

        # Load list of streams names (keys).
        self.input_stream_keys = get_value_list_from_dictionary("input_streams", self.config)
        
        # Get separator.
        self.separator = self.config["separator"]

        # Create file where we will write the results.
        filename = self.config["filename"]
        abs_filename = path.join(self.app_state.log_dir, filename)
        self.file = open(abs_filename, 'w')

        # Export additional line.
        if self.config["export_separator_line_to_csv"]:
            self.file.write("sep={}\n".format(self.separator))

        self.logger.info("Writing values from {} streams to {}".format(self.input_stream_keys, abs_filename))


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
        Exports values from the indicated streams to file.
        :param data_dict: :py:class:`ptp.utils.DataDict` object containing "indices" and other streams that will be exported to file.
        """
        # Get batch size.
        indices = data_dict[self.key_indices]
        batch_size = len(indices)

        # Check present streams.
        absent_streams = []
        present_streams = []
        for stream_key in self.input_stream_keys:
            if stream_key in data_dict.keys():
                present_streams.append(stream_key)
            else:
                absent_streams.append(stream_key)

        # Export values to file.
        for i in range(batch_size):
            val_str = ''
            for stream_key in self.input_stream_keys:
                if stream_key in present_streams:
                    value = data_dict[stream_key][i]
                    # Add value changed to string along with separator.
                    val_str = val_str + '{}'.format(value) + self.separator
            # Remove the last separator.
            val_str = val_str[:-1] + '\n'
            # Write it to file.
            self.file.write(val_str)

        # Log values and inform about missing streams.
        if len(absent_streams) > 0:
            self.logger.warning("Could not export the following (absent) streams: {}".format(absent_streams))
