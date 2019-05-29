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

__author__ = "Tomasz Kornuta & Vincent Marois"

from collections.abc import Mapping


class StatisticsCollector(Mapping):
    """
    Specialized class used for the collection and export of statistics during\
     training, validation and testing.

    Inherits :py:class:`collections.Mapping`, therefore it offers functionality close to a ``dict``.

    """

    def __init__(self):
        """
        Initialization - creates dictionaries for statistics and formatting.
        """
        super(StatisticsCollector, self).__init__()

        # Set default "output streams" for none.
        self.tb_writer = None
        self.csv_file = None

        self.statistics = dict()
        self.formatting = dict()

    def add_statistics(self, key, formatting):
        """
        Add a statistics to collector.
        The value of associated to the key is of type ``list``.

        :param key: Key of the statistics.
        :type key: str

        :param formatting: Formatting that will be used when logging and exporting to CSV.

        """
        self.formatting[key] = formatting

        # instantiate associated value as a list.
        self.statistics[key] = list()

    def __getitem__(self, key):
        """
        Get statistics value for given key.

        :param key: Key to value in parameters.
        :type key: str

        :return: Statistics value list associated with given key.

        """
        return self.statistics[key]

    def __setitem__(self, key, value):
        """
        Add value to the list of the statistic associated with a given key.

        :param key: Key to value in parameters.
        :param value: Statistics value to append to the list associated with given key.

        """
        self.statistics[key].append(value)

    def __delitem__(self, key):
        """
        Delete the specified key.

        :param key: Key to be deleted.

        """
        del self.statistics[key]

    def __len__(self):
        """
        Returns "length" of ``self.statistics`` (i.e. number of tracked values).
        """
        return self.statistics.__len__()

    def __iter__(self):
        """
        Iterator.
        """
        return self.statistics.__iter__()

    def __eq__(self, other):
        """
        Check whether two collectors are equal (just for the purpose of compatibility with the base Mapping class).
        """
        if isinstance(other, self.__class__):
            # Check statistics and formatting.
            return self.statistics == other.statistics and self.formatting == other.formatting
        else:
            return False

    def empty(self):
        """
        Empty the list associated to the keys of the current statistics collector.

        """
        for key in self.statistics.keys():
            del self.statistics[key][:]


    def base_initialize_csv_file(self, log_dir, filename, keys):
        """
        This method creates a new `csv` file and initializes it with a header produced \
        on the base of the statistical aggregators names.

        :param log_dir: Path to file.
        :type log_dir: str

        :param filename: Filename to be created.
        :type filename: str

        :param keys: Names of keys that will be used as header of columns in csv file.

        :return: File stream opened for writing.

        """
        header_str = ''

        # Iterate through keys and concatenate them.
        for key in keys:
            # If formatting is set to '' - ignore this key.
            if self.formatting.get(key) is not None:
                header_str += key + ","

        # Remove last coma.
        if len(header_str) > 0:
            header_str = header_str[:-1]
        #  Add \n.
        header_str = header_str + '\n'

        # Open file for writing.
        self.csv_file = open(log_dir + filename, 'w', 1)
        self.csv_file.write(header_str)

        return self.csv_file        


    def initialize_csv_file(self, log_dir, filename):
        """
        This method creates a new `csv` file and initializes it with a header produced \
        on the base of the statistical aggregators names.

        :param log_dir: Path to file.
        :type log_dir: str

        :param filename: Filename to be created.
        :type filename: str

        :return: File stream opened for writing.

        """
        return self.base_initialize_csv_file(log_dir, filename, self.statistics.keys())


    def export_to_csv(self, csv_file=None):
        """
        Method writes current statistics to csv using the possessed formatting.

        :param csv_file: File stream opened for writing, optional

        """
        # Try to use the remembered one.    
        if csv_file is None:
            csv_file = self.csv_file
        # If it is still None - well, we cannot do anything more.
        if csv_file is None:
            return

        # Iterate through values and concatenate them.
        values_str = ''
        for key, value in self.statistics.items():
            # If formatting is set to None - ignore this key.
            if self.formatting.get(key) is not None:
                # Get formatting - using '{}' as default.
                format_str = self.formatting.get(key, '{}')

                # Add value to string using formatting.
                if len(value) > 0:
                    values_str += format_str.format(value[-1])
                values_str += ","

        # Remove last coma.
        if len(values_str) > 1:
            values_str = values_str[:-1]
        # Add last \n.
        values_str = values_str + '\n'

        csv_file.write(values_str)

    def export_to_checkpoint(self):
        """
        This method exports the collected data into a dictionary using the associated formatting.

        """
        chkpt = {}

        # Iterate through key, values and format them.
        for key, value in self.statistics.items():
            # If formatting is set to None - ignore this key.
            if self.formatting.get(key) is not None:
                # Get formatting - using '{}' as default.
                format_str = self.formatting.get(key, '{}')

                # Add to dict.
                if len(value) > 0:
                    chkpt[key]  = format_str.format(value[-1])

        return chkpt

    def export_to_string(self, additional_tag=''):
        """
        Method returns current statistics in the form of string using the
        possessed formatting.

        :param additional_tag: An additional tag to append at the end of the created string.
        :type additional_tag: str


        :return: String being the concatenation of the statistics names & values.

        """
        # Iterate through keys and values and concatenate them.
        stat_str = ''
        for key, value in self.statistics.items():
            # If formatting is set to None - ignore this key.
            if self.formatting.get(key) is not None:
                stat_str += key + ' '
                # Get formatting - using '{}' as default.
                format_str = self.formatting.get(key, '{}')
                # Add value to string using formatting.
                if len(value) > 0:
                    stat_str += format_str.format(value[-1])
                stat_str += "; "

        # Remove last two elements.
        if len(stat_str) > 2:
            stat_str = stat_str[:-2]
        
        # Add addtional tag.
        stat_str = stat_str + " " + additional_tag

        return stat_str

    def initialize_tensorboard(self, tb_writer):
        """ 
        Memorizes the writer that will be used with this collector.
        """ 
        self.tb_writer = tb_writer

    def export_to_tensorboard(self, tb_writer=None):
        """
        Method exports current statistics to tensorboard.

        :param tb_writer: TensorBoard writer, optional.
        :type tb_writer: :py:class:`tensorboardX.SummaryWriter`

        """
        # Get episode number.
        episode = self.statistics['episode'][-1]

        if tb_writer is None:
            tb_writer = self.tb_writer
        # If it is still None - well, we cannot do anything more.
        if tb_writer is None:
            return

        # Iterate through keys and values and concatenate them.
        for key, value in self.statistics.items():
            # Skip episode.
            if key == 'episode':
                continue
            # If formatting is set to None - ignore this key.
            if self.formatting.get(key) is not None:
                tb_writer.add_scalar(key, value[-1], episode)
