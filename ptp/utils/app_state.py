# -*- coding: utf-8 -*-
#
# Copyright (C) IBM Corporation 2019
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


__author__ = "Alexis Asseman, Tomasz Kornuta"

import torch

from os import path

from ptp.utils.singleton import SingletonMetaClass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import time

# Parameters and DataLoaders
input_size = 5
output_size = 2

batch_size = 30
data_size = 100
from ptp.data_types.data_dict import DataDict



class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):

        # Return data_dict.
        data_dict = DataDict({"index": None})
        data_dict["index"] = index
        return data_dict

        #return self.data[index]

    def __len__(self):
        return self.len

    def collate_fn(self, batch):
        print("Collate!")
        return DataDict({key: torch.utils.data.dataloader.default_collate([sample[key] for sample in batch]) for key in batch[0]})


class Model(nn.Module):
    # Our model
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, datadict):
        input = datadict["index"]
        print("Dummy Model: input size {}, device: {}\n".format(input.size(), input.device))
        output = self.fc(input)
        print("Dummy Model: output size {}\n".format(output.size()))

        return output



class AppState(metaclass=SingletonMetaClass):
    """
    Represents the application state. A singleton that can be accessed by calling:

        >>> app_state = AppState()

    Contains global variables that can be accessed with standard setted/getter methods:

        >>> app_state["test1"] = 1 
        >>> app_state["test2"] = 2
        >>> print(app_state["test1"])

    .. warning::
        It is assumed that global variables are immutable, i.e. once a variable is set, it cannot be changed        

            >>> app_state["test1"] = 3 # Raises AtributeError

    Additionally, it stores all properly parsed commandline arguments.
    """

    def __init__(self):
        """
        Constructor. Initializes dictionary with global variables, sets CPU types as default.

        """
        # Empty commandline arguments.
        self.args = None

        # Field storing global variables.
        self.__globals = dict()

        # Get absolute path to configs from "~/./ptp/configs".
        ptp_path = path.expanduser("~/.ptp/")
        with open(path.join(ptp_path, "config.txt")) as file:
            self.absolute_config_path = file.readline()

        # Initialize logger logfile (as empty for now).
        self.log_file = None
        self.logger = None
        # Set default path to current dir.
        self.log_dir = path.expanduser(".")

        # Set CPU types as default.
        self.set_cpu_types()
        self.use_gpu = False
        self.use_dataparallel = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Reset global counters.
        self.epoch = None # Processor is not using the notion of epoch.
        self.episode = 0

        #### TEST !
        self.test()

    def test(self):
        model = Model(input_size, output_size)
        print("Model DONE!!")
        #time.sleep(2)

        dataset = RandomDataset(input_size, data_size)
        rand_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)
        print("Dataloader DONE!!")
        #time.sleep(2)


        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model = nn.DataParallel(model)        
        model.to(self.device)
        #time.sleep(2)

        for datadict in rand_loader:
            print(type(datadict),"\n")
            data=datadict["index"]
            print("For: before to: input data ({}) size {}, device: {}\n".format(type(data), data.size(), data.device))
            datadict = datadict.to(self.device)
            print("For: before model: input data ({}) size {}, device: {}\n".format(type(data), data.size(), data.device))
            output = model(datadict)
            print("For: after model: output_size ", output.size(),"\n")

        exit(1)


    def set_types(self):
        """
        Enables computations on CUDA if GPU is available.
        Sets the default data types.
        """
        # Determine if GPU/CUDA is available.
        if torch.cuda.is_available() and self.args.use_gpu:
            self.logger.info('Running computations on GPU using CUDA')
            self.set_gpu_types()
            #self.device = torch.device('cuda:0')
            self.use_gpu = True
            if self.args.use_dataparallel:
                self.use_dataparallel = True
        elif self.args.use_gpu:
            self.logger.warning('GPU utilization is demanded but there are no available GPU devices! Using CPUs instead')
        else:
            self.logger.info('GPU utilization is disabled, performing all computations on CPUs')


    def set_cpu_types(self):
        """
        Sets all tensor types to CPU data types.
        """
        self.FloatTensor = torch.FloatTensor
        self.DoubleTensor = torch.DoubleTensor
        self.HalfTensor = torch.HalfTensor
        self.ByteTensor = torch.ByteTensor
        self.CharTensor = torch.CharTensor
        self.ShortTensor = torch.ShortTensor
        self.IntTensor = torch.IntTensor
        self.LongTensor = torch.LongTensor


    def set_gpu_types(self):
        """
        Sets all tensor types to GPU/CUDA data types.
        """
        self.FloatTensor = torch.cuda.FloatTensor
        self.DoubleTensor = torch.cuda.DoubleTensor
        self.HalfTensor = torch.cuda.HalfTensor
        self.ByteTensor = torch.cuda.ByteTensor
        self.CharTensor = torch.cuda.CharTensor
        self.ShortTensor = torch.cuda.ShortTensor
        self.IntTensor = torch.cuda.IntTensor
        self.LongTensor = torch.cuda.LongTensor


    def globalkeys(self):
        """
        Yields global keys.
        """
        for key in self.__globals.keys():
            yield key


    def globalitems(self):
        """
        Yields global keys.
        """
        for key,value in self.__globals.items():
            yield key,value


    def __setitem__(self, key, value, override=False):
        """
        Adds global variable. 

        :param key: Dict Key.

        :param value: Associated value.

        :param override: Indicate whether or not it is authorized to override the existing key.\
        Default: ``False``.
        :type override: bool

        .. warning::
            Once global variable is set, its value cannot be changed (it becomes immutable).
        """
        if not override and key in self.__globals.keys():
            if (self.__globals[key] != value):
                raise KeyError("Global key '{}' already exists and has different value (existing {} vs received {})!".format(key, self.__globals[key], value))
            #msg = 'Cannot add or modify key "{}" as it is already present in global variables'.format(key)
            #raise KeyError(msg)
        else:
            self.__globals[key] = value


    def __getitem__(self, key):
        """
        Value getter function.

        :param key: Dict Key.

        :return: Associated Value.
        """
        if key not in self.__globals.keys():
            msg = "Key '{}' not present in global variables".format(key)
            raise KeyError(msg)
        else:
            return self.__globals[key]
