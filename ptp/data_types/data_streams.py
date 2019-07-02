#!/usr/bin/env python3
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

__author__ = "Tomasz Kornuta, Vincent Marois"

import torch
import collections

class DataStreams(collections.abc.MutableMapping):
    """
    - Mapping: A container object that supports arbitrary key lookups and implements the methods ``__getitem__``, \
    ``__iter__`` and ``__len__``.

    - Mutable objects can change their value but keep their id() -> ease modifying existing keys' value.

    DataStreams: Dict used for storing batches of data by tasks.

    **This is the main object class used to share data between all components through a worker, starting from task to loss and visualization.**
    """

    def __init__(self, *args, **kwargs):
        """
        DataStreams constructor. Can be initialized in different ways:

            >>> data_streams = DataStreams()
            >>> data_streams = DataStreams({'inputs': torch.tensor(), 'targets': numpy.ndarray()})
            >>> # etc.

        :param args: Used to pass a non-keyworded, variable-length argument list.

        :param kwargs: Used to pass a keyworded, variable-length argument list.
        """
        self.__dict__.update(*args, **kwargs)


    def __setitem__(self, key, value, addkey=False):
        """
        key:value setter function.

        :param key: Dict Key.

        :param value: Associated value.

        :param addkey: Indicate whether or not it is authorized to add a new key `on-the-fly`.\
        Default: ``False``.
        :type addkey: bool

        .. warning::

            `addkey` is set to ``False`` by default as setting it to ``True`` removes the constraints of the\
            ``DataStreams`` and enables it to become mutable.
        """
        if not addkey and key not in self.keys():
            msg = 'Cannot modify a non-existing key "{}" in DataStreams'.format(key)
            raise KeyError(msg)
        else:
            self.__dict__[key] = value


    def publish(self, dict_to_add):
        """
        Publishes a new data streams - extends data stream object by adding (keys,values) from data_definitions.

        .. warning::
            This is in-place operation, i.e. extends existing object, does not return a new one.

        :param data_streams: :py:class:`ptp.utils.DataStreams` object to be extended.

        :param data_definitions: key-value pairs.

        """
        for (key,value) in dict_to_add.items():
            if key in self.keys():
                msg = "Cannot extend DataStreams, as {} already present in its keys".format(key)
                raise KeyError(msg)
            # Call setitem with "additional argument".
            self.__setitem__(key, value, addkey=True)


    def reinitialize(self, streams_to_leave):
        """
        Removes all streams (keys and associated values) from DataStreams EXCEPT the ones passed in ``streams_to_leave``.
        """
        # Keys to remove.
        rem_keys =  [key for key in self.keys() if key not in streams_to_leave.keys()]
        # Leave index.
        if 'index' in rem_keys:
            rem_keys.remove('index')
        # Remove.
        for key in rem_keys:
            self.__delitem__(key, delkey=True)


    def __getitem__(self, key):
        """
        Value getter function.

        :param key: Dict Key.

        :return: Associated Value.

        """
        return self.__dict__[key]

    def __delitem__(self, key, delkey=False):
        """
        Delete a key:value pair.

        :param delkey: Indicate whether or not it is authorized to add a delete the key `on-the-fly`.\
        Default: ``False``.
        :type delkey: bool

        .. warning::

            By default, it is not authorized to delete an existing key. Set `delkey` to ``True`` to ignore this\
            restriction and 

        :param key: Dict Key.

        """
        if not delkey:
            msg = 'Cannot delete key "{}" from DataStreams'.format(key)
            raise KeyError(msg)
        else:
            del self.__dict__[key]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __str__(self):
        """
        :return: A simple Dict representation of ``DataStreams``.

        """
        return str(self.__dict__)

    def __repr__(self):
        """
        :return: Echoes class, id, & reproducible representation in the Read–Eval–Print Loop.

        """
        return '{}, DataStreams({})'.format(super(DataStreams, self).__repr__(), self.__dict__)


    def to(self, device=None, keys_to_move=None, non_blocking=False):
        """
        Moves object(s) to device

        .. note::

            Wraps call to ``torch.Tensor.to()``: If this object is already in CUDA memory and on the correct device, \
            then no copy is performed and the original object is returned.
            If an element of `self` is not a ``torch.tensor``, it is returned as is, \
            i.e. We only move the ``torch.tensor`` (s) contained in `self`. \


        :param device: The destination GPU device. Defaults to the current CUDA device.
        :type device: torch.device

        :param non_blocking: If True and the source is in pinned memory, the copy will be asynchronous with respect to \
        the host. Otherwise, the argument has no effect. Default: ``False``.
        :type non_blocking: bool

        """
        for key in self:
            if isinstance(self[key], torch.Tensor):# and (not self[key].is_cuda):
                # Skip keys that are not in the keys_to_move list (if it was passed).
                if keys_to_move is not None and key not in keys_to_move:
                    continue
                self[key] = self[key].to(device=device)#, non_blocking=non_blocking)