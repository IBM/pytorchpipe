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

__author__ = "Alexis Asseman, Tomasz Kornuta"

import yaml
from collections.abc import Mapping

from ptp.configuration.config_registry import ConfigRegistry


class ConfigInterface(Mapping):
    """
    Interface to the :py:class:`ConfigRegistry` singleton.

    Inherits :py:class:`collections.Mapping`, and therefore exposes functionality close to a `dict`.

    Offers a read (through :py:class:`collections.Mapping` interface) and write \
    (through :py:func:`add_default_params` and :py:func:`add_config_params` methods) \
    view of the :py:class:`ConfigRegistry`.

        .. warning::

            This class is the only interface to :py:class:`ConfigRegistry`, and thus the only way to \
            interact with it.

    """

    def __init__(self, *keys):
        """
        Constructor:

            - Call base constructor (:py:class:`Mapping`),
            - Initializes the :py:class:`ConfigRegistry`,
            - Initializes empty keys_path list


        :param keys: Sequence of keys to the subtree of the registry. The subtree hierarchy will be created if it \
        does not exist. If empty, shows the whole registry.
        :type keys: sequence / collection: dict, list etc.

        .. note::

            Calling :py:func:`to_dict` after initializing a :py:class:`ConfigInterface` with ``keys``, \
            will throw a ``KeyError``.

            Adding `default` & `config` params should be done through :py:func:`add_default_param` and \
            :py:func:`add_config_param`.

            ``keys`` is mainly purposed for the recursion of :py:class:`ConfigInterface`.

        """
        # call base constructor
        super(ConfigInterface, self).__init__()

        # empty ConfigRegistry
        self._config_registry = ConfigRegistry()

        # keys_path as a list
        self._keys_path = list(keys)

    def _lookup(self, *keys):
        """
        Returns the :py:class:`ConfigInterface` or the value living under ``keys``.

        :param keys: Sequence of keys to the subtree of the registry. If empty, shows the whole registry.
        :type keys: sequence / collection: dict, list etc.

        """

        def lookup_recursion(dic, key, *keys):
            if keys:
                return lookup_recursion(dic[key], *keys)
            return dic[key]

        # construct the path from the existing keys path
        lookup_keys = self._keys_path + list(keys)

        if len(lookup_keys) > 0:
            r = lookup_recursion(self._config_registry, *lookup_keys)
            return r
        else:
            return self._config_registry

    def _nest_dict(self, d: dict):
        """
        Create a nested dict using ``d`` living under ``self._keys_path``.

        :param d: dict to nest under ``self._keys_path``.
        :type d: dict

        :return: nested ``d``.

        """

        def nest_dict_recursion(dic, key, *keys):
            if keys:
                dic[key] = {}
                return nest_dict_recursion(dic[key], *keys)
            else:
                dic[key] = {}
                dic[key].update(d)

        if len(self._keys_path) > 0:
            nested_dict = {}
            nest_dict_recursion(nested_dict, *self._keys_path)
            return nested_dict
        else:
            return d

    def to_dict(self):
        """

        :return: `dict` containing a snapshot of the current :py:class:`ConfigInterface` tree.
        """
        return dict(self._lookup())

    def __getitem__(self, key):
        """
        Get parameter value under ``key``.

        The parameter dict is derived from the default parameters updated with the config parameters.

        :param key: key to value in the :py:class:`ConfigInterface` tree.
        :type key: str

        :return: :py:class:`ConfigInterface` ``[key]`` or value if leaf of the :py:class:`ConfigRegistry` tree.

        """
        v = self._lookup(key)
        if isinstance(v, dict) or isinstance(v, ConfigRegistry):
            return ConfigInterface(*self._keys_path, key)
        else:  # We are at a leaf of the tree
            return v

    def __len__(self):
        """

        :return: Length of the :py:class:`ConfigInterface`.

        """
        return len(self._lookup())

    def __iter__(self):
        """

        :return: Iterator over the :py:class:`ConfigInterface`.

        """
        return iter(self._lookup())

    def __eq__(self, other):
        """
        Check whether two interfaces are equal (just for the purpose of compatibility with base Mapping class).
        """
        if isinstance(other, self.__class__):

            return self._config_registry == other._config_registry and self._keys_path == other._keys_path
        else:
            return False

            
    def leafs(self):
        """
        Yields the leafs of the current :py:class:`ConfigInterface`.

        """
        for key, value in self.items():
            if isinstance(value, ConfigInterface):
                for inner_key in value.leafs():
                    yield inner_key
            else:
                yield key

    def set_leaf(self, leaf_key, leaf_value):
        """
        Update the value of the specified ``leaf_key`` of the current :py:class:`ConfigInterface` \
        with the specified ``leaf_value``.

        :param leaf_key: leaf key to update.
        :type leaf_key: str

        :param leaf_value: New value to set.

        :return: ``True`` if the leaf value has been changed, ``False`` if ``leaf_key`` is not in \
        :py:func:`ConfigInterface.leafs`.

        """
        # check first if we can access the leaf to change
        if leaf_key not in list(self.leafs()):
            return False

        for key, value in self.items():

            if isinstance(value, ConfigInterface):
                # hit a sub ConfigInterface, recursion
                if value.set_leaf(leaf_key, leaf_value):
                    return True  # leaf has been changed, done
                else:
                    continue  # have not found the key, continue
            elif key == leaf_key:
                self.add_config_params({key: leaf_value})
                return True  # leaf has been changed, done

    def add_default_params(self, default_params: dict):
        """
        Appends ``default_params`` to the `config` parameter dict of the :py:class:`ConfigRegistry`.

        .. note::

            This method should be used by the objects necessitating default values \
            (problems, models, workers etc.).

        :param default_params: Dictionary containing `default` values.
        :type default_params: dict

        The dictionary will be inserted into the subtree keys path indicated at the initialization of the \
        current :py:class:`ConfigInterface`.

        """
        self._config_registry.add_default_params(
            self._nest_dict(default_params)
        )

    def add_config_params(self, config_params: dict):
        """
        Appends ``config_params`` to the `config` parameter dict of the :py:class:`ConfigRegistry`.

        .. note::

            This is intended for the user to dynamically (re)configure his experiments.

        :param config_params: Dictionary containing `config` values.
        :type config_params: dict

        The dictionary will be inserted into the subtree keys path indicated at the initialization of the \
        current :py:class:`ConfigInterface`.

        """
        self._config_registry.add_config_params(
            self._nest_dict(config_params)
        )

    def del_default_params(self, key):
        """
        Removes the entry from the `default` params living under ``key``.

        The entry can either be a subtree or a leaf of the `default` params tree.

        :param key: key to subtree / leaf in the `default` params tree.
        :type key: str

        """
        if type(key) is list:
            keypath = self._keys_path + key
        else:
            keypath = self._keys_path + [key]
        self._config_registry.del_default_params(keypath)

    def del_config_params(self, key):
        """
        Removes the entry from the `config` params living under ``key``.

        The entry can either be a subtree or a leaf of the `config` params tree.

        :param key: key to subtree / leaf in the `config` params tree.
        :type key: str

        """
        if type(key) is list:
            keypath = self._keys_path + key
        else:
            keypath = self._keys_path + [key]
        self._config_registry.del_config_params(keypath)

    def add_config_params_from_yaml(self, yaml_path: str):
        """
        Helper function adding `config` params by loading the file at ``yaml_path``.

        Wraps call to :py:func:`add_default_param`.

        :param yaml_path: Path to a ``.yaml`` file containing config parameters.
        :type yaml_path: str`

        """
        # Open file and try to add that to list of parameter dictionaries.
        with open(yaml_path, 'r') as stream:
            # Load parameters.
            params_from_yaml = yaml.safe_load(stream)

        # add config param
        self.add_config_params(params_from_yaml)
