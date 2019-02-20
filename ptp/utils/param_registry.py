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

import copy
from abc import ABCMeta
from collections import Mapping
from miprometheus.utils.singleton import SingletonMetaClass


class MetaSingletonABC(SingletonMetaClass, ABCMeta):
    """
    Metaclass that inherits both SingletonMetaClass, and ABCMeta \
    (collection.Mappings' metaclass).
    """
    pass


class ParamRegistry(Mapping, metaclass=MetaSingletonABC):
    """
    Registry singleton for the parameters.

    Registers `default` values (coming from workers, models, problems, etc) as well as \
    `config` values loaded by the user for a particular experiment.

    Parameters can be read from the registry by indexing.
    The returned parameters are the `default` ones superseded by all the `config` ones.

    The merging of `default` and `config` parameters is computed every time the registry is changed.

    Can contain nested parameters sections (acts as a dict).


    .. warning::

            This class should not be used except through :py:class:`ParamInterface`.


    """

    def __init__(self):
        """
        Constructor:

            - Call base constructor (:py:class:`Mapping`),
            - Initializes empty parameters dicts for:

                - `Default` parameters,
                - `Config` parameters,
                - Resulting tree.


        """
        super(ParamRegistry, self).__init__()
        # Default parameters set in the code.

        self._default_params = {}
        # Parameters read from configuration files.
        self._superseding_config_params = {}
        # Resulting parameters.
        self._params = dict()

    def _update_params(self):
        """
        Update the resulting parameters dict from the `default` parameters dict superseded by the \
        `config` params registry.

        """
        # deep copy to avoid the config params leaking to `self._default_params`
        self._params = copy.deepcopy(self._default_params)
        self.update_dict_recursively(self._params, self._superseding_config_params)

    def add_default_params(self, default_params: dict):
        """
        Appends ``default_params`` to the `default` parameter dict of the current :py:class:`ParamRegistry`, \
        and update the resulting parameters dict.

        .. note::

            This method should be used by the objects necessitating default values \
            (problems, models, workers etc.).

        :param default_params: Dictionary containing default values.
        :type default_params: dict

        """
        # Update default params list.
        self.update_dict_recursively(self._default_params, default_params)
        # Merge default with config list.
        self._update_params()

    def add_config_params(self, config_params: dict):
        """
        Appends ``config_params`` to the `config` parameter dict of the current :py:class:`ParamRegistry`, \
        and update the resulting parameters dict.

        .. note::

            This is intended for the user to dynamically (re)configure his experiments.

        :param config_params: Dictionary containing config values.
        :type config_params: dict

        """
        # Update config params list.
        self.update_dict_recursively(self._superseding_config_params, config_params)
        # Merge default with config list.
        self._update_params()

    def del_default_params(self, keypath: list):
        """
        Removes an entry from the `default` parameter dict of the current :py:class:`ParamRegistry`, \
        and update the resulting parameters dict.

        The entry can either be a subtree or a leaf of the `default` parameter dict.

        :param keypath: list of keys to subtree / leaf in the `default` parameter dict.
        :type keypath: list

        """
        self.delete_subtree(self._default_params, keypath)
        self._update_params()

    def del_config_params(self, keypath: list):
        """
        Removes an entry from the `config` parameter dict of the current :py:class:`ParamRegistry`, \
        and update the resulting parameters dict.

        The entry can either be a subtree or a leaf of the `config` parameter dict.

        :param keypath: list of keys to subtree / leaf in the `config` parameter dict.
        :type keypath: list

        """
        self.delete_subtree(self._superseding_config_params, keypath)
        self._update_params()

    def __getitem__(self, key):
        """
        Get parameter value under ``key``.

        The parameter dict is derived from the default parameters updated with the config parameters.

        :param key: key to value in the :py:class:`ParamRegistry`.
        :type key: str

        :return: Parameter value

        """
        return self._params[key]

    def __iter__(self):
        """

        :return: Iterator over the :py:class:`ParamRegistry`.

        """
        return iter(self._params)

    def __len__(self):
        """

        :return: Length of the :py:class:`ParamRegistry`.

        """
        return len(self._params)

    def update_dict_recursively(self, current_node, update_node):
        """
        Recursively update the ``current_node`` of the :py:class:`ParamRegistry` with the values of \
        the ``update_node``.

        Starts from the root of the ``current_node``.

        :param current_node: Current (default or config) node.
        :type current_node: :py:class:`ParamRegistry` (inheriting from :py:class:`Mapping`)

        :param update_node: Values to be added/updated to the ``current_node``.
        :type update_node: :py:class:`ParamRegistry` (inheriting from :py:class:`Mapping`)

        :return: Updated current node.

        """
        for k, v in update_node.items():
            if isinstance(v, Mapping):
                current_node[k] = self.update_dict_recursively(current_node.get(k, {}), v)
            else:
                current_node[k] = v
        return current_node

    @staticmethod
    def delete_subtree(current_dict, keypath: list):
        """
        Delete the subtree indexed by the ``keypath`` from the ``current_dict``.

        :param current_dict: dictionary to act on.
        :type current_dict: dict

        :param keypath: list of keys to subtree in ``current_dict`` to delete
        :type keypath: list
        """
        if len(keypath) < 1:
            raise KeyError

        def lookup_recursion(dic, key, *keys):
            if keys:
                return lookup_recursion(dic[key], *keys)
            return dic[key]

        lookup_keys = keypath[:-1]  # We keep the last key for use with `del`
        if len(keypath) > 0:
            r = lookup_recursion(current_dict, *lookup_keys)
            del r[keypath[-1]]
        else:
            del current_dict[keypath[-1]]


if __name__ == '__main__':

    params = ParamRegistry()

    params.add_default_params({'default_0': {'default_1': 'str'}})
    params.add_config_params({'config_0': {'config_1': 'int'}})

    print(dict(params))
    params.del_config_params(['config_0', 'config_1'])
    print(dict(params))

    params.del_default_params(['default_0', 'default_1'])
    print(dict(params))
