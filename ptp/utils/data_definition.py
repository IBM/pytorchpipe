
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

__author__ = "Tomasz Kornuta"

import torch
import collections
from torch.nn import Module


# Helper collection type.
__DataDefinition = collections.namedtuple(
    'DataDefinition',
    (
        'size',
        'type',
        'description'
    ))


class DataDefinition(__DataDefinition):
    """
    Tuple used by for storing definitinos of fields of DataDict.
    Used for DataDict initialization, handshaking and debugging.
    """
    __slots__ = ()