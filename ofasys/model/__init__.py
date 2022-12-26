# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from .fairseq_model import BaseModel
from .ofa import GeneralistModel
from .decoders.pooling import OFAPoolingModel

__all__ = [
    'BaseModel',
    'GeneralistModel',
    'OFAPoolingModel'
]
