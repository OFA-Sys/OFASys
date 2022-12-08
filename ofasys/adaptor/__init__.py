# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from .base import AdaptorOutput
from .general import OFAAdaptorConfig, OFAGeneralAdaptor, default_adaptor

__all__ = [
    'AdaptorOutput',
    'OFAGeneralAdaptor',
    'default_adaptor',
    'OFAAdaptorConfig',
]
