# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import os

import nltk

from ofasys.utils.file_utils import OFA_CACHE_HOME, local_file_lock
from ofasys.utils.oss import oss_to_file


@local_file_lock(os.path.join(OFA_CACHE_HOME, 'nltk.lock'))
def fetch_nltk_data(_nltk_paths):
    for path in _nltk_paths:
        local_path = os.path.join(OFA_CACHE_HOME, path)
        if not os.path.exists(local_path):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            oss_to_file('oss://ofasys/' + path, local_path)
    nltk_path = os.path.join(OFA_CACHE_HOME, 'nltk')
    if nltk_path not in nltk.data.path:
        nltk.data.path.append(nltk_path)
