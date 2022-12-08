# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import os

import nltk

from ofasys.utils.file_utils import OFA_CACHE_HOME, local_file_lock
from ofasys.utils.oss import oss_to_file

_nltk_paths = [
    'nltk/corpora/stopwords/README',
    'nltk/corpora/stopwords/english',
    'nltk/tokenizers/punkt/README',
    'nltk/tokenizers/punkt/english.pickle',
    'nltk/tokenizers/punkt/PY3/README',
    'nltk/tokenizers/punkt/PY3/english.pickle',
]


@local_file_lock(os.path.join(OFA_CACHE_HOME, 'nltk.lock'))
def fetch_nltk_data():
    for path in _nltk_paths:
        local_path = os.path.join(OFA_CACHE_HOME, path)
        if not os.path.exists(local_path):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            oss_to_file('oss://ofasys/' + path, local_path)
    nltk.data.path.append(os.path.join(OFA_CACHE_HOME, 'nltk'))
