# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import re

from .characters import Characters
from .gpt2_bpe import GPT2BPE
from .hf_bert_bpe import BertBPE

__all__ = [
    'GPT2BPE',
    'BertBPE',
    'Characters',
]


SPACE_NORMALIZER = re.compile(r"\s+")


def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()
