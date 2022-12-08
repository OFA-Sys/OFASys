# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from ofasys.utils.file_utils import cached_path
from ofasys.utils.oss import oss_default_resource_path

DEFAULT_VOCAB_BPE = oss_default_resource_path('bpe/bert_cn/vocab.txt')
DEFAULT_DICT_BPE = oss_default_resource_path('bpe/bert_cn/dict.txt')


class BertBPE(object):
    def __init__(self):
        try:
            from transformers import BertTokenizer
        except ImportError:
            raise ImportError("Please install transformers with: pip install transformers")

        self.bert_tokenizer = BertTokenizer(cached_path(DEFAULT_VOCAB_BPE), False)

    def encode(self, x: str) -> str:
        return " ".join(self.bert_tokenizer.tokenize(x))

    def decode(self, x: str) -> str:
        return self.bert_tokenizer.clean_up_tokenization(self.bert_tokenizer.convert_tokens_to_string(x.split(" ")))

    def is_beginning_of_word(self, x: str) -> bool:
        return not x.startswith("##")
