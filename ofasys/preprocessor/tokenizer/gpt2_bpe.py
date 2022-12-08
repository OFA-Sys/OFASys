# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from ofasys.utils.file_utils import cached_path
from ofasys.utils.oss import oss_default_resource_path

from .gpt2_bpe_utils import get_encoder

DEFAULT_ENCODER_JSON = oss_default_resource_path('bpe/encoder.json')
DEFAULT_VOCAB_BPE = oss_default_resource_path('bpe/vocab.bpe')
DEFAULT_DICT_BPE = oss_default_resource_path('bpe/dict.txt')


class GPT2BPE(object):
    def __init__(self):
        encoder_json = cached_path(DEFAULT_ENCODER_JSON)
        vocab_bpe = cached_path(DEFAULT_VOCAB_BPE)
        self.bpe = get_encoder(encoder_json, vocab_bpe)

    def encode(self, x: str) -> str:
        return " ".join(map(str, self.bpe.encode(x)))

    def decode(self, x: str) -> str:
        return self.bpe.decode(
            [int(tok) if tok not in {"<unk>", "<mask>"} and not tok.startswith('<') else tok for tok in x.split()]
        )

    def is_beginning_of_word(self, x: str) -> bool:
        return self.decode(x).startswith(" ")

    def _encode(self, x: str):
        return self.bpe.encode(x)

    @property
    def eod(self):
        return self.bpe.encoder['<|endoftext|>']

    @property
    def vocab_size(self):
        return len(self.bpe.encoder)
