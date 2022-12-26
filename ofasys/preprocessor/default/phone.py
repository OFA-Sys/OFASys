# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import copy
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import torch

from ofasys import ModalityType
from ofasys.configure import register_config
from ofasys.utils.file_utils import cached_path

from ..dictionary import Dictionary
from ..instruction import Slot
from ..utils import collate_tokens
from .base import CollateOutput, PreprocessConfig, SafeBasePreprocess


@dataclass
class PhonePreprocessConfig(PreprocessConfig):
    phone_dict_file: str = field(default='oss://ofasys/tasks/tts/vocab.txt', metadata={"help": "phone dict file"})
    use_t2p: bool = field(default=False, metadata={"help": "whether to use text2phone"})
    lang: str = field(default="zh", metadata={"help": "language of text input", "choices": ["zh", "en"]})


@register_config("ofasys.preprocess", "phone", PhonePreprocessConfig)
class DefaultPhonePreprocess(SafeBasePreprocess):
    def __init__(self, global_dict: Dictionary, cfg: PhonePreprocessConfig):
        super().__init__(global_dict, cfg, ModalityType.PHONE)

        self.add_dict_phone_tokens()

        self.use_t2p = cfg.use_t2p
        self.lang = cfg.lang

    def add_dict_phone_tokens(self):
        self.global_dict.add_symbol("<phone>_dict_begin")
        local_phone_dict_file = cached_path(self.cfg.phone_dict_file)
        with open(Path(local_phone_dict_file), "r") as f:
            for line in f:
                line = line.strip().split(" ")[0]
                self.global_dict.add_symbol("<phone>_{}".format(line))
        self.global_dict.add_symbol("<phone>_unk")
        self.global_dict.add_symbol("<phone>_dict_end")
        self.dict_phone_start = self.global_dict.index("<phone>_dict_begin") + 1  # not counting '<phone>_dict_begin'
        self.dict_phone_end = self.global_dict.index(
            "<phone>_unk"
        )  # not counting '<phone>_unk' and '<phone>_dict_end'

    def dummy_slot(self, slot):
        slot.value = torch.empty(0, dtype=torch.long)
        return slot

    def map(self, slot: Slot) -> Slot:
        super().map(slot)
        if not slot.is_src and slot.value is None:
            return self.dummy_slot(slot)

        phone = slot.value
        if self.use_t2p:
            phone = phonemize(phone, self.lang)

        phone_item = " ".join(["<phone>_{}".format(x) for x in phone.split(" ")])
        tokens = self.encode(phone_item)

        if slot.is_src and slot.split == "train" and slot.get_attr('mask_ratio', float):
            tokens = torch.tensor(tokens)
            tokens = self._add_noise(tokens, p=slot.get_attr('mask_ratio', float))

        tokens = torch.cat(
            [torch.LongTensor([self.global_dict.bos()]), tokens, torch.LongTensor([self.global_dict.eos()])]
        )
        slot.value = tokens
        return slot

    def _add_noise(self, phone, p, random_p=0.1):
        if random_p > 0:
            num_to_mask = int(math.ceil(phone.size(0) * p))
            indices = torch.randperm(phone.size(0))[:num_to_mask]
            mask_random = torch.FloatTensor(num_to_mask).uniform_() < random_p
            phone[indices] = self.dict_phone_end
            if mask_random.sum() > 0:
                phone[indices[mask_random]] = torch.randint(
                    self.dict_phone_start, self.dict_phone_end, size=(mask_random.sum(),)
                )
        return phone

    def group_key(self, slot: Slot):
        return ModalityType.TEXT

    def encode(self, phone_item):
        tokens = self.global_dict.encode_line(line=phone_item, add_if_not_exist=False, append_eos=False).long()
        tokens[tokens == self.global_dict.index("<unk>")] = self.global_dict.index("<phone>_unk")
        return tokens

    def decode(self, tokens, escape_unk=False):
        s = self.global_dict.string(
            tokens.int().cpu(),
            # The default unknown string in fairseq is `<unk>`, but
            # this is tokenized by sacrebleu as `< unk >`, inflating
            # BLEU scores. Instead, we use a somewhat more verbose
            # alternative that is unlikely to appear in the real
            # reference, but doesn't get split into multiple tokens.
            unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
        )

        pattern = re.compile(r'^<phone>_(.*)$')
        s = " ".join(x.strip("<phone>_") if pattern.match(x) is not None else x for x in s.split(" "))

        return s

    def postprocess(self, outputs, **sample):
        for idx, single_output in enumerate(outputs):
            if isinstance(single_output, List):
                for sub_output in single_output:
                    sub_output.text = self.decode(sub_output.tokens)
            else:
                single_output.text = self.decode(single_output.tokens)
        return outputs

    def collate(self, slots: List[Slot]) -> CollateOutput:
        """
        Inputs:
            samples: List of Tensors after preprocess

        Returns:
            dict:
                src_tokens (Tensor): batched tokens with shape `[batch, seq_len]`
        """
        super().collate(slots)

        def _collate(key):
            return collate_tokens(
                [slot.value[key] for slot in slots],
                pad_idx=self.global_dict.pad(),
                eos_idx=self.global_dict.eos(),
                pad_to_multiple=self.cfg.pad_to_multiple,
            )

        if slots[0].is_src:
            slots[0].value = _collate('inputs')
            return CollateOutput(slots[0])
        else:
            input_slot, target_slot = copy.copy(slots[0]), copy.copy(slots[0])
            for slot in slots:
                slot.value['prev_output_tokens'] = slot.value['inputs'][:-1]  # skip <EOS>
            input_slot.value = _collate('prev_output_tokens')
            for slot in slots:
                slot.value['target'] = slot.value['target'][1:]  # skip <BOS>
            target_slot.value = _collate('target')

            # for lagecy compatible
            ntokens = target_slot.value.ne(self.global_dict.pad()).long().sum().item()
            extra_dict = {
                "target": target_slot.value,
                "ntokens": ntokens,
            }
            if slots[0].value['constraint_masks'] is not None:
                extra_dict['constraint_masks'] = _collate('constraint_masks')[:, 1:]
            return CollateOutput(input_slot, target_slot, extra_dict)

    def __call__(self, x):
        raise NotImplementedError


g2p = None


def phonemize(text, lang="en", split_="|"):
    if lang == "en":
        global g2p
        if g2p is None:
            _nltk_paths = [
                'nltk/corpora/cmudict/cmudict',
                'nltk/corpora/cmudict/README',
                'nltk/taggers/averaged_perceptron_tagger/averaged_perceptron_tagger.pickle',
                'nltk/corpora/cmudict.zip',
                'nltk/taggers/averaged_perceptron_tagger.zip',
            ]
            from ofasys.utils.fetch_nltk import fetch_nltk_data

            fetch_nltk_data(_nltk_paths)
            from g2p_en import G2p

            g2p = G2p()
        return " ".join(" ".join(split_ if p == " " else p for p in g2p(text)).split())

    else:
        assert lang == "zh"
        from pypinyin import Style, pinyin

        shengmu = pinyin(text, style=Style.INITIALS, strict=False)
        yunmu = pinyin(text, style=Style.FINALS_TONE3, strict=False)
        assert len(shengmu) == len(yunmu)
        final_phone = []
        for s, y in zip(shengmu, yunmu):
            if s[0] == y[0] or s[0] == "":
                final_phone.append(y[0] + " " + split_)
            else:
                final_phone.append(s[0] + " " + y[0] + " " + split_)
        return " ".join(" ".join(final_phone).strip(" " + split_).split())
