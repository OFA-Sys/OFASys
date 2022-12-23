# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import copy
import json
import string
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from ofasys.configure import ChoiceEnum, register_config
from ofasys.utils.file_utils import cached_path
from ofasys.utils.trie import Trie

from ..dictionary import Dictionary
from ..instruction import ModalityType, Slot
from ..mask_utils import add_whole_word_mask
from ..tokenizer import GPT2BPE, BertBPE, Characters
from ..tokenizer.gpt2_bpe import DEFAULT_DICT_BPE as GPT_DICT
from ..tokenizer.hf_bert_bpe import DEFAULT_DICT_BPE as BERT_DICT
from ..utils import collate_tokens
from .base import CollateOutput, PreprocessConfig, SafeBasePreprocess
from .phone import DefaultPhonePreprocess, PhonePreprocessConfig, phonemize

_transtab = str.maketrans({key: None for key in string.punctuation})


def remove_punctuation(text: str):
    return text.translate(_transtab)


@dataclass
class TextPreprocessConfig(PreprocessConfig):
    ans2label: Optional[str] = field(
        default=None,
        metadata={"help": 'json or the file of ans2label, format: key\tvalue'},
    )
    bpe: ChoiceEnum(['gpt2', 'bert_cn']) = field(
        default='gpt2',
        metadata={"help": "which bpe to use"},
    )
    mask_span_distribution: Optional[str] = field(
        default="span-poisson", metadata={"help": "distribution for masking spans"}
    )
    poisson_lambda: float = field(default=3.0, metadata={"help": "poisson lambda for poisson distribution"})
    random_ratio: float = field(default=0.0, metadata={"help": "random ratio"})
    replace_length: int = field(default=-1, metadata={"help": "replace length"})
    max_src_length: int = field(default=1024, metadata={"help": "max source length of adjacent text slots"})
    max_tgt_length: int = field(default=1024, metadata={"help": "max target length of adjacent text slots"})


@register_config("ofasys.preprocess", "text", TextPreprocessConfig)
class DefaultTextPreprocess(SafeBasePreprocess):
    def build_bpe(self, cfg):
        if cfg.bpe == 'gpt2':
            dict_bpe = GPT_DICT
            bpe = GPT2BPE()
        elif cfg.bpe == 'bert_cn':
            dict_bpe = BERT_DICT
            bpe = BertBPE()
        else:
            raise NotImplementedError
        return bpe, dict_bpe

    def __init__(self, global_dict: Dictionary, cfg: TextPreprocessConfig, sanity_check=False):
        super().__init__(global_dict, cfg, ModalityType.TEXT, sanity_check=sanity_check)
        self.bpe, bpe_dict = self.build_bpe(cfg)
        self.global_dict.add_from_file(cached_path(bpe_dict), prefix='<text>')
        self.global_dict.add_symbol("<mask>", check=False)
        self.dict_text_start, self.dict_text_end = self.global_dict.get_start_end_idx(prefix='<text>')
        assert self.dict_text_end >= self.dict_text_start >= 0
        self.dict_text_end += 1  # move 'end' backwards to cover '<mask>'

        self.ans2label_dict = self.build_ans2label()
        self.constraint_trie = self.build_constraint_trie()

    def prepare_for_generation(self, closed_set, **kwargs):
        self.ans2label_dict = closed_set
        self.constraint_trie = self.build_constraint_trie()

    def dummy_slot(self, slot):
        slot.value = {
            'inputs': torch.empty(0, dtype=torch.long),
            'target': torch.empty(0, dtype=torch.long),
            'constraint_masks': torch.empty(0, dtype=torch.long),
            'raw_tokens': torch.empty(0, dtype=torch.long),
            'prefix_tokens': torch.empty(0, dtype=torch.long),
        }
        return slot

    def map(self, slot: Slot) -> Slot:
        """
        Inputs:
            text: (`str` or `Tensor`) could be:
                A raw text string
                Tokens of a numpy or torch Tensor after user-defined preprocess

        Returns:
            `Torch.LongTensor`: 1-d int64 torch.Tensor
        """
        super().map(slot)
        if not slot.is_src and slot.value is None:
            return self.dummy_slot(slot)

        # Check whether slot.value is already mapped or not.
        # The map function can accept already mapped input even if you call map twice, the second call will return the slot as is.
        # This is helpful where the task class wants to control/use the result of the preprocessor on its own.
        if isinstance(slot.value, dict):
            assert set(slot.value.keys()) == set(['inputs', 'target', 'constraint_masks', 'raw_tokens'])
            return slot

        text = slot.value

        if isinstance(text, str):
            if slot.has_attr('uncased'):
                text = text.lower()
            if slot.has_attr('no_punctuation'):
                text = ' '.join(remove_punctuation(text).strip().split())
            tokens = self.encode(text)
        elif isinstance(text, np.ndarray) and np.issubdtype(text.dtype, np.integer) and text.ndim == 1:
            tokens = self.global_dict.encode(text, add_if_not_exist=False, append_eos=False).long()
        else:
            raise ValueError("Incorrect input for text, only support string or 1-d int Tensor, " f"got {type(text)}")

        # process raw tokens
        if slot.get_attr('max_length', int):
            max_length = slot.get_attr('max_length', int)
            tokens = tokens[:max_length]

        # process input tokens
        if slot.get_attr('noise_ratio', float) and slot.split == 'train':
            inputs = self._add_noise(tokens, slot.get_attr('noise_ratio', float))
        else:
            inputs = tokens

        # mask input tokens following BART(Lewis et al., 2019)
        if slot.get_attr('mask_ratio', float) and slot.split == 'train':
            mask_idx = self.global_dict.index('<mask>')
            inputs = torch.cat(
                [
                    torch.LongTensor([self.global_dict.bos()]),
                    inputs,
                    torch.LongTensor([self.global_dict.eos()]),
                ]
            )
            inputs = add_whole_word_mask(
                source=inputs,
                p=slot.get_attr('mask_ratio', float),
                mask_span_distribution=self.cfg.mask_span_distribution,
                poisson_lambda=self.cfg.poisson_lambda,
                random_ratio=self.cfg.random_ratio,
                mask_idx=mask_idx,
                replace_length=self.cfg.replace_length,
                tgt_dict_size=self.dict_text_end,
            )[1:-1]

        # process target tokens
        if slot.is_src is False:
            no_loss = (slot.is_plaintext and not slot.decoder_plain_with_loss) or slot.has_attr('no_loss')
            loss_mask = torch.ones_like(tokens, dtype=torch.bool) * no_loss
            target = tokens.masked_fill(loss_mask, self.global_dict.pad())
            # prefix_tokens are used in inference
            prefix_tokens = tokens if no_loss and slot.split != 'train' else torch.LongTensor([])
        else:
            target = None
            prefix_tokens = None

        # process constraint mask
        if slot.is_src is False and (slot.has_attr('closed_set')):
            assert tokens[0] != self.global_dict.bos()
            assert tokens[-1] != self.global_dict.eos()
            constraint_masks = torch.zeros((len(tokens), len(self.global_dict)), dtype=torch.bool)
            for i in range(len(tokens)):
                cons_prefix_token = [self.global_dict.bos()] + tokens[:i].tolist()
                cons_nodes = self.constraint_trie.get_next_layer(cons_prefix_token)
                constraint_masks[i][cons_nodes] = True
        else:
            constraint_masks = None

        slot.value = {
            'inputs': inputs,
            'target': target,
            'constraint_masks': constraint_masks,
            'raw_tokens': tokens,
            'prefix_tokens': prefix_tokens,
        }
        return slot

    def group_map(self, slots: List[Slot]) -> List[Slot]:
        super().group_map(slots)
        for slot in slots:
            if isinstance(slot.value, torch.Tensor):  # other mods except text will enter this block
                slot.value = {
                    'inputs': slot.value,
                    'target': None if slot.is_src else slot.value,
                    'constraint_masks': None,
                    'raw_tokens': slot.value,
                    'prefix_tokens': None if slot.is_src else torch.LongTensor([]),
                }

        # process prefix_tokens, skipping tokens with loss is not supported
        if any(map(lambda x: x.value['target'] is not None, slots)):
            no_prefix_tokens_flag = False
            for i, slot in enumerate(slots):
                if len(slot.value['prefix_tokens']) == 0 and len(slot.value['target']) > 0:
                    no_prefix_tokens_flag = True
                if no_prefix_tokens_flag:
                    slot.value['prefix_tokens'] = torch.LongTensor([])

        # process constraint mask
        if any(map(lambda x: x.value['constraint_masks'] is not None, slots)):
            for i, slot in enumerate(slots):
                if slot.value['constraint_masks'] is None:
                    slot.value['constraint_masks'] = torch.zeros(
                        (len(slot.value['raw_tokens']), len(self.global_dict)), dtype=torch.bool
                    )

        value = {}
        for key in slots[0].value.keys():
            if any(map(lambda slot: slot.value[key] is not None, slots)):
                value[key] = torch.cat([slot.value[key] for slot in slots], dim=0)
                if key in ['inputs', 'raw_tokens', 'target', 'prefix_tokens'] and not slots[0].has_attr(
                    'disable_auto_boseos'
                ):
                    value[key] = torch.cat(
                        [
                            torch.LongTensor([self.global_dict.bos()]),
                            value[key],
                            torch.LongTensor([self.global_dict.eos()]),
                        ]
                    )
            else:
                value[key] = None

        if any(map(lambda x: x.value['constraint_masks'] is not None, slots)) and self.constraint_trie is not None:
            # Only if the last slot has constraint_masks, the eos has constraint_masks.
            # Currently, only tasks image_classify and video_classify have closed_set attribute.
            # In these two cases, slots have only one slot. Thus the last slot has constraint_masks.
            assert slots[-1].value['constraint_masks'] is not None
            constraint_eos = torch.zeros((1, len(self.global_dict)), dtype=torch.bool)
            cons_prefix_token = [self.global_dict.bos()] + slots[-1].value['raw_tokens'].tolist()
            cons_nodes = self.constraint_trie.get_next_layer(cons_prefix_token)
            constraint_eos[0][cons_nodes] = True
            value['constraint_masks'] = torch.cat(
                [
                    torch.zeros((1, len(self.global_dict)), dtype=torch.bool),
                    value['constraint_masks'],
                    constraint_eos,
                ]
            )

        for key in value.keys():
            if value[key] is not None:
                if slots[0].is_src:
                    max_length = getattr(self.cfg, 'max_src_length', None)
                elif not slots[0].is_src:
                    max_length = getattr(self.cfg, 'max_tgt_length', None)
                if max_length is not None:
                    value[key] = value[key][: max_length + 1]

        return [
            Slot(
                modality=slots[0].modality,
                is_src=slots[0].is_src,
                value=value,
                global_position=0,
                column_name=','.join([s.column_name for s in slots]),
                preprocess=slots[0].preprocess,
                is_plaintext=False,
                split=slots[0].split,
                attributes=slots[0].attributes,
            )
        ]

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
                slot.value['prev_output_tokens'] = slot.value['inputs'][:-1]
            input_slot.value = _collate('prev_output_tokens')
            for slot in slots:
                slot.value['target'] = slot.value['target'][1:]
            target_slot.value = _collate('target')
            for slot in slots:
                slot.value['prefix_tokens'] = slot.value['prefix_tokens'][1:-1]  # skip bos and eos
            prefix_tokens = _collate('prefix_tokens')
            # for legacy compatible
            ntokens = target_slot.value.ne(self.global_dict.pad()).long().sum().item()
            extra_dict = {
                "target": target_slot.value,
                "ntokens": ntokens,
                "dict_start": getattr(self, 'dict_text_start', None),
                "dict_end": getattr(self, 'dict_text_end', None),
                "prefix_tokens": prefix_tokens,
            }
            if slots[0].value['constraint_masks'] is not None:
                extra_dict['constraint_masks'] = _collate('constraint_masks')[:, 1:]
            return CollateOutput(input_slot, target_slot, extra_dict)

    def encode(self, text):
        s = self.bpe.encode(' ' + text.strip())
        s = self.add_prefix(s, '<text>')
        tokens = self.global_dict.encode_line(line=s, add_if_not_exist=False, append_eos=False).long()
        return tokens

    def add_prefix(self, s, prefix):
        if prefix[-1] != '_':
            prefix = prefix + '_'
        special_symbols = set(['<s>', '<pad>', '</s>', '<unk>', '<mask>'])
        return ' '.join([prefix + token if token not in special_symbols else token for token in s.strip().split()])

    def remove_prefix(self, s, prefix):
        if prefix[-1] != '_':
            prefix = prefix + '_'
        return ' '.join([token[len(prefix) :] if token.startswith(prefix) else token for token in s.strip().split()])

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
        s = self.remove_prefix(s, '<text>')
        s = self.bpe.decode(s).strip()
        return s

    def postprocess(self, outputs, **sample):
        def process_fn(idx: int, output):
            if "prefix_tokens" in sample:
                prefix_len = (
                    sample["prefix_tokens"][idx].ne(self.global_dict.pad())
                    * sample["prefix_tokens"][idx].ne(self.global_dict.eos())
                ).sum()
                output.tokens = output.tokens[prefix_len:]
            output.text = self.decode(output.tokens)

        for idx, single_output in enumerate(outputs):
            if isinstance(single_output, List):
                for sub_output in single_output:
                    process_fn(idx, sub_output)
            else:
                process_fn(idx, single_output)
        return outputs

    def _add_noise(self, target, p: float):
        noise_indices = torch.FloatTensor(target.size(0)).uniform_() < p
        target = target.clone()
        target[noise_indices] = torch.randint(self.dict_text_start, self.dict_text_end, size=(noise_indices.sum(),))
        return target

    def build_ans2label(self):
        if not self.cfg.ans2label:
            return None
        try:
            ans2label_dict = json.loads(self.cfg.ans2label)
        except json.JSONDecodeError:
            ans2label_dict = {}
            with open(cached_path(self.cfg.ans2label)) as reader:
                for line in reader:
                    k, v = line.rstrip().split('\t')
                    ans2label_dict[k] = int(v)
        return ans2label_dict

    def build_constraint_trie(self):
        if not self.ans2label_dict:
            return None
        constraint_trie = Trie(self.global_dict.eos())
        for ans in self.ans2label_dict.keys():
            ans_item = self.global_dict.encode_line(
                line=self.add_prefix(self.bpe.encode(' ' + ans), '<text>'), add_if_not_exist=False, append_eos=False
            ).long()
            constraint_trie.insert([self.global_dict.bos()] + ans_item.tolist() + [self.global_dict.eos()])
        return constraint_trie


@dataclass
class TextForPhonePreprocessConfig(TextPreprocessConfig, PhonePreprocessConfig):
    bpe: ChoiceEnum(['gpt2', 'bert_cn', 'characters']) = field(
        default='gpt2',
        metadata={"help": "which bpe to use"},
    )
    dict_bpe: Optional[str] = field(default=None, metadata={"help": "dictionary for bpe"})
    use_t2p: bool = field(default=True, metadata={"help": "whether to use text2phone"})
    lang: str = field(default="en", metadata={"help": "language of text input", "choices": ["zh", "en"]})


@register_config("ofasys.preprocess", "text_phone", TextForPhonePreprocessConfig)
class TextForPhonePreprocess(DefaultPhonePreprocess, DefaultTextPreprocess):
    def build_bpe(self, cfg):
        if cfg.bpe == 'gpt2':
            dict_bpe = GPT_DICT
            bpe = GPT2BPE()
        elif cfg.bpe == 'bert_cn':
            dict_bpe = BERT_DICT
            bpe = BertBPE()
        elif cfg.bpe == 'characters':
            dict_bpe = cfg.dict_bpe
            bpe = Characters()
        else:
            raise NotImplementedError
        return bpe, dict_bpe

    def __init__(self, global_dict: Dictionary, cfg: TextForPhonePreprocessConfig):
        DefaultTextPreprocess.__init__(self, global_dict, cfg)
        self.add_dict_phone_tokens()

        self.use_t2p = cfg.use_t2p
        self.lang = cfg.lang

    def dummy_slot(self, slot):
        slot.value = {
            'inputs': torch.empty(0, dtype=torch.long),
            'target': torch.empty(0, dtype=torch.long),
            'constraint_masks': torch.empty(0, dtype=torch.long),
            'raw_tokens': torch.empty(0, dtype=torch.long),
            'prefix_tokens': torch.empty(0, dtype=torch.long),
            'phone_tokens': torch.empty(0, dtype=torch.long),
        }
        return slot

    def map(self, slot: Slot) -> Slot:
        if not slot.is_src and slot.value is None:
            return self.dummy_slot(slot)

        if isinstance(slot.value, str):
            text = slot.value
        else:
            return DefaultTextPreprocess.map(self, slot)
        slot = DefaultTextPreprocess.map(self, slot)
        # process phone
        if slot.is_src is False and self.use_t2p:
            phone = phonemize(text, self.lang, "")
            phone_item = " ".join(["<phone>_{}".format(x) for x in phone.split(" ")])
            phone_tokens = self.global_dict.encode_line(
                line=phone_item, add_if_not_exist=False, append_eos=False
            ).long()
            phone_tokens[phone_tokens == self.global_dict.index("<unk>")] = self.global_dict.index("<phone>_unk")
            phone_tokens = torch.cat([phone_tokens, torch.LongTensor([self.global_dict.eos()])])
        else:
            phone_tokens = None
        slot.value['phone_tokens'] = phone_tokens

        return slot

    def collate(self, slots: List[Slot]) -> CollateOutput:
        """
        Inputs:
            samples: List of Tensors after preprocess

        Returns:
            dict:
                src_tokens (Tensor): batched tokens with shape `[batch, seq_len]`
        """
        collate_output = DefaultTextPreprocess.collate(self, slots)

        def _collate(key):
            return collate_tokens(
                [slot.value[key] for slot in slots],
                pad_idx=self.global_dict.pad(),
                eos_idx=self.global_dict.eos(),
                pad_to_multiple=self.cfg.pad_to_multiple,
            )

        # not is_src
        if collate_output.net_target_slot is not None:
            assert collate_output.sample_extra is not None
            if self.use_t2p:
                target_phone_slot = copy.copy(slots[0])
                target_phone_slot.value = _collate('phone_tokens')
                collate_output.sample_extra['dict_start'] = self.global_dict.index("<phone>_dict_begin") + 1
                collate_output.sample_extra['dict_end'] = self.global_dict.index("<phone>_dict_end")
                collate_output.sample_extra['encoder_target'] = target_phone_slot.value
            else:
                collate_output.sample_extra['encoder_target'] = collate_output.sample_extra['target']
            collate_output.sample_extra['blank_id'] = self.global_dict.index("<phone>_dict_begin")
        return collate_output

    def encode(self, text):
        return DefaultTextPreprocess.encode(self, text=text)

    def decode(self, tokens, escape_unk=False):
        return DefaultTextPreprocess.decode(self, tokens=tokens, escape_unk=escape_unk)
