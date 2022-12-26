import copy
import json
import string
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from ofasys.configure import register_config
from ofasys.utils.file_utils import cached_path

from ..dictionary import Dictionary
from ..instruction import ModalityType, Slot
from ..utils import collate_tokens
from .base import CollateOutput, PreprocessConfig, SafeBasePreprocess

_transtab = str.maketrans({key: None for key in string.punctuation})


def remove_punctuation(text: str):
    return text.translate(_transtab)


@dataclass
class CategoryPreprocessConfig(PreprocessConfig):
    ans2label: Optional[str] = field(
        default=None,
        metadata={"help": 'json or the file of ans2label, format: key\tvalue'},
    )


@register_config("ofasys.preprocess", "category", CategoryPreprocessConfig)
class CategoryPreprocess(SafeBasePreprocess):
    def __init__(self, global_dict: Dictionary, cfg: CategoryPreprocessConfig):
        super().__init__(global_dict, cfg, ModalityType.TEXT)
        if not cfg.is_active or cfg.ans2label is None or not cfg.ans2label:
            return
        assert cfg.ans2label is not None
        self.global_dict = Dictionary(bos=None, pad=None, eos=None, unk=None)
        self.dict_start = len(global_dict)
        self.dict_end = len(global_dict)
        self.ans2label_dict = self.build_ans2label()

    def map(self, slot: Slot) -> Slot:
        """
        Inputs:
            text: (`str` or `Tensor`) could be:
                A raw text string
                Tokens of a numpy or torch Tensor after user-defined preprocess

        Returns:
            `Torch.LongTensor`: 1-d int64 torch.Tensor
        """
        SafeBasePreprocess.map(self, slot)
        # Check whether slot.value is already mapped or not.
        # The map function can accept already mapped input even if you call map twice, the second call will return the slot as is.
        # This is helpful where the task class wants to control/use the result of the preprocessor on its own.
        if isinstance(slot.value, dict):
            assert set(slot.value.keys()) == set(['inputs', 'target', 'constraint_masks', 'raw_tokens'])
            return slot
        text = slot.value

        if isinstance(text, str):
            tokens = self.encode_rich(
                text,
                uncased=slot.has_attr('uncased'),
                no_punctuation=slot.has_attr('no_punctuation'),
            )
        elif isinstance(text, np.ndarray) and np.issubdtype(text.dtype, np.integer) and text.ndim == 1:
            tokens = self.global_dict.encode(text, add_if_not_exist=False, append_eos=False).long()
        else:
            raise ValueError("Incorrect input for text, only support string or 1-d int Tensor, " f"got {type(text)}")

        inputs = tokens

        # process target tokens
        assert slot.is_src is False
        if slot.is_src is False:
            no_loss = (slot.is_plaintext and not slot.decoder_plain_with_loss) or slot.has_attr('no_loss')
            target = tokens
            # target = tokens.masked_fill(loss_mask, self.global_dict.pad())
            # prefix_tokens are used in inference
            prefix_tokens = tokens if no_loss and slot.split != 'train' else torch.LongTensor([])
        else:
            target = None
            prefix_tokens = None

        slot.value = {
            'inputs': inputs,
            'target': target,
            'constraint_masks': None,
            'raw_tokens': tokens,
            'prefix_tokens': prefix_tokens,
        }
        return slot

    def collate(self, slots: List[Slot]) -> CollateOutput:
        """
        Inputs:
            samples: List of Tensors after preprocess

        Returns:
            dict:
                src_tokens (Tensor): batched tokens with shape `[batch, seq_len]`
        """
        SafeBasePreprocess.collate(self, slots)

        def _collate(key):
            return collate_tokens(
                [slot.value[key] for slot in slots],
                pad_idx=-100 if self.global_dict.pad() is None else self.global_dict.pad(),
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
            # for slot in slots:
            #     slot.value['target'] = slot.value['target'][1:]
            target_slot.value = _collate('target')
            for slot in slots:
                slot.value['prefix_tokens'] = slot.value['prefix_tokens'][1:]
            prefix_tokens = _collate('prefix_tokens')
            # for legacy compatible
            if self.global_dict.pad() is not None:
                ntokens = target_slot.value.ne(self.global_dict.pad()).long().sum().item()
            else:
                ntokens = target_slot.value.ne(-100).long().sum().item()
            extra_dict = {
                "target": target_slot.value,
                "ntokens": ntokens,
                "dict_start": self.dict_start,
                "dict_end": self.dict_end,
                "prefix_tokens": prefix_tokens,
            }
            if slots[0].value['constraint_masks'] is not None:
                extra_dict['constraint_masks'] = _collate('constraint_masks')[:, 1:]
            return CollateOutput(input_slot, target_slot, extra_dict)

    def encode_rich(self, text, uncased=False, no_punctuation=False):
        tokens = []
        part = text.strip()
        if uncased:
            part = part.lower()
        if no_punctuation:
            part = ' '.join(remove_punctuation(part).strip().split())
        tokens.append(self.encode(' ' + part.strip()))
        if len(tokens) == 0:  # torch.cat will fail when given an empty list
            return torch.LongTensor([])
        return torch.cat(tokens)

    def encode(self, text):
        s = ' ' + text.strip()
        tokens = self.global_dict.encode_line(line=s, add_if_not_exist=False, append_eos=False).long()
        assert len(tokens.shape) == 1 and tokens.size(0) == 1
        return tokens

    def decode(self, tokens, escape_unk=False):
        s = self.global_dict.string(
            tokens[0].int().cpu(),
            # The default unknown string in fairseq is `<unk>`, but
            # this is tokenized by sacrebleu as `< unk >`, inflating
            # BLEU scores. Instead, we use a somewhat more verbose
            # alternative that is unlikely to appear in the real
            # reference, but doesn't get split into multiple tokens.
            unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
        )
        s = s.strip()
        return s

    def postprocess(self, outputs, **sample):
        for idx, single_output in enumerate(outputs):
            if isinstance(single_output, List):
                for sub_output in single_output:
                    sub_output.text = self.decode(sub_output.tokens)
            else:
                single_output.text = self.decode(single_output.tokens)
        return outputs

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
                    self.global_dict.add_symbol(k)

        return ans2label_dict
