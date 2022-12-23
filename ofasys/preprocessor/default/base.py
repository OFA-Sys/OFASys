# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from ofasys import ModalityType
from ofasys.configure import BaseDataclass
from ofasys.preprocessor.utils import collate_tokens

from ..dictionary import Dictionary
from ..instruction import Instruction, Slot


@dataclass
class CollateOutput:
    net_input_slot: Slot
    net_target_slot: Optional[Slot] = None
    sample_extra: Optional[Dict] = None


@dataclass
class PreprocessConfig(BaseDataclass):
    is_active: bool = field(default=False, metadata={"help": "use for config_store, user should not use it."})
    pad_to_multiple: int = field(default=1, metadata={"help": "pad to multiple when batch collating."})


class BasePreprocess(ABC):
    """
    The preprocessor converts the raw modal data of a single sample into the batch input accepted by the neural network.
    The preprocessor runs inside the dataloader and can be parallelized on multiple processes through setting num_workers.

    Each mode has its own preprocessors.
    Preprocessing generally consists of four sequential phases, namely **instruction_map**, **map**, **group_map**, and **collate**.

    Args:
        global_dict (Dictionary): global vocab
        cfg (PreprocessConfig): preprocess config
    """

    def __init__(self, global_dict: Dictionary, cfg: PreprocessConfig):
        self.global_dict = global_dict
        self.cfg = cfg

    def instruction_map(self, ist_data: Instruction) -> Instruction:
        """
        The **instruction_map** phase of the preprocessor takes the whole **Instruction** as input and outputs a preprocessed one.
        This function is mainly used to cooperatively process multiple types of modal inputs.
        """
        return ist_data

    def dummy_slot(self, slot):
        """
        Set dummy value for slot, which is used for inference.
        """
        return slot

    @abstractmethod
    def map(self, slot: Slot) -> Slot:
        """
        The **map** phase of the preprocessor takes a **Slot** as input and outputs a preprocessed Slot. This phase defines how to preprocess the raw data of a single modal.

        Args:
            inputs (Slot): raw input data.

        Returns:
            output (Slot): preprocessed data of a single modal
        """
        raise NotImplementedError

    def group_key(self, slot: Slot) -> ModalityType:
        """
        **group_key** returns a key for reducing continuous modal.
        """
        return slot.modality

    @abstractmethod
    def group_map(slef, slots: List[Slot]) -> List[Slot]:
        """
        The **group_map** phase of the preprocessor takes a list of **Slot** as input, processes their values and outputs a list of **Slot**. This phase defines how to reduce continuous modes in a sample.
        """
        return slots

    @abstractmethod
    def collate(self, slots: List[Slot]) -> CollateOutput:
        """
        The **collate** phase of the preprocessor takes a batch of **Slot** as input, collate their values and outputs a **CollateOutput**. This phase defines how to collate a batch for a single modal.
        """
        raise NotImplementedError

    def postprocess(self, outputs, **sample):
        raise NotImplementedError


class SafeBasePreprocess(BasePreprocess):
    def __init__(
        self,
        global_dict,
        cfg: PreprocessConfig,
        modality_type: ModalityType,
        sanity_check: bool = True,
    ):
        super().__init__(global_dict, cfg)
        self._modality_type = modality_type
        self._sanity_check = sanity_check

    def map(self, slot: Slot) -> Slot:
        if self._sanity_check:
            assert slot.modality == self._modality_type
        return slot

    def group_map(self, slots: List[Slot]) -> List[Slot]:
        if self._sanity_check:
            # assert len(slots) > 1
            for i, slot in enumerate(slots):
                # assert slot.modality == self._modality_type
                assert slot.is_src == slots[0].is_src
                assert slot.global_position == i + slots[0].global_position
                assert slot.split == slots[0].split
        return slots

    def collate(self, slots: List[Slot]) -> CollateOutput:
        if self._sanity_check:
            assert len(slots) >= 1
            for slot in slots:
                # assert slot.modality == self._modality_type
                assert slot.is_src == slots[0].is_src
                assert slot.global_position == slots[0].global_position
                # assert slot.column_name == slots[0].column_name
                assert slot.attributes == slots[0].attributes
                assert slot.split == slots[0].split
        return None  # should not be used


class PreprocessSkipException(Exception):
    pass


@dataclass
class BaseCodePreprocessConfig(PreprocessConfig):
    code_dict_size: int = field(default=8192, metadata={"help": "code dict size"})
    code_entry_prefix: str = field(default='code', metadata={"help": "prefix of code entry in the global_dict"})
    use_encode: bool = field(default=True, metadata={"help": "where to use tokenizer.encode in map"})


class BaseCodePreprocess(SafeBasePreprocess):
    def __init__(self, global_dict: Dictionary, cfg: BaseCodePreprocessConfig, modality_type: ModalityType):
        super().__init__(global_dict, cfg, modality_type)
        self.num_codes = cfg.code_dict_size
        for i in range(self.num_codes):
            # global_dict.add_symbol("<{}_{}>".format(cfg.code_entry_prefix, i))
            global_dict.add_symbol(f"<code>_{i}")
        # get the start position of code entry in global dict
        # self.code_index_start = global_dict.index("<{}_0>".format(cfg.code_entry_prefix))
        self.code_index_start = global_dict.index("<code>_0")
        self.global_dict = global_dict
        self.cfg = cfg

    def map(self, slot: Slot) -> Slot:
        """
        Inputs:
            code: (`str` or `List` or `Tensor`) could be:
                A string separated by single-whitespaces like `6674 4336 4532 5334...` ;
                Tokens of a numpy or torch Tensor after user-defined preprocess

        Returns:
            `Torch.LongTensor`: 1-d int64 torch.Tensor
        """

        super().map(slot)
        if self.cfg.use_encode:
            code = self.encode(slot.value)
        else:
            code = slot.value
        if isinstance(code, np.ndarray) and np.issubdtype(code.dtype, np.integer) and code.ndim == 1:
            tokens = torch.LongTensor(code)
        elif (isinstance(code, torch.IntTensor) or isinstance(code, torch.LongTensor)) and code.ndim == 1:
            tokens = code.long()
        elif isinstance(code, str):
            tokens = self.split_str(code)
        else:
            raise ValueError("Incorrect input for code, only support string or 1-d int Tensor, " f"got {type(code)}")

        # TODO: add a parameter to control whether use these preprocess.

        if slot.get_attr('length') is not None:
            length = int(slot.get_attr('length'))
            tokens = tokens[:length]

        # add vocab size
        tokens = tokens + self.code_index_start
        slot.value = tokens
        return slot

    def collate(self, slots: List[Slot]) -> CollateOutput:
        """
        Inputs:
            samples: List of Tensors after preprocess

        Returns:
            dict:
                src_tokens (Tensor): batched tokens with shape `[batch, seq_len]`
        """
        super().collate(slots)
        if slots[0].is_src:
            slots[0].value = collate_tokens(
                [slot.value for slot in slots],
                pad_idx=self.global_dict.pad(),
                eos_idx=self.global_dict.eos(),
                pad_to_multiple=self.cfg.pad_to_multiple,
            )
            return CollateOutput(slots[0])
        else:
            input_value = collate_tokens(
                [slot.value[:-1] for slot in slots],
                pad_idx=self.global_dict.pad(),
                eos_idx=self.global_dict.eos(),
                pad_to_multiple=self.cfg.pad_to_multiple,
            )
            target_value = collate_tokens(
                [slot.value[1:] for slot in slots],
                pad_idx=self.global_dict.pad(),
                eos_idx=self.global_dict.eos(),
                pad_to_multiple=self.cfg.pad_to_multiple,
            )
            input_slot = Slot(
                slots[0].modality,
                slots[0].is_src,
                input_value,
                slots[0].global_position,
                slots[0].column_name,
                slots[0].attributes,
            )
            target_slot = Slot(
                slots[0].modality,
                slots[0].is_src,
                target_value,
                slots[0].global_position,
                slots[0].column_name,
                slots[0].attributes,
            )

            # for lagecy compatible
            ntokens = target_slot.value.ne(self.global_dict.pad()).long().sum().item()
            extra_dict = {
                "target": target_slot.value,
                "ntokens": ntokens,
            }
            return CollateOutput(input_slot, target_slot, extra_dict)

    def split_str(self, tokens_str):
        tokens = [int(num) for num in tokens_str.strip().split()]
        return torch.LongTensor(tokens)

    @abstractmethod
    def encode(self, raw_input, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def decode(self, tokens: torch.Tensor, **kwargs):
        raise NotImplementedError
