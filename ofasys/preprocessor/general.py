# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

from ofasys import ModalityType
from ofasys.configure import ConfigStore, auto_import

from .default.base import BasePreprocess, PreprocessSkipException
from .dictionary import Dictionary
from .instruction import Instruction, Slot
from .utils import collate_others, group_by_predicator

auto_import(__file__)
PreprocessConfig = ConfigStore().make_dataclass(
    "ofasys.preprocess",
    "PreprocessConfig",
    __name__,
    ['text', 'category', 'image', 'image_vqgan', 'box', 'audio', 'phone'],
)

default_preprocess = {
    ModalityType.TEXT: 'text',
    ModalityType.IMAGE: 'image',
    ModalityType.BOX: 'box',
    ModalityType.AUDIO: 'audio',
    ModalityType.PHONE: 'phone',
    ModalityType.VIDEO: 'video',
    ModalityType.STRUCT: 'table',
}


class GeneralPreprocess:
    def __init__(self, cfg: PreprocessConfig, global_dict: Dictionary):
        self.global_dict = global_dict
        self.name2pre: Dict[str, BasePreprocess] = self.get_name2pre(cfg)

    def get_name2pre(self, cfg):
        name2pre = {}
        for pre_name in cfg.__annotations__:
            node = ConfigStore().get("ofasys.preprocess", pre_name)
            node_cfg = getattr(cfg, pre_name) if hasattr(cfg, pre_name) else node.config
            if node_cfg.is_active:
                name2pre[pre_name] = node.target(self.global_dict, node_cfg)
        return name2pre

    @property
    def bos(self):
        return self.global_dict.bos()

    @property
    def eos(self):
        return self.global_dict.eos()

    @property
    def pad(self):
        return self.global_dict.pad()

    @property
    def bpe(self):
        return self.name2pre['text'].bpe

    def prepare_for_generation(self, closed_set, **kwargs):
        self.name2pre["text"].prepare_for_generation(closed_set, **kwargs)

    def get_preprocess(self, slot: Slot) -> BasePreprocess:
        if slot.get_attr('preprocess'):
            return self.name2pre[slot.get_attr('preprocess')]
        else:
            return self.name2pre[default_preprocess[slot.modality]]

    def __call__(self, ist_data: Optional[Instruction]):
        if ist_data is None:
            return None
        try:
            # slot.preprocess.instruction_map
            visited_preprocessors = set()
            for slot in ist_data.slots:
                pre = self.get_preprocess(slot)
                if pre not in visited_preprocessors:
                    ist_data = pre.instruction_map(ist_data)
                    visited_preprocessors.add(pre)

            # slot.preprocess.map
            slots = [self.get_preprocess(slot).map(slot) for slot in ist_data.slots]
        except PreprocessSkipException:
            return None

        # slot.preprocess.group_map
        def predicator(slot1: Slot, slot2: Slot):
            return (
                self.get_preprocess(slot1).group_key(slot1) == self.get_preprocess(slot2).group_key(slot2)
                and slot1.is_src == slot2.is_src
            )

        group_slots = group_by_predicator(slots, predicator)
        group_slots = [
            self.name2pre[default_preprocess[self.get_preprocess(group[0]).group_key(group[0])]].group_map(group)
            if len(group) > 1
            else self.get_preprocess(group[0]).group_map(group)
            for group in group_slots
        ]
        slots = [slot for group in group_slots for slot in group]

        # reset global position
        for i, slot in enumerate(slots):
            slot.global_position = i
        ist_data.slots = slots
        return ist_data

    def collate(self, samples: List[Instruction]) -> Dict:
        if len(samples) == 0:
            return {}
        for i in range(1, len(samples)):
            if len(samples[i].slots) != len(samples[0].slots):
                raise ValueError("Do not support to batch various modality slot.")

        result = {
            "net_input": {
                "slots": [],
            },
            "net_target": {
                "slots": [],
            },
            "nsentences": len(samples),
            "template": samples[0].template,
        }
        for i in range(len(samples[0].slots)):
            collate_output = self.get_preprocess(samples[0].slots[i]).collate([ist.slots[i] for ist in samples])
            if collate_output.net_input_slot:
                result["net_input"]["slots"].append(collate_output.net_input_slot)
            if collate_output.net_target_slot:
                result["net_target"]["slots"].append(collate_output.net_target_slot)
            if collate_output.sample_extra:
                result.update(collate_output.sample_extra)

        for key in samples[0].others.keys():
            data = [ist.others[key] for ist in samples]
            result[key] = collate_others(data)
        return result

    def postprocess(self, outputs, **sample):
        target_slot = Slot.get_target_slot_from_sample(sample)
        processor = self.get_preprocess(target_slot)
        try:
            return processor.postprocess(outputs, **sample)
        except NotImplementedError:
            if target_slot.get_attr('preprocess'):
                preprocessor_name = target_slot.get_attr('preprocess')
            else:
                preprocessor_name = default_preprocess[target_slot.modality]
            raise NotImplementedError(
                f"{preprocessor_name} preprocessor has no postprocess function, but it is used for postprocessing."
            )
