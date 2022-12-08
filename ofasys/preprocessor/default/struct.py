# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from dataclasses import dataclass, field

from ofasys.configure import register_config

from ..instruction import ModalityType, Slot
from .base import CollateOutput, PreprocessConfig, SafeBasePreprocess
from .text import DefaultTextPreprocess, TextPreprocessConfig


@dataclass
class StructPreprocessConfig(TextPreprocessConfig):
    row_seperator: str = field(default=' |', metadata={"help": "row seperator or struct data"})
    col_seperator: str = field(default=' : ', metadata={"help": "col seperator or struct data"})

    def __post_init__(self):
        self.is_active = True


@register_config("ofasys.preprocess", "struct", StructPreprocessConfig)
class DefaultStructPreprocess(DefaultTextPreprocess):
    def map(self, slot: Slot) -> Slot:
        assert isinstance(slot.value, list)
        data = []
        for row in slot.value:
            assert isinstance(row, list)
            for col in row:
                assert isinstance(col, str)
            row = self.cfg.col_seperator.join(map(str.strip, row))
            data.append(row)
        data = self.cfg.row_seperator.join(map(str.strip, data))
        slot.value = data
        return super().map(slot)

    def group_key(self, slot: Slot):
        return ModalityType.TEXT
