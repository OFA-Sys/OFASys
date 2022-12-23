# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from dataclasses import dataclass, field
from typing import Any, Dict

from ofasys.configure import register_config
from ofasys.task.base import OFATask, TaskConfig


@dataclass
class SudokuConfig(TaskConfig):
    seg_embedding: bool = field(
        default=False, metadata={"help": "using segement embeddings?"}
    )  # TODO: add segment embedding to model


@register_config("ofasys.task", "sudoku", dataclass=SudokuConfig)
class SudokuTask(OFATask):
    def __init__(self, cfg: SudokuConfig, **kwargs):
        super().__init__(cfg, **kwargs)
        self.seg_embedding = cfg.seg_embedding
        if self.seg_embedding:  # no use
            self.input_puzzle_row = []
            self.input_puzzle_col = []
            for idx in range(9):
                for jdx in range(9):
                    self.input_puzzle_row.append(jdx + 1)
                    self.input_puzzle_col.append(idx + 1)
                    if not (idx == 8 and jdx == 8):
                        self.input_puzzle_row.append(0)
                        self.input_puzzle_col.append(0)

    def preprocess(self, data: Dict[str, Any], split: str) -> Dict[str, Any]:
        src, tgt = data['src'], data['tgt']
        case_id, mask_ratio = data["uid"], data["mask_ratio"]
        if isinstance(mask_ratio, str):
            mask_ratio = float(mask_ratio)

        input_puzzle = input_reformat(src)
        output_ans = input_reformat(tgt)

        data['src'] = input_puzzle
        data['tgt'] = output_ans
        data["target_field"] = [mask_ratio, input_puzzle, output_ans]
        return data


def input_reformat(input_puzzle):
    input_puzzle = input_puzzle.lower().replace('<unk>', 'unk')
    input_puzzle = ' '.join(input_puzzle.lower().strip().split())
    list_puzzle = []
    for row in input_puzzle.split(" | "):
        list_puzzle.append([col for col in row.split(" : ")])
    return list_puzzle
