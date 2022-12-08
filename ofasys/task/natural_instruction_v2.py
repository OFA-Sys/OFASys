# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import pickle
import random
import string
from dataclasses import dataclass, field
from typing import Any, Dict

from ofasys.configure import register_config
from ofasys.task.base import OFATask, TaskConfig
from ofasys.utils.file_utils import cached_path


@dataclass
class NaturalInstructionTaskConfig(TaskConfig):
    pos_example_num: int = field(default=0, metadata={"help": "postive emamples number"})
    neg_example_num: int = field(default=0, metadata={"help": "negative emamples number"})
    add_task_name: bool = field(default=False, metadata={"help": "whether to add task name"})


@register_config("ofasys.task", "natural_instruction_v2", dataclass=NaturalInstructionTaskConfig)
class NaturalInstructionTask(OFATask):
    def __init__(self, cfg: NaturalInstructionTaskConfig, **kwargs):
        super().__init__(cfg, **kwargs)
        with open(cached_path('oss://ofasys/data/natural_instructions/task_info.pkl'), 'rb') as f:
            self.task_info = pickle.load(f)
        self.cfg = cfg

    def preprocess(self, data: Dict[str, Any], split: str) -> Dict[str, Any]:
        src, tgt = data['src'], data['tgt']
        if src == None or tgt == None or src == "" or tgt == "":
            return None
        task_info = self.task_info[data['task_name']]

        src = src.lower().replace('<unk>', 'unk')
        tgt = tgt.lower().replace('<unk>', 'unk')
        prompt = random.choice(task_info['def'])

        # example num
        pos_example_num = self.cfg.pos_example_num
        neg_example_num = self.cfg.neg_example_num

        pos_example = task_info['pos']
        neg_example = task_info['neg']

        if pos_example_num > len(pos_example):
            pos_example_num = len(pos_example)
        if neg_example_num > len(neg_example):
            neg_example_num = len(neg_example)

        pos_res = random.sample(pos_example, pos_example_num)
        neg_res = random.sample(neg_example, neg_example_num)

        task_input = ""

        # add the input first
        task_input += "Now complete the following example -\n"
        task_input += f"Input: {src.strip()}"
        if not task_input[-1] in string.punctuation:
            task_input += "."
        task_input += "\n"
        task_input += "Output: "

        task_name = ""
        if self.cfg.add_task_name:
            task_name += data['task_name'] + ". "

        definition = ""
        definition = "Definition: " + prompt.strip()
        if not definition[-1] in string.punctuation:
            definition += "."
        definition += "\n\n"

        # add positive examples
        pos_examples = []
        for idx, pos_example in enumerate(pos_res):
            pos_example_str = f" Positive Example {idx+1} -\n"
            pos_example_str += f"Input: {pos_example['input'].strip()}"
            if not pos_example_str[-1] in string.punctuation:
                pos_example_str += "."
            pos_example_str += "\n"
            pos_example_str += f"Output: {pos_example['output'].strip()}"
            if not pos_example_str[-1] in string.punctuation:
                pos_example_str += "."
            pos_example_str += "\n"
            pos_examples.append(pos_example_str)
            if (
                len((definition + " ".join(pos_examples) + pos_example_str + task_input).split(" "))
                > self.cfg.max_src_length
            ):
                break

        # add negative examples.
        neg_examples = []
        for idx, neg_example in enumerate(neg_res):
            neg_example_str = f" Negative Example {idx+1} -\n"
            neg_example_str += f"Input: {neg_example['input'].strip()}"
            if not neg_example_str[-1] in string.punctuation:
                neg_example_str += "."
            neg_example_str += "\n"
            neg_example_str += f"Output: {neg_example['output'].strip()}"
            if not neg_example_str[-1] in string.punctuation:
                neg_example_str += "."
            neg_example_str += "\n"
            neg_examples.append(neg_example_str)
            if (
                len(
                    (
                        definition + " ".join(pos_examples) + " ".join(neg_examples) + neg_example_str + task_input
                    ).split(" ")
                )
                > self.cfg.max_src_length
            ):
                break
        data['src'] = task_name + definition + "".join(pos_examples) + "".join(neg_examples) + task_input
        data['tgt'] = tgt
        # to unify bleu metric where references are more than one
        if split != 'train':
            data['ref_list'] = tgt.split("|&*|")
        return data
