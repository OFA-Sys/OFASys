# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from typing import Any, Dict

from ofasys.configure import register_config
from ofasys.task.base import OFATask, TaskConfig


@register_config("ofasys.task", "dart", dataclass=TaskConfig)
class DartTask(OFATask):
    def preprocess(self, data: Dict[str, Any], split: str = "train") -> Dict[str, Any]:
        if split == 'test':
            data['tgt'] = ''
            data['src'] = data['database']
        src, tgt = data['src'], data['tgt']

        src, tgt = input_reformat(src, tgt)
        data['database'] = src  # a table list
        data['tgt'] = tgt  # text
        # to unify bleu metric where references are more than one
        data['ref_list'] = tgt.split("&&")
        return data


def input_reformat(src, tgt):
    # to re-format the input structure data into list
    # if there need any process the let tripleset perfomed better write here
    tripleset = src.lower().replace('<unk>', 'unk')
    text = tgt.lower().replace('<unk>', 'unk')
    tripleset_list = tripleset.split("|")
    table_list = []
    for idx in range(len(tripleset_list)):
        if "randolph : 25" in tripleset_list[idx]:
            obj, key, _, _ = tripleset_list[idx].split(" : ")
            val = "randolph : 25"
        elif "2-10 : 0-14" in tripleset_list[idx]:
            obj, key, _, _ = tripleset_list[idx].split(" : ")
            val = "2-10 : 0-14"
        else:
            obj, key, val = tripleset_list[idx].split(" : ")
        table_list.append([obj, key, val])
    return table_list, text
