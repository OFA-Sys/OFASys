# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import re
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
        src = src.lower().replace('<unk>', 'unk')
        tgt = tgt.lower().replace('<unk>', 'unk')

        seq2seq_dict = seq2seq_input(src, tgt)
        struct_in = seq2seq_dict["struct_in"]
        seq_out = seq2seq_dict["seq_out"]
        data['database'] = struct_in
        data['tgt'] = seq_out
        # to unify bleu metric where references are more than one
        data['ref_list'] = seq_out.split("&&")
        return data


def seq2seq_input(tripleset, text):
    # if there need any process the let tripleset perfomed better write here
    tripleset_list = tripleset.split("|")
    for idx in range(len(tripleset_list)):
        if len(tripleset_list[idx].split(" : ")) != 3:
            print(tripleset_list[idx])
        if "randolph : 25" in tripleset_list[idx]:
            obj, key, _, _ = tripleset_list[idx].split(" : ")
            val = "randolph : 25"
        elif "2-10 : 0-14" in tripleset_list[idx]:
            obj, key, _, _ = tripleset_list[idx].split(" : ")
            val = "2-10 : 0-14"
        else:
            obj, key, val = tripleset_list[idx].split(" : ")
        obj = get_nodes(obj)
        val = get_nodes(val)
        key = get_relation(key)
        edge_split = camel_case_split(key)
        key = " ".join(edge_split)
        tripleset_list[idx] = " : ".join([obj, key, val])
    tripleset = " | ".join(tripleset_list)
    return {"struct_in": tripleset, "seq_out": text}


def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    d = [m.group(0) for m in matches]
    new_d = []
    for token in d:
        token = token.replace('(', '')
        token_split = token.split('_')
        for t in token_split:
            new_d.append(t)
    return new_d


def get_nodes(n):
    n = n.strip()
    n = n.replace('(', '')
    n = n.replace('\"', '')
    n = n.replace(')', '')
    n = n.replace(',', ' ')
    n = n.replace('_', ' ')
    # n = ' '.join(re.split('(\W)', n))
    # n = unidecode.unidecode(n)
    # n = n.lower()

    return n


def get_relation(n):
    n = n.replace('(', '')
    n = n.replace(')', '')
    n = n.strip()
    n = n.split()
    n = "_".join(n)
    n = n.lower()
    return n
