# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from typing import Any, Dict
import re

from ofasys.configure import register_config
from ofasys.task.base import OFATask, TaskConfig
pattern = re.compile("\[table name\] (.+) \[table head\] (.+) \[table content\] (.+)")


@register_config("ofasys.task", "fetaqa", dataclass=TaskConfig)
class FetaqaTask(OFATask):
    def preprocess(self, data: Dict[str, Any], split: str) -> Dict[str, Any]:
        table, question, answer = data['database'], data['src'], data['tgt']
        question = question.lower().replace('<unk>', 'unk')
        answer = answer.lower().replace('<unk>', 'unk')

        table_name, table_head, table_content = pattern.findall(table)[0]
        table_content = table_content.lower()
        table_content = [[col for col in row.split(" : ")] for row in table_content.split(" | ")]
        data['src'] = question.lower()
        data['database'] = {
            "table_name": table_name,
            "table_head": table_head,
            "table_content": table_content
        }
        data['tgt'] = answer.lower()
        # to unify bleu metric where references are more than one
        data['ref_list'] = answer.lower().split("&&")
        return data


def preprocess_tsv(paths, data_dir):

    def concat_tbl(rows):
        seq = " [table head] "
        seq += " : ".join(rows[0])
        seq += " [table content] "
        for i, row in enumerate(rows[1:]):
            for idx, _row in enumerate(row):
                if " : " in _row or " | " in _row:
                    row[idx] = _row.replace(" : ", ":")
                    row[idx] = _row.replace(" | ", "|")
            seq += " : ".join(row)
            seq += " | "
        return seq

    for path in paths:
        rows = []
        # jsonl_path = data_dir + path.split("/")[-1].split(".")[0].split("_")[1] + ".jsonl"
        jsonl_path = data_dir + path.split("/")[-1].split(".")[0].split("_")[1] + ".jsonl"
        with open(jsonl_path, encoding="utf-8") as f:
            for line in jsonlines.Reader(f):
                question = line["question"]
                answer = line["answer"]
                table = f"[table name] {line['table_page_title']} {line['table_section_title']} {concat_tbl(line['table_array'])}"

                rows.append([table, question, answer])

            print("FeTaQA data example: ", path)
            print(rows[0])
            write_to_tsv(path, rows)

    print("the tsv files are ready!")


def write_to_tsv(output_path: str, data: list):
    def _txt_process(txt):
        txt = txt.replace("\n", "")
        txt = txt.replace("\r", "")
        txt = txt.replace("\t", " ")
        txt = txt.replace("  ", " ")
        if "\t" in txt:
            print("txt", txt)
            assert False
        return txt

    with open(output_path, 'w') as f:
        for sample in data:
            f.write('\t'.join([_txt_process(x) for x in sample]) + "\n")