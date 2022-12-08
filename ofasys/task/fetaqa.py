# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from typing import Any, Dict

from ofasys.configure import register_config
from ofasys.task.base import OFATask, TaskConfig


@register_config("ofasys.task", "fetaqa", dataclass=TaskConfig)
class FetaqaTask(OFATask):
    def preprocess(self, data: Dict[str, Any], split: str) -> Dict[str, Any]:
        table, question, answer = data['database'], data['src'], data['tgt']
        question = question.lower().replace('<unk>', 'unk')
        answer = answer.lower().replace('<unk>', 'unk')

        seq2seq_dict = seq2seq_input(table, question, answer, self.cfg)
        struct_in = seq2seq_dict["struct_in"]
        text_in = seq2seq_dict["text_in"]
        seq_out = seq2seq_dict["seq_out"]
        data['src'] = text_in
        data['database'] = struct_in
        data['tgt'] = seq_out
        # to unify bleu metric where references are more than one
        data['ref_list'] = seq_out.split("&&")
        return data


def seq2seq_input(table, question, answer, args):
    # if there need any process the let tripleset perfomed better write here
    return {"struct_in": table, "text_in": question, "seq_out": answer}


def preprocess_tsv(paths, data_dir):
    import jsonlines

    def concat_tbl(rows):
        seq = " [col] "
        seq += " ".join(rows[0])

        for i, row in enumerate(rows[1:]):
            seq += " | [row"
            seq += str(i)
            seq += "] "
            seq += " : ".join(row)
        return seq

    for path in paths:
        rows = []
        jsonl_path = data_dir + path.split("/")[-1].split(".")[0].split("_")[1] + ".jsonl"
        with open(jsonl_path, encoding="utf-8") as f:
            for line in jsonlines.Reader(f):
                question = line["question"]
                answer = line["answer"]
                table = line["table_page_title"] + " " + line["table_section_title"] + concat_tbl(line["table_array"])
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
