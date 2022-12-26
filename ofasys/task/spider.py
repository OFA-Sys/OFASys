# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import os
import re
import torch
from dataclasses import dataclass, field
from typing import Any, Dict

import torch

from ofasys.configure import register_config
from ofasys.task.base import OFATask, TaskConfig
from ofasys.utils.file_utils import download_and_unzip


@dataclass
class SpiderConfig(TaskConfig):
    database_path: str = field(default=None, metadata={"help": 'locate the database path'})
    target_with_db_id: bool = field(default=False, metadata={"help": 'does the target contain database id?'})


@register_config("ofasys.task", "spider", dataclass=SpiderConfig)
class SpiderTask(OFATask):
    def __init__(self, cfg: SpiderConfig, **kwargs):
        super().__init__(cfg, **kwargs)
        from ofasys.utils.spider.fetch_nltk import fetch_nltk_data

        fetch_nltk_data()
        database_path = cfg.database_path
        if database_path is None:
            paths = self.cfg.dataset.train_data.split(',')
            database_path = "_".join(paths[-1].split("_")[:-1]) + "/database/"
        elif database_path.startswith("oss://"):
            if database_path.endswith(".zip"):
                absolute_path = download_and_unzip(database_path, "dataset/spider_data/")
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
            else:
                assert False
            database_path = os.path.join(absolute_path, "spider/database")
        self.database_path = database_path

    def preprocess(self, data: Dict[str, Any], split: str) -> Dict[str, Any]:
        src, tgt = data['src'], data['tgt']
        db_id = data['db_id']
        if db_id == "baseball_1":
            print("the long dataset baseball_1 is not del")
            return None
        src = src.lower().replace('<unk>', 'unk')
        tgt = tgt.lower().replace('<unk>', 'unk')
        if not hasattr(self, "schema_cache"):
            self.schema_cache = {}
        if db_id not in self.schema_cache:
            from ofasys.utils.spider.get_tables import dump_db_json_schema

            self.schema_cache[db_id] = dump_db_json_schema(
                self.database_path + "/" + db_id + "/" + db_id + ".sqlite", db_id
            )
        seq_out = spider_get_target(
            query=tgt,
            db_id=db_id,
            normalize_query=True,
            target_with_db_id=self.cfg.target_with_db_id,
        )
        data['src'] = src.strip()
        data['database'] = form_input_for_construction(None, src, db_id, self.database_path, self.schema_cache[db_id])
        data['tgt'] = seq_out
        data['db_struct'] = form_input_for_construction(tgt, src, db_id, self.database_path, self.schema_cache[db_id])
        return data


def spider_get_target(
    query: str,
    db_id: str,
    normalize_query: bool,
    target_with_db_id: bool,
) -> str:
    _normalize = normalize if normalize_query else (lambda x: x)
    return f"{db_id} | {_normalize(query)}" if target_with_db_id else _normalize(query)


def form_input_for_construction(query, question, db_id, db_path, schema):
    return {
        "query": query,
        "question": question,
        "db_id": db_id,
        "db_path": db_path,
        "db_table_names": schema["table_names_original"],
        "db_column_names": {
            "table_id": [table_id for table_id, column_name in schema["column_names_original"]],
            "column_name": [column_name for table_id, column_name in schema["column_names_original"]],
        },
        "db_column_types": schema["column_types"],
        "db_primary_keys": [{"column_id": column_id} for column_id in schema["primary_keys"]],
        "db_foreign_keys": {
            "column_id": [column_id for column_id, other_column_id in schema["foreign_keys"]],
            "other_column_id": [other_column_id for column_id, other_column_id in schema["foreign_keys"]],
        },
    }


def normalize(query: str) -> str:
    def comma_fix(s):
        # Remove spaces in front of commas
        return s.replace(" , ", ", ")

    def white_space_fix(s):
        # Remove double and triple spaces
        return " ".join(s.split())

    def lower(s):
        # Convert everything except text between (single or double) quotation marks to lower case
        return re.sub(r"\b(?<!['\"])(\w+)(?!['\"])\b", lambda match: match.group(1).lower(), s)

    return comma_fix(white_space_fix(lower(query)))

