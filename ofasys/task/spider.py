# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import os
from dataclasses import dataclass, field
from typing import Any, Dict

from ofasys.configure import register_config
from ofasys.task.base import OFATask, TaskConfig
from ofasys.utils.file_utils import download_and_unzip


@dataclass
class SpiderConfig(TaskConfig):
    database_path: str = field(default=None, metadata={"help": 'locate the database path'})
    schema_serialization_with_nl: bool = field(
        default=False, metadata={"help": 'need to describe the structural data into natural language or not?'}
    )
    schema_serialization_with_db_content: bool = field(
        default=True, metadata={"help": 'need to contain database content or not?'}
    )
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
        # self.use_dummpy = False

    def preprocess(self, data: Dict[str, Any], split: str) -> Dict[str, Any]:
        src, tgt = data['src'], data['tgt']
        db_id = data['db_id']

        src = src.lower().replace('<unk>', 'unk')
        tgt = tgt.lower().replace('<unk>', 'unk')
        if not hasattr(self, "schema_cache"):
            self.schema_cache = {}
        if db_id not in self.schema_cache:
            from ofasys.utils.spider.get_tables import dump_db_json_schema

            self.schema_cache[db_id] = dump_db_json_schema(
                self.database_path + "/" + db_id + "/" + db_id + ".sqlite", db_id
            )
        seq2seq_dict = seq2seq_input(tgt, src, db_id, self.database_path, self.schema_cache[db_id], self.cfg)
        struct_in = seq2seq_dict["struct_in"]
        text_in = seq2seq_dict["text_in"]
        seq_out = seq2seq_dict["seq_out"]
        data['src'] = text_in
        data['database'] = struct_in
        data['tgt'] = seq_out
        data['db_struct'] = form_input_for_construction(tgt, src, db_id, self.database_path, self.schema_cache[db_id])
        return data


# ------read database
import random
import re
from typing import Dict, List

from torch.utils.data import Dataset

from ofasys.utils.spider.bridge_content_encoder import get_database_matches


def seq2seq_input(query, question, db_id, db_path, schema, args):
    ex = form_input_for_construction(query, question, db_id, db_path, schema)
    serialized_schema = spider_add_serialized_schema(ex, args)["serialized_schema"].strip()
    question, seq_out = spider_pre_process_one_function(ex, args)
    return {"struct_in": serialized_schema, "text_in": question, "seq_out": seq_out}


def spider_get_input(
    question: str,
    serialized_schema: str,
    prefix: str,
) -> str:
    return prefix + question.strip() + " " + serialized_schema.strip()


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
        # "db_column_names": [
        #     {"table_id": table_id, "column_name": column_name}
        #     for table_id, column_name in schema["column_names_original"]
        # ],
        "db_column_names": {
            "table_id": [table_id for table_id, column_name in schema["column_names_original"]],
            "column_name": [column_name for table_id, column_name in schema["column_names_original"]],
        },
        "db_column_types": schema["column_types"],
        "db_primary_keys": [{"column_id": column_id} for column_id in schema["primary_keys"]],
        # "db_foreign_keys": [
        #     {"column_id": column_id, "other_column_id": other_column_id}
        #     for column_id, other_column_id in schema["foreign_keys"]
        # ],
        "db_foreign_keys": {
            "column_id": [column_id for column_id, other_column_id in schema["foreign_keys"]],
            "other_column_id": [other_column_id for column_id, other_column_id in schema["foreign_keys"]],
        },
    }


def spider_add_serialized_schema(ex: dict, args) -> dict:
    if getattr(args, "schema_serialization_with_nl", False):
        serialized_schema = serialize_schema_natural_language(
            question=ex["question"],
            db_path=ex["db_path"],
            db_id=ex["db_id"],
            db_column_names=ex["db_column_names"],
            db_table_names=ex["db_table_names"],
            db_primary_keys=ex["db_primary_keys"],
            db_foreign_keys=ex["db_foreign_keys"],
            schema_serialization_with_db_content=args.schema_serialization_with_db_content,
            normalize_query=True,
        )
    else:
        serialized_schema = serialize_schema(
            question=ex["question"],
            db_path=ex["db_path"],
            db_id=ex["db_id"],
            db_column_names=ex["db_column_names"],
            db_table_names=ex["db_table_names"],
            schema_serialization_type="peteshaw",
            schema_serialization_randomized=False,
            schema_serialization_with_db_id=True,
            schema_serialization_with_db_content=args.schema_serialization_with_db_content,
            normalize_query=True,
        )
    return {"serialized_schema": serialized_schema}


def spider_pre_process_function(batch: dict, args):
    prefix = ""

    inputs = [
        spider_get_input(question=question, serialized_schema=serialized_schema, prefix=prefix)
        for question, serialized_schema in zip(batch["question"], batch["serialized_schema"])
    ]

    targets = [
        spider_get_target(
            query=query,
            db_id=db_id,
            normalize_query=True,
            target_with_db_id=getattr(args, "target_with_db_id", False),
        )
        for db_id, query in zip(batch["db_id"], batch["query"])
    ]

    return zip(inputs, targets)


def spider_pre_process_one_function(item: dict, args):
    prefix = ""

    seq_out = spider_get_target(
        query=item["query"],
        db_id=item["db_id"],
        normalize_query=True,
        target_with_db_id=getattr(args, "target_with_db_id", False),
    )

    return prefix + item["question"].strip(), seq_out


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


def serialize_schema_natural_language(
    question: str,
    db_path: str,
    db_id: str,
    db_column_names: Dict[str, str],
    db_table_names: List[str],
    db_primary_keys,
    db_foreign_keys,
    schema_serialization_with_db_content: bool = False,
    normalize_query: bool = True,
) -> str:
    overall_description = (
        f'{db_id} contains tables such as '
        f'{", ".join([table_name.lower() if normalize_query else table_name for table_name in db_table_names])}.'
    )
    table_description_primary_key_template = lambda table_name, primary_key: f'{primary_key} is the primary key.'
    table_description = (
        lambda table_name, column_names: f'Table {table_name} has columns such as {", ".join(column_names)}.'
    )
    value_description = (
        lambda column_value_pairs: f'{"".join(["The {} contains values such as {}.".format(column, value) for column, value in column_value_pairs])}'
    )
    foreign_key_description = (
        lambda table_1, column_1, table_2, column_2: f'The {column_1} of {table_1} is the foreign key of {column_2} of {table_2}.'
    )

    db_primary_keys = db_primary_keys["column_id"]
    db_foreign_keys = list(zip(db_foreign_keys["column_id"], db_foreign_keys["other_column_id"]))

    descriptions = [overall_description]
    db_table_name_strs = []
    db_column_name_strs = []
    value_sep = ", "
    for table_id, table_name in enumerate(db_table_names):
        table_name_str = table_name.lower() if normalize_query else table_name
        db_table_name_strs.append(table_name_str)
        columns = []
        column_value_pairs = []
        primary_keys = []
        for column_id, (x, y) in enumerate(zip(db_column_names["table_id"], db_column_names["column_name"])):
            if column_id == 0:
                continue
            column_str = y.lower() if normalize_query else y
            db_column_name_strs.append(column_str)
            if x == table_id:
                columns.append(column_str)
                if column_id in db_primary_keys:
                    primary_keys.append(column_str)
                if schema_serialization_with_db_content:
                    matches = get_database_matches(
                        question=question,
                        table_name=table_name,
                        column_name=y,
                        db_path=(db_path + "/" + db_id + "/" + db_id + ".sqlite"),
                    )
                    if matches:
                        column_value_pairs.append((column_str, value_sep.join(matches)))

        table_description_columns_str = table_description(table_name_str, columns)
        descriptions.append(table_description_columns_str)
        table_description_primary_key_str = table_description_primary_key_template(
            table_name_str, ", ".join(primary_keys)
        )
        descriptions.append(table_description_primary_key_str)
        if len(column_value_pairs) > 0:
            value_description_str = value_description(column_value_pairs)
            descriptions.append(value_description_str)

    for x, y in db_foreign_keys:
        # get the table and column of x
        x_table_name = db_table_name_strs[db_column_names["table_id"][x]]
        x_column_name = db_column_name_strs[x]
        # get the table and column of y
        y_table_name = db_table_name_strs[db_column_names["table_id"][y]]
        y_column_name = db_column_name_strs[y]
        foreign_key_description_str = foreign_key_description(x_table_name, x_column_name, y_table_name, y_column_name)
        descriptions.append(foreign_key_description_str)
    return " ".join(descriptions)


def serialize_schema(
    question: str,
    db_path: str,
    db_id: str,
    db_column_names: Dict[str, str],
    db_table_names: List[str],
    schema_serialization_type: str = "peteshaw",
    schema_serialization_randomized: bool = False,
    schema_serialization_with_db_id: bool = True,
    schema_serialization_with_db_content: bool = False,
    normalize_query: bool = True,
) -> str:
    if schema_serialization_type == "verbose":
        db_id_str = "Database: {db_id}. "
        table_sep = ". "
        table_str = "Table: {table}. Columns: {columns}"
        column_sep = ", "
        column_str_with_values = "{column} ({values})"
        column_str_without_values = "{column}"
        value_sep = ", "
    elif schema_serialization_type == "peteshaw":
        # see https://github.com/google-research/language/blob/master/language/nqg/tasks/spider/append_schema.py#L42
        db_id_str = " | {db_id}"
        table_sep = ""
        table_str = " | {table} : {columns}"
        column_sep = " , "
        column_str_with_values = "{column} ( {values} )"
        column_str_without_values = "{column}"
        value_sep = " , "
    else:
        raise NotImplementedError

    def get_column_str(table_name: str, column_name: str) -> str:
        column_name_str = column_name.lower() if normalize_query else column_name
        if schema_serialization_with_db_content:
            matches = get_database_matches(
                question=question,
                table_name=table_name,
                column_name=column_name,
                db_path=(db_path + "/" + db_id + "/" + db_id + ".sqlite"),
            )
            if matches:
                return column_str_with_values.format(column=column_name_str, values=value_sep.join(matches))
            else:
                return column_str_without_values.format(column=column_name_str)
        else:
            return column_str_without_values.format(column=column_name_str)

    tables = [
        table_str.format(
            table=table_name.lower() if normalize_query else table_name,
            columns=column_sep.join(
                map(
                    lambda y: get_column_str(table_name=table_name, column_name=y[1]),
                    filter(
                        lambda y: y[0] == table_id,
                        zip(
                            db_column_names["table_id"],
                            db_column_names["column_name"],
                        ),
                    ),
                )
            ),
        )
        for table_id, table_name in enumerate(db_table_names)
    ]
    if schema_serialization_randomized:
        random.shuffle(tables)
    if schema_serialization_with_db_id:
        serialized_schema = db_id_str.format(db_id=db_id) + table_sep.join(tables)
    else:
        serialized_schema = table_sep.join(tables)
    return serialized_schema


def _get_schemas(examples: Dataset) -> Dict[str, dict]:
    schemas: Dict[str, dict] = dict()
    for ex in examples:
        if ex["db_id"] not in schemas:
            schemas[ex["db_id"]] = {
                "db_table_names": ex["db_table_names"],
                "db_column_names": ex["db_column_names"],
                "db_column_types": ex["db_column_types"],
                "db_primary_keys": ex["db_primary_keys"],
                "db_foreign_keys": ex["db_foreign_keys"],
            }
    return schemas


"""
    Wrap the raw dataset into the seq2seq one.
    And the raw dataset item is formatted as
    {
        "query": sample["query"],
        "question": sample["question"],
        "db_id": db_id,
        "db_path": db_path,
        "db_table_names": schema["table_names_original"],
        "db_column_names": [
            {"table_id": table_id, "column_name": column_name}
            for table_id, column_name in schema["column_names_original"]
        ],
        "db_column_types": schema["column_types"],
        "db_primary_keys": [{"column_id": column_id} for column_id in schema["primary_keys"]],
        "db_foreign_keys": [
            {"column_id": column_id, "other_column_id": other_column_id}
            for column_id, other_column_id in schema["foreign_keys"]
        ],
    }
    """
