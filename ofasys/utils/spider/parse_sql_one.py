# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import json
import os
import random
import re
import sqlite3
import sys
import traceback
from collections import OrderedDict
from os import listdir, makedirs
from os.path import exists, isdir, isfile, join, split, splitext

from nltk import tokenize, word_tokenize

from ..process_sql import get_sql
from .schema import Schema, get_schemas_from_json

if __name__ == "__main__":

    sql = "SELECT name ,  country ,  age FROM singer ORDER BY age DESC"
    db_id = "concert_singer"
    table_file = "tables.json"

    schemas, db_names, tables = get_schemas_from_json(table_file)
    schema = schemas[db_id]
    table = tables[db_id]
    schema = Schema(schema, table)
    sql_label = get_sql(schema, sql)
