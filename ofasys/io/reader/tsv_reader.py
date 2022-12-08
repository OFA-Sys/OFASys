# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import re
from typing import Any, Dict, Optional

from .base_reader import BaseReader


class TsvReader(BaseReader):
    """
    A Tsv reader wrapper that reads several columns from a BaseReader

    Args:
        reader:        BaseReader
        seperator:     The charactor that seperates columns, default: '\t'
        header:        Whether has header for tsv file.
        selected_cols: String of select some columes seperated by comma.
            if header is True, selected_cols are the column names;
            if header is False, selected_cols are the indices of column start on 0;
            if selected_cols is None, select all columns.
    """

    _HEADER_REGEX = re.compile('[_A-Za-z0-9]+')

    def __init__(
        self,
        reader: BaseReader,
        seperator: str = '\t',
        header: bool = False,
        selected_cols: Optional[str] = None,
        column2alias: Optional[Dict[Any, Any]] = None,
    ):
        super().__init__()
        self.reader = reader
        self.seperator = seperator
        self.header = header
        self.selected_cols = selected_cols
        self.column_names = selected_cols.split(',') if selected_cols is not None else None
        self.column2alias = column2alias
        self.cur_pos = 0

        assert selected_cols is not None and column2alias is not None

        if header:
            # fetch tsv header
            self.reader.open()
            headers = self.reader.read().split(self.seperator)
            self.reader.close()

            # check header sanity
            for header in headers:
                if not self._HEADER_REGEX.fullmatch(header):
                    raise ValueError('{} is not a valid column name'.format(header))

            # update selected_cols to a list of indices of columns
            if self.selected_cols is not None:
                col_ids = []
                for v in self.selected_cols.split(','):
                    if v not in headers:
                        raise ValueError('{} should be in column names ({})'.format(v, str(headers)))
                    col_ids.append(headers.index(v))

                self.selected_cols = col_ids

        else:
            # update selected_cols to a list of indices of columns
            if self.selected_cols is not None:
                self.selected_cols = [int(v) for v in self.selected_cols.split(',')]

    def __len__(self):
        if self.header:
            return len(self.reader) - 1
        else:
            return len(self.reader)

    def open(self):
        self.reader.open()
        self.cur_pos = 0
        if self.header:
            self.reader.seek(1)

    def seek(self, offset: int = 0):
        if offset > len(self):
            raise ValueError("file reader seek error: {} > {}".format(offset, len(self)))
        self.cur_pos = offset
        if self.header:
            self.reader.seek(offset + 1)
        else:
            self.reader.seek(offset)

    def read(self):
        if self.is_eof():
            raise EOFError("reach end of file.")
        data = self.reader.read()
        self.cur_pos += 1
        data = data.split(self.seperator)
        data_dict = {}
        for name, i in zip(self.column_names, self.selected_cols):
            data_dict[self.column2alias[name]] = data[i]
        return data_dict

    def is_eof(self):
        return self.cur_pos >= len(self)

    def close(self):
        self.reader.close()
