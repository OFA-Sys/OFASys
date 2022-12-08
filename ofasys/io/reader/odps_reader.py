# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import logging
import sys
import time
from typing import Any, Dict, Optional

from .base_reader import BaseReader

logger = logging.getLogger(__name__)

_no_common_io_help = "No common_io found for ODPSReader"


class ODPSReader(BaseReader):
    def __init__(
        self,
        table_path,
        selected_cols="",
        common_io_num_threads=1,
        common_io_capacity=1024,
        column2alias: Optional[Dict[Any, Any]] = None,
    ):
        super().__init__()
        self.table_path = table_path
        self.selected_cols = selected_cols
        self.column_names = selected_cols.split(",")
        self.column2alias = column2alias
        self.common_io_num_threads = common_io_num_threads
        self.common_io_capacity = common_io_capacity

        try:
            import common_io
        except ImportError:
            logger.error(_no_common_io_help)
            exit(1)

        assert selected_cols is not None and column2alias is not None

        with common_io.table.TableReader(self.table_path, num_threads=0, capacity=1) as reader:
            self.n = reader.get_row_count()
        self.reader = None
        self.cur_pos = 0

    def __len__(self):
        return self.n

    def open(self):
        assert self.reader is None
        self.cur_pos = 0
        import common_io

        self.reader = common_io.table.TableReader(
            self.table_path,
            selected_cols=self.selected_cols,
            num_threads=self.common_io_num_threads,
            capacity=self.common_io_capacity,
        )

    def seek(self, offset: int = 0) -> None:
        if offset > len(self):
            raise ValueError("odps reader seek error: {} > {}".format(offset, len(self)))
        self.cur_pos = offset
        self.reader.seek(offset % len(self))

    def read(self) -> Dict[Any, Any]:
        if self.is_eof():
            raise EOFError("reach end of file.")

        retry_cnt = 0
        retry_total = 20
        while True:
            try:
                data = self.reader.read(1)[0]
                break
            except Exception as e:
                if 'Read table time out' in str(e) and retry_cnt < retry_total:
                    time.sleep(1)
                    retry_cnt += 1
                    print('Read table time out: Retry {}/{}'.format(retry_cnt, retry_total))
                else:
                    raise e

        data_dict = {}
        for i, name in enumerate(self.column_names):
            d = data[i]
            data_dict[self.column2alias[name]] = d.decode(encoding='utf8') if isinstance(d, bytes) else d

        self.cur_pos += 1
        return data_dict

    def is_eof(self):
        return self.cur_pos >= len(self)

    def close(self):
        if hasattr(self, 'reader') and self.reader is not None:
            self.reader.close()
            self.reader = None
