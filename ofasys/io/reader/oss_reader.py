# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import os
from io import BytesIO
from typing import Any, Dict, Optional

import numpy as np

from ofasys.utils.oss import oss_exists, oss_get, oss_size

from .base_reader import BaseReader
from .utils import FifoLineReader


class OssLineReader(BaseReader):
    def __init__(self, file_path: str, buffer_capacity=64):
        super().__init__()
        if not file_path.startswith('oss://'):
            raise ValueError(
                '{} only support a path like oss://xxx'.format(
                    self.__class__.__name__,
                )
            )
        if not oss_exists(file_path):
            raise ValueError('OSS file {} not exists!'.format(file_path))

        # check cache file: from oss://xxx/yy.tsv?host=zz to oss://xxx/.yy.tsv.cache?host=zz
        basename = os.path.basename(file_path)
        basename = basename.split('?')
        basename[0] = '.' + basename[0] + '.cache'
        basename = '?'.join(basename)
        cache_path = os.path.join(os.path.dirname(file_path), basename)
        if not oss_exists(cache_path):
            raise ValueError('OSS file {} not exists!'.format(cache_path))

        fin = oss_get(cache_path)
        self.line_pos = np.load(BytesIO(fin.read()))
        if hasattr(fin, 'close'):
            fin.close()

        self.file_path = file_path
        self.buffer_capacity = buffer_capacity
        self.n = self.line_pos.shape[0]
        self.byte_size = oss_size(self.file_path)
        self.reader = None
        self.cur_pos = 0

    def __len__(self):
        return self.n

    def open(self):
        self.cur_pos = 0
        self.seek(0)

    def seek(self, offset: int = 0) -> None:
        if offset > len(self):
            raise ValueError("file reader seek error: {} > {}".format(offset, len(self)))
        if self.reader is not None and hasattr(self.reader, 'close'):
            self.reader.close()
        # hack: line_pos may exceed
        start_pos = int(self.line_pos[offset % len(self)])
        self.reader = FifoLineReader(
            oss_get(self.file_path, byte_range=(start_pos, self.byte_size - 1)),
            self.buffer_capacity,
        )
        self.cur_pos = offset

    def read(self) -> str:
        if self.is_eof():
            raise EOFError("reach end of file.")
        data = self.reader.readline()
        data = data.decode('utf-8').rstrip('\r\n')
        self.cur_pos += 1
        return data

    def is_eof(self):
        return self.cur_pos >= len(self)

    def close(self) -> None:
        if hasattr(self, 'reader') and self.reader is not None:
            self.reader.close()
            self.reader = None


class OssTextBinReader(BaseReader):
    def __init__(
        self,
        file_path: str,
        buffer_capacity=64,
        max_sentence_length=1024,
        column2alias: Optional[Dict[Any, Any]] = None,
    ):
        super().__init__()
        if not file_path.startswith('oss://'):
            raise ValueError(
                '{} only support a path like oss://xxx'.format(
                    self.__class__.__name__,
                )
            )

        self.file_path = file_path
        if not oss_exists(self.file_path):
            raise ValueError('OSS file {} not exists!'.format(self.file_path))

        self.column2alias = column2alias
        assert column2alias is not None

        self.buffer_capacity = buffer_capacity
        self.max_sentence_length = max_sentence_length
        self.byte_size = oss_size(self.file_path)
        assert self.byte_size % 2 == 0
        self.n = self.byte_size // 2 // max_sentence_length
        self.reader = None
        self.cur_pos = 0

    def __len__(self):
        return self.n

    def open(self):
        self.cur_pos = 0
        self.seek(0)

    def seek(self, offset: int = 0) -> None:
        if offset > len(self):
            raise ValueError("file reader seek error: {} > {}".format(offset, len(self)))
        elif offset == len(self):
            offset = 0
        if self.reader is not None and hasattr(self.reader, 'close'):
            self.reader.close()
        start_pos = offset * 2 * self.max_sentence_length
        self.reader = FifoLineReader(
            oss_get(self.file_path, byte_range=(start_pos, self.byte_size - 1)),
            self.buffer_capacity,
        )
        self.cur_pos = offset

    def read(self) -> Dict[Any, Any]:
        if self.is_eof():
            raise EOFError("reach end of file.")
        data = self.reader.readn(self.max_sentence_length * 2)
        data = np.frombuffer(data, dtype=np.uint16)
        data = data.astype(np.int32)
        self.cur_pos += 1
        return {self.column2alias['text']: data}

    def is_eof(self):
        return self.cur_pos >= len(self)

    def close(self) -> None:
        if hasattr(self, 'reader') and self.reader is not None:
            self.reader.close()
            self.reader = None
