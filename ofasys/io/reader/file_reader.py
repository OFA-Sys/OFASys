# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import json
import logging
import os
import pickle

import numpy as np

from ofasys.utils import OFA_CACHE_HOME, file_utils

from .base_reader import BaseReader
from .utils import line_locate


class FileLineReader(BaseReader):
    """
    A text reader that reads lines one-by-one from a local file_path.

    Args:
        file_path(str):     The local file path.
        cached_index(bool): Whether caches the begining position of lines.
                            It will leave a file named by `file_path.cache`.
    """

    def __init__(self, file_path: str, cached_index: bool = True):
        super().__init__()
        self.file_path = os.path.abspath(file_path)
        self.cached_index = cached_index
        self.reader = None

        assert os.path.isfile(self.file_path), 'Local file {} not exists!'.format(self.file_path)
        if self.cached_index:
            self.line_pos = self._get_cached_line_pos()
        else:
            self.line_pos = line_locate(self.file_path)
        self.n = self.line_pos.shape[0]
        self.cur_pos = 0

    def __len__(self):
        return self.n

    def open(self):
        assert self.reader is None
        self.cur_pos = 0
        self.reader = open(self.file_path, 'rb')

    def seek(self, offset: int = 0) -> None:
        if offset > len(self):
            raise ValueError("file reader seek error: {} > {}".format(offset, self.n))
        self.cur_pos = offset
        # hack: if offset equals self.n, it will throw an error, so we set seek(0), but cur_pos equals offset
        self.reader.seek(self.line_pos[offset % len(self)])

    def read(self) -> str:
        if self.is_eof():
            raise EOFError("reach end of file")
        data = self.reader.readline()
        data = data.decode('utf-8').rstrip('\r\n')
        self.cur_pos += 1
        return data

    def close(self) -> None:
        if self.reader is not None:
            self.reader.close()
            self.reader = None

    def is_eof(self):
        return self.cur_pos >= len(self)

    @file_utils.local_file_lock(os.path.join(OFA_CACHE_HOME, 'reader.lock'))
    def _get_cached_line_pos(self):
        cache_path = "{}.index".format(self.file_path)
        if os.path.exists(cache_path):
            total_row_count, lineid_to_offset = pickle.load(open(cache_path, "rb"))
            assert total_row_count == len(lineid_to_offset) and lineid_to_offset[0] == 0
            lineid_to_offset = np.array(lineid_to_offset, dtype=np.int64)
            return lineid_to_offset

        cache_file_name = file_utils.url_to_filename(self.file_path)
        cache_file_path = os.path.join(file_utils.OFA_CACHE_HOME, cache_file_name + '.cache')
        meta_file_path = os.path.join(file_utils.OFA_CACHE_HOME, cache_file_name + '.json')

        if os.path.exists(cache_file_path):
            with open(cache_file_path, 'rb') as f:
                line_pos = np.load(f)
            logging.debug('Load cached line_pos from {}'.format(cache_file_path))
        else:
            line_pos = line_locate(self.file_path)
            with open(cache_file_path, 'wb') as f:
                np.save(f, line_pos)
            logging.debug('Save cached line_pos to {}'.format(cache_file_path))
            with open(meta_file_path, 'w') as f:
                f.write(json.dumps({"url": self.file_path, "etag": None, "cache": True}))
        self.cache_file_path = cache_file_path

        return line_pos
