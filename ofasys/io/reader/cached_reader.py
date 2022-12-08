# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from typing import Any

import numpy as np
import tqdm

from .base_reader import BaseReader


class CachedReader(BaseReader):
    def __init__(self, reader: BaseReader):
        self.epoch_data = []
        self.cur_pos = 0
        reader.open()
        # TODO: only fetch the worker's data
        for i in tqdm.tqdm(range(len(reader)), desc='cached reader'):
            self.epoch_data.append(reader.read())
        reader.close()

    def reset(self, epoch, shuffle=True):
        if shuffle:
            np.random.RandomState(epoch).shuffle(self.epoch_data)
        self.cur_pos = 0

    def __len__(self) -> int:
        return len(self.epoch_data)

    def open(self) -> None:
        self.cur_pos = 0

    def seek(self, offset: int = 0) -> None:
        if offset > len(self):
            raise ValueError("file reader seek error: {} > {}".format(offset, len(self)))
        self.cur_pos = offset

    def read(self) -> Any:
        if self.is_eof():
            raise EOFError("reach end of file.")
        item = self.epoch_data[self.cur_pos]
        self.cur_pos += 1
        return item

    def is_eof(self):
        return self.cur_pos >= len(self)
