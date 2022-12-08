# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import bisect
import copy
from typing import List, Union

from .base_reader import BaseReader


class ConcatReader(BaseReader):
    """
    A reader wrapper that merges multiple readers into a single reader.
    The read order remains the same, including both inter-/intra-reader order.

    Args:
        readers(List[BaseReader]): a list of readers to merge.
        sample_ratios(Union[float, List[float]]): sampling ratios of each reader.
            when `sample_ratios` equals to 1, all data will be read and read only once.
            when `sample_ratios` is greater than 1, the first reader will be sampled `sample_ratios` times.
            when `sample_ratios` is less than 1, the first reader will only take previous `sample_ratios`*100% data.
    """

    @staticmethod
    def cumsum(sequence, sample_ratios):
        r, s = [], 0
        for e, ratio in zip(sequence, sample_ratios):
            curr_len = int(ratio * len(e))
            r.append(curr_len + s)
            s += curr_len
        return r

    def __init__(self, readers: List[BaseReader], sample_ratios=1.0):
        assert len(readers) > 0, "readers should not be an empty iterable"
        self.readers = list(readers)
        if isinstance(sample_ratios, float):
            sample_ratios = [sample_ratios] * len(self.readers)
        assert len(readers) == len(sample_ratios), "lengths of readers and sample_ratios don't match"
        self.sample_ratios = sample_ratios
        self.cumulative_sizes = self.cumsum(self.readers, sample_ratios)
        self.real_sizes = [len(d) for d in self.readers]

    def __len__(self):
        return self.cumulative_sizes[-1]

    def open(self):
        self.cur_pos = 0
        for reader in self.readers:
            reader.open()

    def seek(self, offset: int = 0):
        if offset > len(self):
            raise ValueError("file reader seek error: {} > {}".format(offset, len(self)))
        self.cur_pos = offset
        reader_idx, sample_idx = self._find_position(offset % len(self))
        for i, reader in enumerate(self.readers):
            if i < reader_idx:
                reader.seek(self.real_sizes[i])
            elif i == reader_idx:
                reader.seek(sample_idx)
            else:
                reader.seek(0)

    def read(self):
        if self.is_eof():
            raise EOFError("reach end of file.")
        reader_idx, sample_idx = self._find_position(self.cur_pos)
        if sample_idx == 0:
            self.readers[reader_idx].seek(0)

        data = self.readers[reader_idx].read()
        self.cur_pos += 1
        return data

    def close(self):
        for reader in self.readers:
            reader.close()

    def is_eof(self):
        return self.cur_pos >= len(self)

    def _find_position(self, idx):
        reader_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if reader_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[reader_idx - 1]
        sample_idx = sample_idx % self.real_sizes[reader_idx]
        return reader_idx, sample_idx
