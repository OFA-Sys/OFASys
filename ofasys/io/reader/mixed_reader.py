# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import bisect
import copy
from typing import List, Union

import numpy as np

from .base_reader import BaseReader


class MixedReader(BaseReader):
    """
    A reader wrapper that merges multiple readers into a single reader.
    The order remains the same inside each reader.

    Args:
        readers(List[BaseReader]): a list of readers to merge.
        sample_ratios(Union[float, List[float]]): sampling ratios of each reader.
            < 1.0: only task previes `sample_ratios` data.
            > 1.0: take `sample_ratios` times sequencially.
            = 1.0: read once only for all data
        interleaved:
            True (default): interleave all reader's data
                v1: [1, 2, 3]  v2: [a, b, c, d, e, f]
                return: [1, a, b, 2, c, d, 3, e, f]
            False: concat all reader's data
                v1: [1, 2, 3]  v2: [a, b, c, d, e, f]
                return: [1, 2, 3, a, b, c, d, e, f]
    """

    def __init__(self, readers: List[BaseReader], sample_ratios=1.0, interleaved=True):
        assert len(readers) > 0, "readers should not be an empty iterable"
        assert len(readers) < 128, "number of readers must not exceed 128."
        self.readers = list(readers)
        if isinstance(sample_ratios, float):
            sample_ratios = [sample_ratios] * len(self.readers)
        assert len(readers) == len(sample_ratios), "lengths of readers and sample_ratios don't match"
        self.sample_ratios = sample_ratios

        # reader_indices holds the sequences of readers to call read
        self.reader_indices = self._build_reader_indices(self.readers, sample_ratios, interleaved)
        # real_sizes holds the total size of each reader
        self.real_sizes = [len(d) for d in self.readers]
        # cur_sizes holds current position of each reader
        self.cur_sizes = [0 for d in self.readers]

    def __len__(self):
        return len(self.reader_indices)

    def open(self):
        self.cur_pos = 0
        for reader in self.readers:
            reader.open()

    def seek(self, offset: int = 0):
        if offset > len(self):
            raise ValueError("file reader seek error: {} > {}".format(offset, len(self)))
        self.cur_pos = offset

        self.cur_sizes = [0 for _ in self.readers]
        for i in range(self.cur_pos):
            self.cur_sizes[self.reader_indices[i]] += 1

        for i, reader in enumerate(self.readers):
            self.cur_sizes[i] %= self.real_sizes[i]
            reader.seek(self.cur_sizes[i])

    def read(self):
        if self.is_eof():
            raise EOFError("reach end of file.")

        reader_idx = self.reader_indices[self.cur_pos]
        data = self.readers[reader_idx].read()

        self.cur_sizes[reader_idx] += 1
        if self.cur_sizes[reader_idx] == self.real_sizes[reader_idx]:
            self.cur_sizes[reader_idx] = 0
            self.readers[reader_idx].seek(0)

        self.cur_pos += 1
        return data

    def close(self):
        for reader in self.readers:
            reader.close()

    def is_eof(self):
        return self.cur_pos >= len(self)

    @staticmethod
    def _build_reader_indices(sequence, sample_ratios, interleaved=False):
        n_sequences = len(sequence)
        reader_count = np.empty(n_sequences, dtype=np.int64)
        for i in range(len(sequence)):
            reader_count[i] = int(len(sequence[i]) * sample_ratios[i])
        total_count = reader_count.sum()

        # NOTE: int8 limits the maximum number of readers to 128
        reader_indices = np.empty(total_count, dtype=np.int8)

        if interleaved:
            left_count = reader_count.copy()
            left_ratio = np.ones(n_sequences, dtype=np.float64)
            for i in range(total_count):
                # TODO: if necessary, use a stable priority queue to reduce the complexity
                max_idx = left_ratio.argmax()
                reader_indices[i] = max_idx
                left_count[max_idx] -= 1
                left_ratio[max_idx] = left_count[max_idx] / reader_count[max_idx]
            assert np.all(left_ratio == 0)
            return reader_indices
        else:
            s = 0
            for i, cr in enumerate(reader_count):
                reader_indices[s : s + cr] = i
                s += cr
            return reader_indices
