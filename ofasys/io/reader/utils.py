# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import os
import random
import re
import random
import subprocess
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch


def line_locate_py(file_path: str) -> np.array:
    pos = [0]
    with open(file_path, 'rb') as f:
        for line in f:
            if line[-1] == 10:  # 10 is the ascii of '\n'
                n = pos[-1] + len(line)
                pos.append(n)
    pos.pop()
    return np.array(pos, dtype=np.int64)


def line_locate(file_path: str) -> np.array:
    """
    Return a array of the starting position of all lines of a text file.

    Args:
        file_path: A path of text file.

    Returns:
        np.array(np.int64): The starting position of all lines.
    """
    exe_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.line_locate.out'))
    if not os.path.exists(exe_path):
        os.system('cd {}; make'.format(os.path.dirname(__file__)))
    status, res = subprocess.getstatusoutput('{} {}'.format(exe_path, file_path))
    if status != 0:
        print('fail to use c++ mmap to build index, transfer to python way')
        return line_locate_py(file_path)
    assert status == 0, res
    n_line, pos = res.strip().split('\n')
    n_line = int(n_line)
    pos = list(map(int, pos.split()))
    assert n_line == len(pos)
    pos = np.array(pos, dtype=np.int64)
    return pos


def partition_data_size(
    data_size: int,
    slice_id: int,
    slice_count: int,
    left_prior: bool = True,
) -> Tuple[int, int]:
    """
    Partition `data_size` into `slice_count` parts, the max diff per part is 1.
    Return a tuple of [begin, end)

    Args:
        data_size:   The total number of data.
        slice_id:    Indicate which part will be computed.
        slice_count: The number of slices.
        left_prior:  If set True, guarantee the number on the left is always bigger,
                     and vice versa.

    Returns:
        start: The start position of slice, inclusive.
        end:   The end position of slice, exclusive.
    """
    assert data_size > 0 and slice_count > 0 and slice_id >= 0
    assert slice_id < slice_count
    size = int(data_size / slice_count)
    split_point = data_size % slice_count
    if left_prior:
        if slice_id < split_point:
            start = slice_id * (size + 1)
            end = start + (size + 1)
        else:
            start = split_point * (size + 1) + (slice_id - split_point) * size
            end = start + size
    else:
        split_point = slice_count - split_point
        if slice_id < split_point:
            start = slice_id * size
            end = start + size
        else:
            start = split_point * size + (slice_id - split_point) * (size + 1)
            end = start + (size + 1)
    return start, end


_PATH_ALT = re.compile(r'(\[\d+-\d+\])')


def parse_dataset_paths(data_paths: str) -> List[List[str]]:
    """
    Extend dataset path str to a list, for example:
    Input:
        "./train_data_[1-3],train_data_5 ||| extra_train_data"
    Output:
        [["./train_data_1", "./train_data_2", "./train_data_3", "train_data_5"], ["extra_train_data"]]
    """
    paths = []
    for path in data_paths.split('|||'):
        sub_paths = []
        for sub_path in path.strip().split(','):
            mat = _PATH_ALT.findall(sub_path)
            if len(mat) == 0:
                sub_paths.append(sub_path)
            elif len(mat) == 1:
                start, end = tuple(map(int, mat[0].strip('[]').split('-')))
                for i in range(start, end + 1):
                    sub_paths.append(_PATH_ALT.sub(str(i), sub_path))
            else:
                raise ValueError(f"only one expansion is supported, get {sub_path}")
        if len(sub_paths) > 0:
            paths.append(sub_paths)
    return paths


def parse_sample_ratios(sample_ratios: Union[str, float]):
    if isinstance(sample_ratios, str):
        sample_ratios = eval(sample_ratios)
    return sample_ratios


def parse_selected_cols(selected_cols: str) -> (str, Dict[str, str]):
    """
    Parse selected_columes to a list of raw colume names and a dict of colume alias map, for example:
    Input:
        "0:v1, k2:v2, 3, k4"
    Output:
        raw_column_names: "0,k2,3,k4"
        col_name_alias: {
            "0": "v1",
            "k2": "v2",
            "3": "3",
            "k4": "k4",
        }
    """
    if selected_cols is None:
        raise ValueError("Must give selected_columes for current readers")
    selected_cols = selected_cols.replace(' ', '')
    keys = []
    i2name = {}
    for i, kv in enumerate(selected_cols.split(',')):
        try:
            k, v = kv.split(':')
        except ValueError:
            k, v = kv, kv
        keys.append(k)
        i2name[k] = v
    return ','.join(keys), i2name


def parse_template(template: Optional[str]) -> Optional[List[str]]:
    if template is None:
        return None
    return [t.strip() for t in template.split('|||')]


class FifoQueue:
    """FIFO Queue, see http://users.ece.utexas.edu/~valvano/embed/chap7/fifo.gif"""

    def __init__(self):
        self.buf = BytesIO()
        self.available = 0  # Bytes available for reading
        self.size = 0  # Total size of buffer
        self.write_fp = 0  # Write pointer

    def read(self, size=None):
        """Reads size bytes from buffer"""
        if size is None or size > self.available:
            size = self.available
        size = max(size, 0)

        result = self.buf.read(size)
        self.available -= size

        if len(result) < size:
            self.buf.seek(0)
            result += self.buf.read(size - len(result))
        return result

    def write(self, data):
        """Appends data to buffer"""
        if self.size < self.available + len(data):
            # Expand buffer
            new_buf = BytesIO()
            new_buf.write(self.read())
            self.write_fp = self.available = new_buf.tell()
            read_fp = 0
            while self.size <= self.available + len(data):
                self.size = max(self.size, 1024) * 2
            new_buf.write(b'0' * (self.size - self.write_fp))
            self.buf = new_buf
        else:
            read_fp = self.buf.tell()

        self.buf.seek(self.write_fp)
        written = self.size - self.write_fp
        self.buf.write(data[:written])
        self.write_fp += len(data)
        self.available += len(data)
        if written < len(data):
            self.write_fp -= self.size
            self.buf.seek(0)
            self.buf.write(data[written:])
        self.buf.seek(read_fp)


class FifoLineReader(FifoQueue):
    def __init__(self, reader, buffer_capacity=64):
        """
        Args:
            reader: file-object which has read() interface
            buffer_capacity: initial capacith of buffer, unit: Kb
        """
        super().__init__()
        self.reader = reader
        self.buffer_capacity = buffer_capacity

    def readline(self):
        while True:
            idx = self.buf.tell()
            if idx == self.size:
                idx = 0
            view = self.buf.getbuffer()
            for i in range(self.available):
                if view[idx] == 10:
                    del view
                    return self.read(i + 1)
                idx += 1
                if idx == self.size:
                    idx = 0
            del view
            reader_data = self.reader.read(1024 * self.buffer_capacity)
            if reader_data:
                self.write(reader_data)
            else:
                break
        return self.read()

    def readn(self, size):
        while True:
            if size <= self.available:
                return self.read(size)
            reader_data = self.reader.read(1024 * self.buffer_capacity)
            if reader_data:
                self.write(reader_data)
            else:
                break
        return self.read()

    def close(self):
        if hasattr(self.reader, 'close'):
            self.reader.close()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
