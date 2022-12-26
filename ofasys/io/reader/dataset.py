# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import math
import platform
import warnings
from typing import Callable

import torch
import torch.utils.data
from datasets import Dataset, IterableDataset, concatenate_datasets, load_dataset

from .base_reader import BaseReader
from .cached_reader import CachedReader
from .concat_reader import ConcatReader
from .file_reader import FileLineReader
from .mixed_reader import MixedReader
from .odps_reader import ODPSReader
from .oss_reader import OssLineReader, OssTextBinReader
from .tsv_reader import TsvReader
from .utils import (
    parse_dataset_paths,
    parse_sample_ratios,
    parse_selected_cols,
    partition_data_size,
    set_seed,
)


class HfDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        dataset: Dataset,
        group: torch.distributed.ProcessGroup = None,
        batch_size: int = 1,
        offset: int = 0,
        process_fn: Callable = None,
        collate_fn: Callable = None,
    ):
        super().__init__()
        self.dataset: Dataset = dataset

        self.batch_size = batch_size
        self.offset = offset
        self.collate_fn = collate_fn if collate_fn is not None else lambda x: x
        self.process_fn = process_fn if process_fn is not None else lambda x: x

        if torch.distributed.is_initialized():
            shard_id = torch.distributed.get_rank(group)
            num_shards = torch.distributed.get_world_size(group)
            # todo: set contiguous=True in order to be compatible with previous version
            self.dataset = self.dataset.shard(num_shards=num_shards, index=shard_id, contiguous=True)

        self.num_steps = math.ceil(len(self.dataset) / self.batch_size)

    def __len__(self):
        return self.num_steps

    def __iter__(self):
        dataset = self.skip(self.dataset, offset=self.offset * self.batch_size)

        batch_data = []
        for data in dataset:
            batch_data.append(self.process_fn(data))
            if len(batch_data) == self.batch_size:
                yield self.collate_fn(batch_data)
                batch_data = []

        if len(batch_data) > 0:
            yield self.collate_fn(batch_data)

    def skip(self, dataset, offset=0):
        if offset > 0:
            dataset = dataset[offset:]
        return dataset

    def seek(self, offset=0):
        if offset > self.num_steps:
            raise IndexError
        self.offset = offset


class ReaderDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        reader: BaseReader,
        group: torch.distributed.ProcessGroup = None,
        batch_size: int = 1,
        offset: int = 0,
        process_fn: Callable = None,
        collate_fn: Callable = None,
    ):
        super().__init__()
        assert batch_size >= 1
        self.reader = reader
        self.global_data_size = len(reader)

        self.batch_size = batch_size
        self.offset = offset
        self.collate_fn = collate_fn if collate_fn is not None else lambda x: x
        self.process_fn = process_fn if process_fn is not None else lambda x: x

        if torch.distributed.is_initialized():
            self.rank_id = torch.distributed.get_rank(group)
            self.rank_size = torch.distributed.get_world_size(group)
        else:
            self.rank_id, self.rank_size = 0, 1

        self.rank_start, self.rank_end = partition_data_size(self.global_data_size, self.rank_id, self.rank_size)
        self.rank_data_size = self.rank_end - self.rank_start

        self.num_steps = math.ceil(math.ceil(self.global_data_size / self.rank_size) / batch_size)

    def __len__(self):
        return self.num_steps

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id, worker_size = 0, 1
        else:
            worker_id, worker_size = worker_info.id, worker_info.num_workers

        step_start, step_end = partition_data_size(self.num_steps, worker_id, worker_size, False)
        step_start += self.offset
        assert step_start <= step_end

        data_start = step_start * self.batch_size + self.rank_start
        data_end = min(self.rank_end, step_end * self.batch_size + self.rank_start)
        self.reader.open()
        self.reader.seek(data_start)

        idx = data_start
        for step in range(step_start, step_end):
            batch_data = []
            for i in range(self.batch_size):
                if idx < data_end:
                    data = self.reader.read()
                    data = self.process_fn(data)
                    if data is not None:
                        batch_data.append(data)
                    idx += 1
            yield self.collate_fn(batch_data)
        self.reader.close()

    def seek(self, offset=0):
        if offset > self.num_steps:
            raise IndexError
        self.offset = offset


class CountingIterator:
    def __init__(self, iterable, offset=0):
        self.itr = iter(iterable)
        self.n = offset
        self.total = len(iterable)

    def __len__(self):
        return self.total

    def __iter__(self):
        return self

    def __next__(self):
        if not self.has_next():
            raise StopIteration
        try:
            x = next(self.itr)
        except StopIteration:
            raise IndexError(f"Iterator expected to have length {self.total}, " "but exhausted at position {self.n}.")
        self.n += 1
        return x

    def has_next(self):
        return self.n < self.total

    def skip(self, n):
        for _ in range(n):
            next(self)
        return self


class GroupedIterator(CountingIterator):
    def __init__(self, iterable, chunk_size):
        self.itr = iterable
        self.chunk_size = chunk_size
        self.n = int(math.ceil(iterable.n / float(chunk_size)))
        self.total = int(math.ceil(len(iterable) / float(chunk_size)))

    def __next__(self):
        if not self.has_next():
            raise StopIteration
        chunk = []
        for _ in range(self.chunk_size):
            if self.itr.has_next():
                chunk.append(next(self.itr))
            else:
                break
        self.n += 1
        return chunk


class EpochBatchIterator:
    def __init__(
        self,
        cfg,
        data_paths=None,
        dataset=None,
        split="train",
        process_fn=None,
        collate_fn=None,
        update_freq=None,
        batch_size=1,
        num_workers=0,
        prefetch_factor=0,
        group=None,
        seed=1,
        epoch=1,
        shuffle=False,
    ):
        # Other system may raise unpickable exception.
        if platform.system().lower() != "linux" and num_workers > 0:
            # TODO: add warning or set default num_workers is None
            num_workers = 0
        if update_freq is not None and not isinstance(update_freq, list):
            raise ValueError("update_freq must be a list of int.")

        self.cfg = cfg
        self.split = split
        self.hf_dataset = None

        assert data_paths is not None or dataset is None, "Neither data_paths nor dataset arguments are set"

        if data_paths:
            self.data_paths = parse_dataset_paths(data_paths)
            self.selected_cols, self.colname2alias = parse_selected_cols(cfg.selected_cols)

            if split == "train":
                sample_ratios = parse_sample_ratios(cfg.sample_ratios)
                if isinstance(sample_ratios, (float, int)):
                    self.sample_ratios = [1.0] * len(self.data_paths)
                    self.sample_ratios[0] = sample_ratios
                else:
                    self.sample_ratios = sample_ratios
            else:
                self.sample_ratios = [1.0] * len(self.data_paths)
        else:
            self.hf_dataset = dataset

        self.process_fn = process_fn
        self.collate_fn = collate_fn
        self.update_freq = update_freq
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.group = group
        self.seed = seed
        self.epoch = max(epoch, 1)
        self.shuffle = shuffle
        self._dataset = None
        self._persist_reader = None
        self._cur_epoch_itr = self._get_iterator_for_epoch()

    def total_num_updates(self, max_epoch: int) -> int:
        return sum(
            math.ceil(len(self._dataset) / self.update_freq[i])
            if i < len(self.update_freq)
            else math.ceil(len(self._dataset) / self.update_freq[-1])
            for i in range(max_epoch)
        )

    @property
    def cur_epoch_itr(self):
        return self._cur_epoch_itr

    def next_epoch(self):
        self.epoch += 1
        self._cur_epoch_itr = self._get_iterator_for_epoch()
        return self._cur_epoch_itr

    @property
    def end_of_epoch(self) -> bool:
        return not self._cur_epoch_itr.has_next()

    @property
    def iterations_in_epoch(self):
        """The number of consumed batches in the current epoch."""
        if self._cur_epoch_itr is not None:
            return self._cur_epoch_itr.n
        return 0

    @property
    def epoch_str(self) -> str:
        if self._cur_epoch_itr is not None:
            return f'{self.epoch}[{self._cur_epoch_itr.n}/{self._cur_epoch_itr.total}]'
        else:
            return 'None'

    def state_dict(self):
        """Returns a dictionary containing a whole state of the iterator."""
        if self.end_of_epoch:
            epoch = self.epoch + 1
            iter_in_epoch = 0
        else:
            epoch = self.epoch
            iter_in_epoch = self.iterations_in_epoch
        return {
            "version": 'ofasys',
            "epoch": epoch,
            "iterations_in_epoch": iter_in_epoch,
        }

    def load_state_dict(self, state_dict):
        self.epoch = state_dict["epoch"]
        itr_pos = state_dict.get("iterations_in_epoch", 0)
        self._cur_epoch_itr = self._get_iterator_for_epoch(itr_pos)

    def _get_reader(self):

        if self._persist_reader is not None:
            self._persist_reader.reset(self.epoch, self.shuffle)
            return self._persist_reader

        readers = []
        for data_path in self.data_paths:
            epoch_data_path = data_path[(self.epoch - 1) % len(data_path)]
            if epoch_data_path.startswith('odps://'):
                reader = ODPSReader(
                    epoch_data_path,
                    selected_cols=self.selected_cols,
                    common_io_num_threads=self.cfg.common_io_num_threads,
                    common_io_capacity=self.cfg.common_io_capacity,
                    column2alias=self.colname2alias,
                )
            elif epoch_data_path.startswith('oss://'):
                if epoch_data_path.endswith('.bin'):
                    reader = OssTextBinReader(
                        epoch_data_path,
                        buffer_capacity=self.cfg.oss_buffer_capacity,
                        max_sentence_length=self.cfg.text_bin_length,
                        column2alias=self.colname2alias,
                    )
                else:
                    reader = TsvReader(
                        OssLineReader(
                            epoch_data_path,
                            buffer_capacity=self.cfg.oss_buffer_capacity,
                        ),
                        header=self.cfg.header,
                        selected_cols=self.selected_cols,
                        seperator=self.cfg.seperator,
                        column2alias=self.colname2alias,
                    )
            else:
                reader = TsvReader(
                    FileLineReader(epoch_data_path),
                    header=self.cfg.header,
                    selected_cols=self.selected_cols,
                    seperator=self.cfg.seperator,
                    column2alias=self.colname2alias,
                )

            readers.append(reader)

        # ConcatReader can be used to repeat or down-sample a dataset even if there is only one dataset.
        if len(readers) == 1 and tuple(self.sample_ratios) == (1,):
            reader = readers[0]
        else:
            reader = MixedReader(readers, self.sample_ratios, self.cfg.interleaved_multiple_reader)

        if self.cfg.cached:
            self._persist_reader = CachedReader(reader)
            self._persist_reader.reset(self.epoch, self.shuffle)
            return self._persist_reader

        return reader

    def _get_dataset(self, streaming=False):
        if self.hf_dataset is not None:
            return self.hf_dataset

        dataset_list = []
        for data_path in self.data_paths:
            epoch_data_path = data_path[(self.epoch - 1) % len(data_path)]
            if epoch_data_path.endswith(("csv", "tsv")):
                dataset = load_dataset(
                    "csv",
                    data_files=epoch_data_path,
                    sep="\t",
                    header=0 if self.cfg.header else None,
                    streaming=streaming,
                )["train"]
            elif epoch_data_path.endswith(("json", "jsonl")):
                dataset = load_dataset("json", data_files=epoch_data_path, streaming=streaming)["train"]
            else:
                raise NotImplementedError
            dataset_list.append(dataset)

        if len(dataset_list) > 1:
            dataset = concatenate_datasets(dataset_list)
        else:
            dataset = dataset_list[0]

        for original_column_name, new_column_name in self.colname2alias.items():
            dataset = dataset.rename_column(original_column_name, new_column_name)

        if streaming:
            dataset = dataset.with_format("torch")

        self.hf_dataset = dataset
        return dataset

    def _get_iterator_for_epoch(self, offset=0):
        if self.hf_dataset is not None or self.cfg.use_hf_datasets:
            hf_dataset = self._get_dataset(streaming=False)
            assert isinstance(hf_dataset, Dataset), f"{hf_dataset}(IterableDataset) is not support yet."

            if self.shuffle:
                hf_dataset = hf_dataset.shuffle(self.epoch + self.seed)
            self._dataset = HfDataset(
                hf_dataset,
                group=self.group,
                batch_size=self.batch_size,
                process_fn=self.process_fn,
                collate_fn=self.collate_fn,
            )
        else:
            reader = self._get_reader()
            self._dataset = ReaderDataset(
                reader,
                group=self.group,
                batch_size=self.batch_size,
                process_fn=self.process_fn,
                collate_fn=self.collate_fn,
            )

        if self.update_freq is None:
            update_freq = 1
        elif self.epoch <= len(self.update_freq):
            update_freq = self.update_freq[self.epoch - 1]
        else:
            update_freq = self.update_freq[-1]
        self._dataset.seek(offset * update_freq // (self.num_workers if self.num_workers > 0 else 1))

        if self.num_workers == 0 and self.prefetch_factor != 2:
            warnings.warn(
                'prefetch_factor option could only be specified in multiprocessing. '
                'Let num_workers > 0 to enable multiprocessing. See '
                'https://github.com/pytorch/pytorch/blob/master/torch/utils/data/dataloader.py#L236.'
            )
            self.prefetch_factor = 2

        def worker_init_fn(worker_id):
            set_seed(self.seed + worker_id)

        itr = torch.utils.data.DataLoader(
            self._dataset,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor,  # support only when pytorch >= 1.7.0
            worker_init_fn=worker_init_fn,
        )
        itr = CountingIterator(itr, offset=offset * update_freq)
        if self.update_freq is None:
            return itr
        return GroupedIterator(itr, chunk_size=update_freq)
