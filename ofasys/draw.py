# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import argparse
import math
import os
import re
from dataclasses import dataclass, fields, make_dataclass
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

train_regex = re.compile(
    r'INFO: epoch (\d+):\s+(\d+)\s+/\s+(\d+)\s+'
    r'loss=([-0-9.e]+)(?:.*?)'
    r'ups=([-0-9.e]+)(?:.*?)'
    r'bsz=([-0-9.e]+)(?:.*?)'
    r'num_updates=([-0-9.e]+)(?:.*?)'
    r'lr=([-0-9.e]+)(?:.*?)'
    r'gnorm=([-0-9.e]+)(?:.*?)'
    r'loss_scale=([-0-9.e]+)'
)


@dataclass
class TrainRecord:
    epoch: float
    loss: float
    ups: float
    bsz: float
    num_updates: int
    lr: float
    gnorm: float
    loss_scale_log2: int


def parse_train_record(line: str) -> Optional[TrainRecord]:
    mat = train_regex.search(line)
    if mat is None:
        return None
    mat = mat.groups()
    return TrainRecord(
        epoch=float(int(mat[0]) - 1 + int(mat[1]) / int(mat[2])),
        loss=float(mat[3]),
        ups=float(mat[4]),
        bsz=float(mat[5]),
        num_updates=int(mat[6]),
        lr=float(mat[7]),
        gnorm=float(mat[8]),
        loss_scale_log2=int(math.log2(int(mat[9]))),
    )


def dc_list2numpy(dc_list, dc_class):
    kwargs = {}
    for field in fields(dc_class):
        kwargs[field.name] = np.array([getattr(dc, field.name) for dc in dc_list])
    return dc_class(**kwargs)


valid_regex_str = r"INFO: epoch \d+ \| valid on 'valid' subset \| " r'loss ([-0-9.e]+)'


def parse_file(log_file: str, valid_metrics: List[str] = []):
    valid_regex = valid_regex_str
    metric_fields = []
    for metric_name in valid_metrics:
        valid_regex += r'(?:.*?)\| ' + metric_name + r' ([-0-9.e]+)'
        metric_fields.append((metric_name, float))
    valid_regex += r'(?:.*?)\| num_updates (\d+)'
    valid_regex = re.compile(valid_regex)

    ValidRecord = make_dataclass('ValidRecord', [('num_updates', int), ('loss', float)] + metric_fields)

    train_records, valid_records = [], []
    with open(log_file) as fin:
        for line in fin:
            record = parse_train_record(line)
            if record is not None:
                train_records.append(record)

            mat = valid_regex.search(line)
            if mat is not None:
                mat = mat.groups()
                kwargs = {metric_name: float(mat[i + 1]) for i, metric_name in enumerate(valid_metrics)}
                valid_records.append(
                    ValidRecord(
                        num_updates=int(mat[-1]),
                        loss=float(mat[0]),
                        **kwargs,
                    )
                )

    train_records = dc_list2numpy(train_records, TrainRecord)
    valid_records = dc_list2numpy(valid_records, ValidRecord)
    return train_records, valid_records


guess_regex = re.compile(r'valid on (?:.*?)ppl [-0-9.e]+ \| ((?:[-A-Za-z0-9_]+ [-0-9.e]+ \| )*)wps')


def guess_valid_metrics(log_file: str):
    with open(log_file) as f:
        for line in f:
            mat = guess_regex.search(line)
            if mat is not None:
                valid_metrics = [v.split()[0] for v in mat.group(1).strip(' |').split(' | ')]
                print('guessed valid_metrics:', valid_metrics)
                return valid_metrics
    return []


def draw_simple(log_files: List[str], valid_metrics: List[str]):
    n = len(valid_metrics) + 1
    fig, axs = plt.subplots(1, n, figsize=(n * 5, 5))
    for log_file in log_files:
        train_records, valid_records = parse_file(log_file, valid_metrics)
        label = '.'.join(os.path.basename(log_file).split('.')[:-1])

        axs[0].plot(train_records.epoch, train_records.loss, label=label)
        axs[0].legend()
        axs[0].set_xlabel('epoch')
        axs[0].set_ylabel('train loss')

        for i, metric_name in enumerate(valid_metrics):
            best_score = float(getattr(valid_records, metric_name).max())
            axs[i + 1].plot(
                valid_records.num_updates, getattr(valid_records, metric_name), label=label + ': ' + str(best_score)
            )
            axs[i + 1].legend()
            axs[i + 1].set_xlabel('num_updates')
            axs[i + 1].set_ylabel('valid ' + metric_name)
    plt.show()


def draw_full(log_files: List[str], valid_metrics: List[str]):
    fig, (ax1, ax2) = plt.subplots(2, 3, figsize=(15, 8))
    for log_file in log_files:
        train_records, valid_records = parse_file(log_file, valid_metrics)
        label = '.'.join(os.path.basename(log_file).split('.')[:-1])

        ax1[0].plot(train_records.epoch, train_records.loss, label=label)
        ax1[0].legend()
        ax1[0].set_xlabel('epoch')
        ax1[0].set_ylabel('train loss')

        ax2[1].plot(train_records.epoch, train_records.lr, label=label)
        ax2[1].legend()
        ax2[1].set_xlabel('epoch')
        ax2[1].set_ylabel('learning rate')

        ax1[2].plot(train_records.epoch, train_records.ups * train_records.bsz, label=label)
        ax1[2].legend()
        ax1[2].set_xlabel('epoch')
        ax1[2].set_ylabel('nsamples / s')

        ax2[2].plot(train_records.num_updates, train_records.loss_scale_log2, label=label)
        ax2[2].legend()
        ax2[2].set_xlabel('num_updates')
        ax2[2].set_ylabel('loss scale (log2)')

        ax2[0].plot(valid_records.num_updates, valid_records.loss, label=label)
        ax2[0].legend()
        ax2[0].set_xlabel('num_updates')
        ax2[0].set_ylabel('valid loss')

        for i, metric_name in enumerate(valid_metrics[:1]):
            try:
                best_score = float(getattr(valid_records, metric_name).max())
            except ValueError:
                continue
            ax1[i + 1].plot(
                valid_records.num_updates, getattr(valid_records, metric_name), label=label + ': ' + str(best_score)
            )
            ax1[i + 1].legend()
            ax1[i + 1].set_xlabel('num_updates')
            ax1[i + 1].set_ylabel('valid ' + valid_metrics[0])
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'draw',
        description=(
            "draw ofasys logfiles by matplotlib, " "for example: python ofasys/draw.py x1.txt x2.txt --simple"
        ),
    )
    parser.add_argument('log_files', nargs='+')
    parser.add_argument('--simple', action='store_true')
    parser.add_argument('--metrics', nargs='+')
    args = parser.parse_args()

    if not args.metrics:
        args.metrics = guess_valid_metrics(args.log_files[0])
    if args.simple:
        draw_simple(args.log_files, args.metrics)
    else:
        draw_full(args.log_files, args.metrics)
