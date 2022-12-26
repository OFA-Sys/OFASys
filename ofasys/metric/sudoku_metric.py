"""Sudoku metric."""


# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import logging
from dataclasses import dataclass, field
from itertools import zip_longest
from typing import Any, Dict, Optional

import numpy as np

from ofasys.configure import register_config

from .base import BaseMetric, MetricConfig

logger = logging.getLogger(__name__)


@dataclass
class SudokuConfig(MetricConfig):
    eval_by_mask_ratio: Optional[bool] = field(default=False, metadata={"help": "see the acc in different maskratio"})


@register_config("ofasys.metric", "solved_acc", SudokuConfig)
class Solved_acc(BaseMetric):
    def __init__(self, cfg: SudokuConfig):
        super().__init__(cfg)
        self.eval_by_mask_ratio = cfg.eval_by_mask_ratio

    def compute(self, hyps, refs) -> Dict:
        logging_output = {}
        if self.eval_by_mask_ratio:
            return self.mask_ratio_compute(hyps, refs)
        else:
            mask_ratios, questions, answers = list(zip_longest(*refs))
            align, right, solved = sudoku_evaluate(hyps, questions, answers)
            logging_output["_align_counts"] = align
            logging_output["_right_counts"] = right
            logging_output["_solved_counts"] = solved
            logging_output["_solved_totals"] = 1.0
            return logging_output

    def compute_acc_metrics(self, metric_name, mask_prefix):
        def _compute_acc(meters):
            _tmp = meters[metric_name].sum / meters[mask_prefix + "solved_totals"].sum
            _tmp = _tmp if isinstance(_tmp, float) else _tmp.item()
            return round(_tmp, 4)

        return _compute_acc

    def report(self, logging_outputs: Dict) -> None:
        # counts, totals = [], []
        if self.eval_by_mask_ratio:
            self.mask_ratio_report(logging_outputs)
        else:
            align = self.sum_logs(logging_outputs, "_align_counts")
            right = self.sum_logs(logging_outputs, "_right_counts")
            solved = self.sum_logs(logging_outputs, "_solved_counts")
            total = self.sum_logs(logging_outputs, "_solved_totals")

            if total > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                self.metrics.log_scalar("_align_counts", align)
                self.metrics.log_scalar("_right_counts", right)
                self.metrics.log_scalar("_solved_counts", solved)
                self.metrics.log_scalar("_solved_totals", total)
                self.metrics.log_derived("align_acc", self.compute_acc_metrics("_align_counts", "_"))
                self.metrics.log_derived("right_acc", self.compute_acc_metrics("_right_counts", "_"))
                self.metrics.log_derived("solved_acc", self.compute_acc_metrics("_solved_counts", "_"))

    def mask_ratio_compute(self, hyps, refs) -> Dict:
        logging_output = {}
        mask_ratios, questions, answers = list(zip_longest(*refs))
        maskratio_metric = {"_": {"align_counts": 0, "right_counts": 0, "solved_counts": 0, "solved_totals": 0}}
        for idx in range(len(mask_ratios)):
            mask_ratio = int(mask_ratios[idx] * 10) / 10.0
            mask_prefix = "_mr_" + str(mask_ratio) + "_"
            align, right, solved = sudoku_evaluate([hyps[idx]], [questions[idx]], [answers[idx]])
            maskratio_metric["_"]["align_counts"] += align
            maskratio_metric["_"]["right_counts"] += right
            maskratio_metric["_"]["solved_counts"] += solved
            maskratio_metric["_"]["solved_totals"] += 1.0

            if mask_prefix not in maskratio_metric:
                maskratio_metric[mask_prefix] = {
                    "align_counts": align,
                    "right_counts": right,
                    "solved_counts": solved,
                    "solved_totals": 1.0,
                }
            else:
                maskratio_metric[mask_prefix]["align_counts"] += align
                maskratio_metric[mask_prefix]["right_counts"] += right
                maskratio_metric[mask_prefix]["solved_counts"] += solved
                maskratio_metric[mask_prefix]["solved_totals"] += 1.0
        for mask_prefix in maskratio_metric.keys():
            for metric_name in maskratio_metric[mask_prefix]:
                logging_output[mask_prefix + metric_name] = maskratio_metric[mask_prefix][metric_name]
        return logging_output

    def mask_ratio_report(self, logging_outputs: Dict) -> None:
        # counts, totals = [], []
        mask_ratio_list = ['', 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        for mask_ratio in mask_ratio_list:
            if mask_ratio != '':
                mask_prefix = "_mr_" + str(mask_ratio) + "_"
            else:
                mask_prefix = "_"
            align = self.sum_logs(logging_outputs, mask_prefix + "align_counts")
            right = self.sum_logs(logging_outputs, mask_prefix + "right_counts")
            solved = self.sum_logs(logging_outputs, mask_prefix + "solved_counts")
            total = self.sum_logs(logging_outputs, mask_prefix + "solved_totals")

            if total > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                self.metrics.log_scalar(mask_prefix + "align_counts", align)
                self.metrics.log_scalar(mask_prefix + "right_counts", right)
                self.metrics.log_scalar(mask_prefix + "solved_counts", solved)
                self.metrics.log_scalar(mask_prefix + "solved_totals", total)
                self.metrics.log_derived(
                    mask_prefix[1:] + "align_acc", self.compute_acc_metrics(mask_prefix + "align_counts", mask_prefix)
                )
                self.metrics.log_derived(
                    mask_prefix[1:] + "right_acc", self.compute_acc_metrics(mask_prefix + "right_counts", mask_prefix)
                )
                self.metrics.log_derived(
                    mask_prefix[1:] + "solved_acc",
                    self.compute_acc_metrics(mask_prefix + "solved_counts", mask_prefix),
                )


def transfertxt2sudoku(txt):
    if isinstance(txt, list) and len(txt) == 9:
        # it has already been a sudoku list
        sudokus = [list(map(int, t_)) for t_ in txt]
        return sudokus
    sudokus = txt.split(" | ")
    sudokus = [list(map(int, su_.split(" : "))) for su_ in sudokus]
    return sudokus


def sudoku_evaluate(outputs, inputs, labels):
    align, right, solved = 0.0, 0.0, 0.0
    length = len(inputs)
    for input, output, label in zip(*(inputs, outputs, labels)):
        tmp_align, tmp_right, tmp_solved = each_sudoku_evaluate(input, output, label)
        align += tmp_align
        right += tmp_right
        solved += tmp_solved
    return align / length, right / length, solved / length


def each_sudoku_evaluate(input, output, label):
    if isinstance(output, list) and len(output) == 1:
        output = output[0]
    if isinstance(input, list) and len(input) == 1:
        input = input[0]
    if isinstance(label, list) and len(label) == 1:
        label = label[0]
    if not is_generate_form_right(output):
        print("False output", output)
        return 0.0, 0.0, 0.0
    puzzle = transfertxt2sudoku(input)
    solved = transfertxt2sudoku(output)
    is_right = check_solution(solved)
    is_align = check_align(puzzle, solved)
    total_right = is_align & is_right
    return int(is_align), int(is_right), int(total_right)


def is_generate_form_right(output):
    out_line = output.split("|")
    if len(out_line) != 9:
        return False
    for each_line in out_line:
        out_col = each_line.strip().split(":")
        if len(out_col) != 9:
            return False
        for each_num in out_col:
            numbs = each_num.strip()
            if len(numbs) != 1:
                return False
            if not numbs.isdigit():
                return False
    return True


def check_solution(m):
    if isinstance(m, list):
        m = np.array(m)
    elif isinstance(m, str):
        m = np.loadtxt(m, dtype=np.int, delimiter=",")
    set_rg = set(np.arange(1, m.shape[0] + 1))
    success = True
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            r1 = set(m[3 * (i // 3) : 3 * (i // 3 + 1), 3 * (j // 3) : 3 * (j // 3 + 1)].ravel()) == set_rg
            r2 = set(m[i, :]) == set_rg
            r3 = set(m[:, j]) == set_rg
            if not (r1 and r2 and r3):
                success = False
                break
        if not success:
            break
    return success


def check_align(tmp_sudo, solved):
    for idx in range(9):
        for jdx in range(9):
            if tmp_sudo[idx][jdx] != 0:
                if tmp_sudo[idx][jdx] != solved[idx][jdx]:
                    return False
    return True
