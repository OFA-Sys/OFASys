"""Spider exact match metric."""
"""Borrow from the PICARD code"""
"""Spider Test Suite Execution Accuracy metric."""

# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import logging
from typing import Any, Dict, Optional

from ofasys.configure import register_config

from .base import BaseMetric, MetricConfig

logger = logging.getLogger(__name__)


@register_config("ofasys.metric", "exact_match", MetricConfig)
class Exact_match(BaseMetric):
    def __init__(self, cfg: MetricConfig):
        super().__init__(cfg)
        # self.tokenized_eval = tokenized_eval

    def compute(self, hyps, refs) -> Dict:
        logging_output = {}
        db_struct = refs
        exact_match = compute_exact_match_metric(hyps, db_struct)
        logging_output["_em_counts"] = exact_match["exact_match"]
        logging_output["_em_totals"] = 1.0

        return logging_output

    def report(self, logging_outputs: Dict) -> None:
        # counts, totals = [], []
        counts = self.sum_logs(logging_outputs, "_em_counts")
        totals = self.sum_logs(logging_outputs, "_em_totals")
        if totals > 0:
            # log counts as numpy arrays -- log_scalar will sum them correctly
            def compute_exact_match(meters):
                _tmp = meters["_em_counts"].sum / meters["_em_totals"].sum
                _tmp = _tmp if isinstance(_tmp, float) else _tmp.item()
                return round(_tmp, 3)

            self.metrics.log_scalar("_em_counts", counts)
            self.metrics.log_scalar("_em_totals", totals)
            self.metrics.log_derived("exact_match", compute_exact_match)


def compute_exact_match_metric(predictions, references) -> Dict[str, Any]:
    from ofasys.utils.spider import evaluation as spider_evaluation

    foreign_key_maps = dict()
    for reference in references:
        if reference["db_id"] not in foreign_key_maps:
            foreign_key_maps[reference["db_id"]] = spider_evaluation.build_foreign_key_map(
                {
                    "table_names_original": reference["db_table_names"],
                    "column_names_original": list(
                        zip(
                            reference["db_column_names"]["table_id"],
                            reference["db_column_names"]["column_name"],
                        )
                    ),
                    "foreign_keys": list(
                        zip(
                            reference["db_foreign_keys"]["column_id"],
                            reference["db_foreign_keys"]["other_column_id"],
                        )
                    ),
                }
            )
    evaluator = spider_evaluation.Evaluator(references[0]["db_path"], foreign_key_maps, "match")
    for prediction, reference in zip(predictions, references):
        turn_idx = reference.get("turn_idx", 0)
        # skip final utterance-query pairs
        if turn_idx < 0:
            continue
        _ = evaluator.evaluate_one(reference["db_id"], reference["query"], prediction)
    evaluator.finalize()
    return {
        "exact_match": evaluator.scores["all"]["exact"],
    }


def compute_test_suite_metric(predictions, references, db_dir: Optional[str] = None) -> Dict[str, Any]:
    from ofasys.utils.test_suite import evaluation as test_suite_evaluation

    if db_dir is None:
        db_dir = references[0]["db_path"]

    foreign_key_maps = dict()
    for reference in references:
        if reference["db_id"] not in foreign_key_maps:
            foreign_key_maps[reference["db_id"]] = test_suite_evaluation.build_foreign_key_map(
                {
                    "table_names_original": reference["db_table_names"],
                    "column_names_original": list(
                        zip(
                            reference["db_column_names"]["table_id"],
                            reference["db_column_names"]["column_name"],
                        )
                    ),
                    "foreign_keys": list(
                        zip(
                            reference["db_foreign_keys"]["column_id"],
                            reference["db_foreign_keys"]["other_column_id"],
                        )
                    ),
                }
            )

    evaluator = test_suite_evaluation.Evaluator(
        db_dir=db_dir,
        kmaps=foreign_key_maps,
        etype="exec",
        plug_value=False,
        keep_distinct=False,
        progress_bar_for_each_datapoint=False,
    )
    # Only used for Sparc/CoSQL
    turn_scores = {"exec": [], "exact": []}
    for prediction, reference in zip(predictions, references):
        turn_idx = reference.get("turn_idx", 0)
        # skip final utterance-query pairs
        if turn_idx < 0:
            continue
        try:
            _ = evaluator.evaluate_one(
                reference["db_id"],
                reference["query"],
                prediction,
                turn_scores,
                idx=turn_idx,
            )
        except AssertionError as e:
            logger.warning(f"unexpected evaluation error: {e.args[0]}")
    evaluator.finalize()
    return {
        "exec": evaluator.scores["all"]["exec"],
    }
