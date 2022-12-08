# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import inspect
import string
from dataclasses import dataclass, field
from itertools import zip_longest
from typing import Dict, List, Optional

import numpy as np
import sacrebleu
from sacrebleu.metrics import BLEU

from ofasys.configure import register_config

from .base import BaseMetric, MetricConfig

_tok_dict = {
    "(": "-lrb-",
    ")": "-rrb-",
    "[": "-lsb-",
    "]": "-rsb-",
    "{": "-lcb-",
    "}": "-rcb-",
    "[UNK]": "UNK",
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
}


def _is_digit(w):
    for ch in w:
        if not (ch.isdigit() or ch == ','):
            return False
    return True


def fix_tokenization(text):
    input_tokens = text.split()
    output_tokens = []
    has_left_quote = False
    has_left_single_quote = False

    i = 0
    prev_dash = False
    while i < len(input_tokens):
        tok = input_tokens[i]
        flag_prev_dash = False
        if tok in _tok_dict.keys():
            output_tokens.append(_tok_dict[tok])
            i += 1
        elif tok == "\"":
            if has_left_quote:
                output_tokens.append("''")
            else:
                output_tokens.append("``")
            has_left_quote = not has_left_quote
            i += 1
        elif (
            tok == "'"
            and len(output_tokens) > 0
            and output_tokens[-1].endswith("n")
            and i < len(input_tokens) - 1
            and input_tokens[i + 1] == "t"
        ):
            output_tokens[-1] = output_tokens[-1][:-1]
            output_tokens.append("n't")
            i += 2
        elif tok == "'" and i < len(input_tokens) - 1 and input_tokens[i + 1] in ("s", "d", "ll"):
            output_tokens.append("'" + input_tokens[i + 1])
            i += 2
        elif tok == "'":
            if has_left_single_quote:
                output_tokens.append("'")
            else:
                output_tokens.append("`")
            has_left_single_quote = not has_left_single_quote
            i += 1
        elif tok == "." and i < len(input_tokens) - 2 and input_tokens[i + 1] == "." and input_tokens[i + 2] == ".":
            output_tokens.append("...")
            i += 3
        elif (
            tok == ","
            and len(output_tokens) > 0
            and _is_digit(output_tokens[-1])
            and i < len(input_tokens) - 1
            and _is_digit(input_tokens[i + 1])
        ):
            # $ 3 , 000 -> $ 3,000
            output_tokens[-1] += ',' + input_tokens[i + 1]
            i += 2
        elif (
            tok == "."
            and len(output_tokens) > 0
            and output_tokens[-1].isdigit()
            and i < len(input_tokens) - 1
            and input_tokens[i + 1].isdigit()
        ):
            # 3 . 03 -> $ 3.03
            output_tokens[-1] += '.' + input_tokens[i + 1]
            i += 2
        elif (
            tok == "."
            and len(output_tokens) > 0
            and len(output_tokens[-1]) == 1
            and output_tokens[-1].isupper()
            and i < len(input_tokens) - 2
            and len(input_tokens[i + 1]) == 1
            and input_tokens[i + 1].isupper()
            and input_tokens[i + 2] == '.'
        ):
            # U . N . -> U.N.
            k = i + 3
            while k + 2 < len(input_tokens):
                if len(input_tokens[k + 1]) == 1 and input_tokens[k + 1].isupper() and input_tokens[k + 2] == '.':
                    k += 2
                else:
                    break
            output_tokens[-1] += ''.join(input_tokens[i:k])
            i += 2
        elif tok == "-":
            if i < len(input_tokens) - 1 and input_tokens[i + 1] == "-":
                output_tokens.append("--")
                i += 2
            elif i == len(input_tokens) - 1 or i == 0:
                output_tokens.append("-")
                i += 1
            elif output_tokens[-1] not in string.punctuation and input_tokens[i + 1][0] not in string.punctuation:
                output_tokens[-1] += "-"
                i += 1
                flag_prev_dash = True
            else:
                output_tokens.append("-")
                i += 1
        elif prev_dash and len(output_tokens) > 0 and tok[0] not in string.punctuation:
            output_tokens[-1] += tok
            i += 1
        else:
            output_tokens.append(tok)
            i += 1
        prev_dash = flag_prev_dash
    return " ".join(output_tokens)


@dataclass
class BleuConfig(MetricConfig):
    tokenized_bleu: Optional[bool] = field(
        default=False, metadata={"help": "calculate bleu after tokenized sentences"}
    )


@register_config("ofasys.metric", "bleu", BleuConfig)
class Bleu(BaseMetric):
    EVAL_BLEU_ORDER = 4

    def __init__(self, cfg: BleuConfig):
        super().__init__(cfg)
        self.tokenized_bleu = cfg.tokenized_bleu

    def compute(self, hyps, refs) -> Dict:
        logging_output = {}
        if self.tokenized_bleu:
            # TODO: tokenized hyps and refs has not given yet
            bleu = sacrebleu.corpus_bleu(hyps, list(zip_longest(*refs)), tokenize="none")
        else:
            hyps = [fix_tokenization(_hyp) for _hyp in hyps]
            refs = [[fix_tokenization(_ref) for _ref in _ref_list] for _ref_list in refs]
            bleu = sacrebleu.corpus_bleu(hyps, list(zip_longest(*refs)))
        logging_output["_bleu_sys_len"] = bleu.sys_len
        logging_output["_bleu_ref_len"] = bleu.ref_len

        # we split counts into separate entries so that they can be
        # summed efficiently across workers using fast-stat-sync
        assert len(bleu.counts) == Bleu.EVAL_BLEU_ORDER
        for i in range(Bleu.EVAL_BLEU_ORDER):
            logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
            logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
        return logging_output

    def report(self, logging_outputs: Dict) -> None:
        counts, totals = [], []
        for i in range(Bleu.EVAL_BLEU_ORDER):
            counts.append(self.sum_logs(logging_outputs, "_bleu_counts_" + str(i)))
            totals.append(self.sum_logs(logging_outputs, "_bleu_totals_" + str(i)))

        if max(totals) > 0:
            # log counts as numpy arrays -- log_scalar will sum them correctly
            self.metrics.log_scalar("_bleu_counts", np.array(counts))
            self.metrics.log_scalar("_bleu_totals", np.array(totals))
            self.metrics.log_scalar("_bleu_sys_len", self.sum_logs(logging_outputs, "_bleu_sys_len"))
            self.metrics.log_scalar("_bleu_ref_len", self.sum_logs(logging_outputs, "_bleu_ref_len"))

            def compute_bleu(meters):
                fn_sig = inspect.getfullargspec(BLEU.compute_bleu)[0]
                if "smooth_method" in fn_sig:
                    smooth = {"smooth_method": "exp"}
                else:
                    smooth = {"smooth": "exp"}
                bleu = BLEU.compute_bleu(
                    correct=meters["_bleu_counts"].sum,
                    total=meters["_bleu_totals"].sum,
                    sys_len=int(meters["_bleu_sys_len"].sum),
                    ref_len=int(meters["_bleu_ref_len"].sum),
                    **smooth
                )
                return round(bleu.score, 2)

            self.metrics.log_derived("bleu", compute_bleu)
