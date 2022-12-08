# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

from ofasys.configure import BaseDataclass
from ofasys.logging import metrics


@dataclass
class MetricConfig(BaseDataclass):
    target_field: Optional[str] = field(default=None, metadata={"help": "target field"})


class BaseMetric(ABC):
    def __init__(self, cfg: MetricConfig):
        self.cfg = cfg
        self.metrics = metrics

    @abstractmethod
    def compute(self, hyps, refs) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def report(self, logging_outputs: Dict) -> None:
        raise NotImplementedError

    def sum_logs(self, logging_outputs: Dict, key: str):
        result = sum(log.get(key, 0) for log in logging_outputs)
        if torch.is_tensor(result):
            result = result.cpu()
        return result
