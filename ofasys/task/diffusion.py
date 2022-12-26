# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
from torch.distributed import get_rank

from ofasys.configure import register_config
from ofasys.generator import MotionOutput
from ofasys.task.base import OFATask, TaskConfig

# TODO: To be removed. It might be better to just use OFATask.


@dataclass
class DiffusionTaskConfig(TaskConfig):
    prompt_slot: str = field(default="", metadata={"help": ""})
    drop_prompt: float = field(default=0.1, metadata={"help": ""})


@register_config("ofasys.task", "diffusion", dataclass=DiffusionTaskConfig)
class DiffusionTask(OFATask):
    def __init__(self, cfg: DiffusionTaskConfig, **kwargs):
        super().__init__(cfg, **kwargs)
        self.prompt_slot = cfg.prompt_slot
        self.drop_prompt = cfg.drop_prompt

    def preprocess(self, data: Dict[str, Any], split: str) -> Dict[str, Any]:
        if split == "train":
            if (self.prompt_slot != "") and (np.random.rand() < self.drop_prompt):
                if isinstance(data[self.prompt_slot], str):
                    data[self.prompt_slot] = ""
                else:
                    raise NotImplementedError("This implementation only supports a text prompt as the condition.")
        return data

    def inference(self, model, sample, **kwargs):
        try:
            worker_id = get_rank()
        except RuntimeError:
            worker_id = 0

        outputs = super().inference(model, sample, **kwargs)

        if self.cfg.evaluation.output_dir:
            os.makedirs(self.cfg.evaluation.output_dir, exist_ok=True)

            single_output: MotionOutput
            for idx, single_output in enumerate(outputs):
                single_output.prompt = sample[self.prompt_slot][idx]
                save_path = os.path.join(
                    self.cfg.evaluation.output_dir, '%d_%d.npz' % (int(time.time() * 1000), worker_id)
                )
                # note: The feature here have not been decoded by the preprocessor
                single_output.save_features(save_path)

        return outputs
