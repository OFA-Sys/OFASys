# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
from torch.distributed import get_rank

from ofasys.configure import register_config
from ofasys.generator import DiffusionGenerator, MotionOutput
from ofasys.module.diffusion import ElucidatedDiffusion, GaussianDiffusion
from ofasys.task.base import OFATask, TaskConfig


@dataclass
class DiffusionTaskConfig(TaskConfig):
    diffuser: str = field(
        default="ddpm",
        metadata={"help": "choose from one of the diffusion implementations: ddpm, ddim, elucidated"},
    )
    diffuser_args: str = field(
        default='{"num_sample_steps": 1000}', metadata={"help": "args for the diffuser, as JSON string"}
    )
    prompt_slot: str = field(default="", metadata={"help": ""})
    drop_prompt: float = field(default=0.5, metadata={"help": ""})
    stepwise_clamp: bool = field(default=True, metadata={"help": ""})


@register_config("ofasys.task", "diffusion", dataclass=DiffusionTaskConfig)
class DiffusionTask(OFATask):
    def __init__(self, cfg: DiffusionTaskConfig, **kwargs):
        diffuser_args = json.loads(cfg.diffuser_args)

        if cfg.diffuser == 'ddpm':
            self.diffusion = GaussianDiffusion(**diffuser_args)
        elif cfg.diffuser == 'ddim':
            self.diffusion = GaussianDiffusion(use_ddim=True, **diffuser_args)
        elif cfg.diffuser == 'elucidated':
            self.diffusion = ElucidatedDiffusion(**diffuser_args)
        else:
            raise NotImplementedError
        self.prompt_slot = cfg.prompt_slot
        self.drop_prompt = cfg.drop_prompt
        self.stepwise_clamp = cfg.stepwise_clamp

        # Put super().__init__ after initializing the diffuser, because super().__init__
        # will build the diffusion criterion and the criterion will access task.diffusion.
        super().__init__(cfg, **kwargs)

    def preprocess(self, data: Dict[str, Any], split: str) -> Dict[str, Any]:
        if split == "train":
            if (self.prompt_slot != "") and (np.random.rand() < self.drop_prompt):
                if isinstance(data[self.prompt_slot], str):
                    data[self.prompt_slot] = ""
                else:
                    raise NotImplementedError("This implementation only supports a text prompt as the condition.")
        return data

    def build_motion_diffusion_generator(self, **gen_kwargs):
        return DiffusionGenerator(
            self.general_preprocess.name2pre["motion"],
            diffusion=self.diffusion,
            stepwise_clamp=self.stepwise_clamp,
        )

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
