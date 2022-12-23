# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch

from ofasys import ModalityType
from ofasys.generator.base import (
    BatchGeneratorOutput,
    Generator,
    GeneratorOutput,
    to_numpy,
)
from ofasys.module.diffusion import DiffusionWrapper, build_denoise_fn
from ofasys.module.motion_6d import BvhHeader


@dataclass
class MotionOutput(GeneratorOutput):
    """
    Output of DiffusionGenerator.
    Output with origin data format (e.g. bvh, gif) are available.
    Original output in tensor format and extra information are also provided.
    """

    feature: Union[torch.FloatTensor, np.ndarray]
    target_feature: Optional[Union[torch.FloatTensor, np.ndarray]] = None
    prompt: Optional[str] = None
    bvh_header: Optional[BvhHeader] = None
    bvh_motion: Optional[np.ndarray] = None

    def save_as_gif(self, gif_name: str):
        """
        save output as a gif file.

        Args:
            gif_name: save file path.

        """
        if not gif_name.endswith(".gif"):
            gif_name = gif_name + ".gif"
        self.bvh_header.save_as_gif(self.bvh_motion, gif_name)

    def save_as_bvh(self, bvh_name: str):
        """
        save output as a bvh file.

        Args:
            bvh_name: save file path.

        """
        if not bvh_name.endswith(".bvh"):
            bvh_name = bvh_name + ".bvh"
        self.bvh_header.save_as_bvh(self.bvh_motion, bvh_name)

    def save_features(self, feature_name: str):
        """
        save output feature as a npz file.

        Args:
            feature_name: save file path.

        """
        data = {
            "feature": to_numpy(self.feature),
            "target_feature": to_numpy(self.target_feature),
            "prompt": self.prompt,
        }
        if not feature_name.endswith(".npz"):
            feature_name = feature_name + ".npz"
        np.savez(file=feature_name, **data)


class DiffusionGenerator(Generator):
    def __init__(self, general_preprocess, diffuser_args, **kwargs):
        """Diffusion generator.

        Args:
            general_preprocess: object of general preprocessor.
            diffuser_args: arguments passed to the __init__ of a Diffusion implementation such as GaussianDiffusion
        """
        super().__init__()
        self.general_preprocess = general_preprocess
        self.dtype = kwargs.get("dtype", torch.float32)
        self.device = kwargs.get("device", None)
        self.diffusion = DiffusionWrapper(**diffuser_args)
        self.guidance_weight = kwargs.get("guidance_weight", 0.0)

    @torch.no_grad()
    def generate(self, model, sample, **kwargs):
        """
        Generate function. Should be overridden by all subclasses.
        """
        model.eval()

        denoise_fn, x_dummy, target_slot = build_denoise_fn(sample["net_input"], model, reuse_encoder_out=True)
        preprocessor = self.general_preprocess.get_preprocess(target_slot)

        assert target_slot.modality == ModalityType.MOTION, "Modality other than MOTION not tested yet."
        bsz, output_shape = x_dummy.shape[0], x_dummy.shape[1:]

        postproc_fn = preprocessor.build_clamp_fn(slot=target_slot)
        outputs = self.diffusion.sample(
            denoise_fn,
            bsz,
            output_shape,
            device=self.device,
            float_dtype=self.dtype,
            postproc_fn=postproc_fn,
            guidance_weight=self.guidance_weight,
        )

        finalized: BatchGeneratorOutput = [MotionOutput(feature=outputs[i]) for i in range(bsz)]
        return finalized
