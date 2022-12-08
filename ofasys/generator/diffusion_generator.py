# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch

from ofasys import ModalityType
from ofasys.module.diffusion import ElucidatedDiffusion, GaussianDiffusion
from ofasys.module.motion_6d import BvhObject
from ofasys.preprocessor import Slot

from .base import BatchGeneratorOutput, Generator, GeneratorOutput, to_numpy


@dataclass
class MotionOutput(GeneratorOutput):
    """
    Output of DiffusionGenerator.
    Output with origin data format (e.g. bvh, gif) are available.
    Original output in tensor format and extra information are also provided.
    """

    feature: Union[torch.FloatTensor, np.ndarray]
    bvh: Optional[BvhObject] = None
    target_feature: Optional[Union[torch.FloatTensor, np.ndarray]] = None
    prompt: Optional[str] = None

    def save_as_gif(self, gif_name: str):
        """
        save output as a gif file.

        Args:
            gif_name: save file path.

        """
        assert self.bvh is not None
        if not gif_name.endswith(".gif"):
            gif_name = gif_name + ".gif"
        self.bvh.save_as_gif(gif_name)

    def save_as_bvh(self, bvh_name: str):
        """
        save output as a bvh file.

        Args:
            bvh_name: save file path.

        """
        assert self.bvh is not None
        if not bvh_name.endswith(".bvh"):
            bvh_name = bvh_name + ".bvh"
        self.bvh.save_as_bvh(bvh_name)

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
        if not feature_name.endwith(".npz"):
            feature_name = feature_name + ".npz"
        np.savez(file=feature_name, **data)


class DiffusionGenerator(Generator):
    def __init__(
        self, preprocessor, diffusion=None, diffuser_type="ddpm", stepwise_clamp=True, output_shape=None, **kwargs
    ):
        """Diffusion generator for motion modality.

        Args:
            preprocessor: object of preprocessor.
            diffusion: diffuser object, if None will init according to diffuser_type.
            diffuser_type: diffuser_type. ddpm, ddim, elucidated are available.
            stepwise_clamp: whether use step wise clamp.
            output_shape: output shape.
        """
        super().__init__()
        self.preprocessor = preprocessor
        self.stepwise_clamp = stepwise_clamp

        self.dtype = kwargs.pop("dtype", torch.float32)
        self.device = kwargs.pop("device", None)

        if diffusion is not None:
            self.diffusion = diffusion
        elif diffuser_type == "ddpm":
            num_sample_steps = kwargs.pop("num_sample_steps", 1000)
            beta_schedule = kwargs.pop("beta_schedule", "cosine")
            self.diffusion = GaussianDiffusion(
                num_sample_steps=num_sample_steps, beta_schedule=beta_schedule, device=self.device
            )
        elif diffuser_type == "ddim":
            num_sample_steps = kwargs.pop("num_sample_steps", 1000)
            beta_schedule = kwargs.pop("beta_schedule", "cosine")
            ddim_steps = kwargs.pop("ddim_steps", 200)
            ddim_eta = kwargs.pop("ddim_eta", 1.0)
            self.diffusion = GaussianDiffusion(
                num_sample_steps=num_sample_steps,
                beta_schedule=beta_schedule,
                use_ddim=True,
                ddim_steps=ddim_steps,
                ddim_eta=ddim_eta,
                device=self.device,
            )
        elif diffuser_type == "elucidated":
            num_sample_steps = kwargs.pop("num_sample_steps", 32)
            sigma_min = kwargs.pop("sigma_min", 0.002)
            sigma_max = kwargs.pop("sigma_max", 80)
            sigma_data = kwargs.pop("sigma_data", 0.5)
            rho = kwargs.pop("rho", 7)
            p_mean = kwargs.pop("p_mean", -1.2)
            p_std = kwargs.pop("p_std", 1.2)
            s_churn = kwargs.pop("s_churn", 80)
            s_tmin = kwargs.pop("s_tmin", 0.05)
            s_tmax = kwargs.pop("s_tmax", 50)
            s_noise = kwargs.pop("s_noise", 1.003)
            self.diffusion = ElucidatedDiffusion(
                num_sample_steps=num_sample_steps,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                sigma_data=sigma_data,
                rho=rho,
                p_mean=p_mean,
                p_std=p_std,
                s_churn=s_churn,
                s_tmin=s_tmin,
                s_tmax=s_tmax,
                s_noise=s_noise,
            )
        else:
            raise ValueError
        self.output_shape = output_shape

    @torch.no_grad()
    def generate(self, model, sample, **kwargs):
        """
        Generate function. Should be overridden by all subclasses.
        """
        model.eval()

        net_input = sample["net_input"]
        source_slots = list(filter(lambda x: x.is_src, net_input['slots']))
        target_slot = Slot.get_target_slot_from_slots(net_input["slots"])
        origin_target_slot_value = target_slot.value
        assert target_slot.modality == ModalityType.MOTION, (
            f"the target slot does not match the generator,"
            f" target_slot: {target_slot.modality}, generator: DiffusionGenerator"
        )

        output_shape = self.output_shape
        if output_shape is None:
            assert target_slot.value is not None
            output_shape = target_slot.value["value"].shape[1:]

        if source_slots[0].modality == ModalityType.AUDIO:
            src_tokens = source_slots[0].value["fbank"]
        else:
            src_tokens = source_slots[0].value
        bsz = src_tokens.shape[0]

        encoder_out = model.encoder.forward(slots=source_slots)

        if self.stepwise_clamp:
            postproc_fn = getattr(self.preprocessor, 'custom_clamp', None)
        else:
            postproc_fn = None

        def denoise_fn(noised_inputs, noise_levels):
            target_slot.value = {}
            target_slot.value.update(origin_target_slot_value)
            target_slot.value["value"] = noised_inputs
            target_slot.value["noise_level"] = noise_levels

            net_output = model.decoder.forward(
                slots=[target_slot],
                encoder_out=encoder_out,
                full_context_alignment=True,
            )
            return net_output[0]

        outputs = self.diffusion.sample(
            denoise_fn, bsz, output_shape, device=self.device, float_dtype=self.dtype, postproc_fn=postproc_fn
        )

        finalized: BatchGeneratorOutput = [MotionOutput(feature=outputs[i]) for i in range(bsz)]
        return finalized
