# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.
#
import logging
from typing import Tuple

import diffusers
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from diffusers import DDIMScheduler, DDPMScheduler
from tqdm import tqdm
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def build_denoise_fn(net_input, model, reuse_encoder_out=False):
    slot = None
    for s in net_input["slots"]:
        if not s.is_src:
            assert slot is None, "The diffusion decoder does not support multiple target slots or plain text yet."
            slot = s

    ori_slot_value = slot.value
    assert (
        isinstance(ori_slot_value, dict) and ("value" in ori_slot_value) and ("noise_level" not in ori_slot_value)
    ), (
        "The diffusion decoder assumes the slot value to be a dict containing 'value', "
        "while not containing 'noise_level', "
        "so that it can corrupt 'value' and inject 'noise_level' into the dict when training."
    )

    if "value_0" not in ori_slot_value:
        ori_slot_value["value_0"] = ori_slot_value["value"]
    x_start = ori_slot_value["value_0"]

    encoder_out = None
    if reuse_encoder_out:
        assert (
            not model.training
        ), "No redundant encoding during training and thus need not set reuse_encoder_out=True."
        assert len(set(net_input.keys()) - {"slots"}) == 0
        encoder_out = model.encoder(
            slots=list(filter(lambda x: x.is_src, net_input["slots"])),
        )

    def denoise_fn(noised_images, noise_levels):
        slot.value = {}
        slot.value.update(ori_slot_value)
        slot.value["value"] = noised_images
        slot.value["noise_level"] = noise_levels
        if reuse_encoder_out:
            assert not model.training
            net_output = model.decoder(
                slots=list(filter(lambda x: not x.is_src, net_input["slots"])),
                encoder_out=encoder_out,
                full_context_alignment=True,
            )
        else:
            net_output = model(**net_input, full_context_alignment=True)
        return net_output[0]

    return denoise_fn, x_start, slot


# Please call this after (not before) calling scheduler.add_noise, because add_noise is
# responsible for converting stuff like alphas_cumprod to cuda and fp 16 when necessary.
def compute_snr(scheduler, timesteps):
    if isinstance(scheduler, DDIMScheduler) or isinstance(scheduler, DDPMScheduler):
        sqrt_alpha_prod = scheduler.alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - scheduler.alphas_cumprod[timesteps]) ** 0.5
        signal_to_noise_ratio = (sqrt_alpha_prod / sqrt_one_minus_alpha_prod).square()
    else:
        raise NotImplementedError
    return signal_to_noise_ratio


class DiffusionWrapper(object):
    def __init__(
        self,
        **kwargs,
    ):
        kwargs = kwargs.copy()
        default_kws = {
            'scheduler': 'DDIMScheduler',
            'beta_schedule': 'squaredcos_cap_v2',
            'clip_sample': False,
            'prediction_type': 'sample',
        }
        for k, v in default_kws.items():
            if k not in kwargs:
                kwargs[k] = v
        logger.info('Diffusion Config: %s' % str(kwargs))
        self.prediction_type = kwargs['prediction_type']
        self.num_inference_steps = kwargs.pop('num_inference_steps', None)
        self.scheduler = getattr(diffusers, kwargs.pop('scheduler'))(**kwargs)
        if self.num_inference_steps is None:
            self.num_inference_steps = self.scheduler.num_train_timesteps

    @torch.no_grad()
    def sample(
        self,
        denoise_fn,
        batch_size: int,
        image_shape: Tuple[int],
        device,
        float_dtype=torch.float32,
        postproc_fn=None,
        guidance_weight: float = 0.0,
    ):
        if guidance_weight != 0.0:
            logger.warning(
                "This is a temporary API for negative prompting and classifier-free guidance. "
                "Please construct your batch such that sample[0::2] contains the actual conditions "
                "while sample[1::2] contains the negative/NULL conditions to accompany sample[0::2]. "
                "P.S.: NULL text is an empty string. "
                "Guidance weight: %f" % guidance_weight
            )
            assert batch_size % 2 == 0

        self.scheduler.set_timesteps(self.num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        x_t = torch.randn(size=(batch_size, *image_shape), device=device, dtype=float_dtype)
        x_t = x_t * self.scheduler.init_noise_sigma
        img = x_t

        for i, t in enumerate(tqdm(timesteps, desc='diffusion sampling step')):
            x_t = self.scheduler.scale_model_input(x_t, t)
            noise = torch.randn(size=(batch_size, *image_shape), device=device, dtype=float_dtype)
            if guidance_weight != 0.0:
                x_t[1::2] = x_t[0::2]
                noise[1::2] = noise[0::2]
            model_output = denoise_fn(x_t, t.unsqueeze(0).expand(batch_size))
            if guidance_weight != 0.0:
                model_output[0::2] = (1 + guidance_weight) * model_output[0::2] - guidance_weight * model_output[1::2]
            if postproc_fn is not None:
                assert self.prediction_type == 'sample', (
                    'It is currently inconvenient to do custom postprocessing beyond clamp(-1,+1) if '
                    'prediction_type != sample, unless we modify the source of the diffusers library.'
                )
                model_output = postproc_fn(model_output)
            step_output = self.scheduler.step(
                model_output=model_output,
                timestep=t,
                sample=x_t,
                variance_noise=noise,
                use_clipped_model_output=True,  # https://github.com/huggingface/diffusers/issues/1486
            )
            x_t, img = step_output.prev_sample, step_output.pred_original_sample

        return img

    def p_losses(self, denoise_fn, x_start, loss_fn=F.l1_loss):
        b, device = x_start.shape[0], x_start.device
        noise = torch.randn_like(x_start)
        t = torch.randint(0, self.scheduler.num_train_timesteps, (b,), device=device, dtype=torch.long)
        x_t = self.scheduler.add_noise(x_start, noise, t)
        model_output = denoise_fn(x_t, t)
        if self.prediction_type == 'epsilon':
            x_predict = None
            sample_weights = None
            losses = loss_fn(model_output, noise, reduction="none")
        elif self.prediction_type == 'sample':
            x_predict = model_output
            sample_weights = (compute_snr(self.scheduler, t) + 1.0).sqrt()
            w = sample_weights.view(*([b] + [1] * (len(x_start.shape) - 1)))
            losses = loss_fn(w * x_predict, w * x_start, reduction="none")
        else:
            raise NotImplementedError
        return losses, x_predict, sample_weights
