# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

#
# based on https://github.com/lucidrains/denoising-diffusion-pytorch
#
import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm


def build_denoise_fn(net_input, model, reuse_encoder_out=False):
    slot = None
    for s in net_input["slots"]:
        if not s.is_src:
            assert slot is None, "The diffusion decoder does not support multiple target slots or plain text yet."
            slot = s

    ori_slot_value = slot.value
    assert isinstance(ori_slot_value, dict) and "value" in ori_slot_value, (
        "The diffusion decoder assumes the slot value to be a dict containing 'value', "
        "so that it can corrupt 'value' and inject 'noise_level' into the dict when training."
    )

    x_start = ori_slot_value["value"]

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


class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super(LearnedSinusoidalPosEmb, self).__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = x.unsqueeze(-1)  # [B]->[B,1]
        freqs = x * self.weights.unsqueeze(0) * 2 * math.pi  # [B,1]*[1,D]
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


class ContinuousTimeMLP(nn.Module):
    def __init__(self, time_dim, learned_sinusoidal_dim=16):
        super(ContinuousTimeMLP, self).__init__()
        if learned_sinusoidal_dim % 2 != 0:
            learned_sinusoidal_dim += 1
        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
        fourier_dim = learned_sinusoidal_dim + 1

        self.time_mlp = nn.Sequential(
            sinu_pos_emb, nn.Linear(fourier_dim, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim)
        )
        self.time_dim = time_dim

    def forward(self, t):
        shape = t.shape
        t = t.reshape(-1)
        t = self.time_mlp(t)
        t = t.reshape(*shape, self.time_dim)
        return t


def _extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def _linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def _cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion(object):
    def __init__(
        self,
        num_sample_steps=1000,
        loss_fn=F.l1_loss,
        beta_schedule='cosine',
        use_ddim=False,
        ddim_steps=200,
        ddim_eta=1.0,
        device=None,
    ):
        self.loss_fn = loss_fn

        if beta_schedule == 'linear':
            betas = _linear_beta_schedule(num_sample_steps)
        elif beta_schedule == 'cosine':
            betas = _cosine_beta_schedule(num_sample_steps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        assert betas.shape == (num_sample_steps,)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.num_timesteps = int(num_sample_steps)

        if use_ddim:
            assert (
                0 < ddim_steps < num_sample_steps
            ), 'Please specify ddim_steps in the diffuser args, which should be smaller than num_sample_steps.'

        self.use_ddim = use_ddim
        self.ddim_eta = ddim_eta
        self.ddim_steps = ddim_steps

        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

        # helper function to register buffer from float64 to float32
        def register_buffer(name, val):
            setattr(self, name, val.to(dtype=torch.float32).to(device=device))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1.0 - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        register_buffer(
            'posterior_mean_coef2', (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

    def _predict_start_from_noise(self, x_t, t, noise):
        dtype = x_t.dtype
        # noinspection PyUnresolvedReferences
        return (
            _extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape).to(dtype=dtype) * x_t
            - _extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape).to(dtype=dtype) * noise
        )

    def _predict_noise_from_start(self, x_t, t, x0):
        dtype = x_t.dtype
        # noinspection PyUnresolvedReferences
        return (_extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape).to(dtype=dtype) * x_t - x0) / _extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        ).to(dtype=dtype)

    # noinspection PyUnresolvedReferences
    def _q_posterior(self, x_start, x_t, t):
        dtype = x_start.dtype
        posterior_mean = (
            _extract(self.posterior_mean_coef1, t, x_t.shape).to(dtype=dtype) * x_start
            + _extract(self.posterior_mean_coef2, t, x_t.shape).to(dtype=dtype) * x_t
        )
        posterior_variance = _extract(self.posterior_variance, t, x_t.shape).to(dtype=dtype)
        posterior_log_variance_clipped = _extract(self.posterior_log_variance_clipped, t, x_t.shape).to(dtype=dtype)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    @torch.no_grad()
    def sample(
        self, denoise_fn, batch_size: int, image_shape: Tuple[int], device, float_dtype=torch.float32, postproc_fn=None
    ):
        postproc_fn = postproc_fn or (lambda x: x)
        img = torch.randn(size=(batch_size, *image_shape), device=device, dtype=float_dtype)
        if self.use_ddim:
            return self._ddim_sample(denoise_fn, img, postproc_fn)
        return self._ddpm_sample(denoise_fn, img, postproc_fn)

    @torch.no_grad()
    def _ddpm_sample(self, denoise_fn, x_t, postproc_fn):
        steps = list(reversed(range(0, self.num_timesteps)))
        steps = tqdm(steps, desc='ddpm diffusion sampling step')
        for i in steps:
            t = torch.full((x_t.shape[0],), i, device=x_t.device, dtype=torch.long)
            x_start = denoise_fn(x_t, t)
            x_start = postproc_fn(x_start)
            x_t, _, model_log_variance = self._q_posterior(x_start=x_start, x_t=x_t, t=t)
            if i > 0:
                noise = (0.5 * model_log_variance).exp() * torch.randn_like(x_t)
                x_t = x_t + noise
        return x_t

    @torch.no_grad()
    def _ddim_sample(self, denoise_fn, img, postproc_fn):
        dtype = img.dtype
        times = torch.linspace(0.0, self.num_timesteps, steps=self.ddim_steps + 2)[:-1]
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        time_pairs = tqdm(time_pairs, desc='ddim diffusion sampling step')

        for time, time_next in time_pairs:
            time_cond = torch.full((img.shape[0],), time, device=img.device, dtype=torch.long)
            x_start = denoise_fn(img, time_cond)
            x_start = postproc_fn(x_start)
            pred_noise = self._predict_noise_from_start(img, time_cond, x_start)

            # noinspection PyUnresolvedReferences
            alpha, alpha_next = self.alphas_cumprod_prev[time], self.alphas_cumprod_prev[time_next]
            alpha, alpha_next = alpha.to(dtype=dtype), alpha_next.to(dtype=dtype)
            sigma = self.ddim_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = ((1 - alpha_next) - sigma**2).sqrt()

            noise = torch.randn_like(img) if time_next > 0 else 0.0
            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        return img

    def _q_sample(self, x_start, t):
        dtype = x_start.dtype
        noise = torch.randn_like(x_start)
        # noinspection PyUnresolvedReferences
        return (
            _extract(self.sqrt_alphas_cumprod, t, x_start.shape).to(dtype=dtype) * x_start
            + _extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape).to(dtype=dtype) * noise
        )

    def p_losses(self, denoise_fn, x_start):
        b, device = x_start.shape[0], x_start.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device, dtype=torch.long)
        x_t = self._q_sample(x_start=x_start, t=t)
        x_predict = denoise_fn(x_t, t)
        losses = self.loss_fn(input=x_predict, target=x_start, reduction='none')
        return losses, x_predict


class ElucidatedDiffusion(object):
    def __init__(
        self,
        num_sample_steps=32,  # number of sampling steps
        loss_fn=F.mse_loss,
        sigma_min=0.002,  # min noise level
        sigma_max=80,  # max noise level
        sigma_data=0.5,  # standard deviation of data distribution
        rho=7,  # controls the sampling schedule
        p_mean=-1.2,  # mean of log-normal distribution from which noise is drawn for training
        p_std=1.2,  # standard deviation of log-normal distribution from which noise is drawn for training
        s_churn=80,  # parameters for stochastic sampling - depends on dataset, Table 5 in paper
        s_tmin=0.05,
        s_tmax=50,
        s_noise=1.003,
    ):
        self.loss_fn = loss_fn

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        self.rho = rho

        self.P_mean = p_mean
        self.P_std = p_std

        self.num_sample_steps = num_sample_steps  # known as N in the paper

        self.S_churn = s_churn
        self.S_tmin = s_tmin
        self.S_tmax = s_tmax
        self.S_noise = s_noise

    # derived preconditioning params - Table 1

    def _c_skip(self, sigma):
        return (self.sigma_data**2) / (sigma**2 + self.sigma_data**2)

    def _c_out(self, sigma):
        return sigma * self.sigma_data * (self.sigma_data**2 + sigma**2) ** -0.5

    def _c_in(self, sigma):
        return 1 * (sigma**2 + self.sigma_data**2) ** -0.5

    @staticmethod
    def _c_noise(sigma):
        return torch.log(sigma.clamp(min=1e-20)) * 0.25

    # preconditioned network output
    # equation (7) in the paper

    def _precond_net_forward(self, denoise_fn, noised_images, sigma):
        batch, device, dtype = noised_images.shape[0], noised_images.device, noised_images.dtype

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device=device, dtype=dtype)
        padded_sigma = sigma.view(batch, *([1] * len(noised_images.shape[1:])))

        net_out = denoise_fn(self._c_in(padded_sigma) * noised_images, self._c_noise(sigma))
        out = self._c_skip(padded_sigma) * noised_images + self._c_out(padded_sigma) * net_out

        return out

    # sampling

    # sample schedule
    # equation (5) in the paper

    def _sample_schedule(self, num_sample_steps, device, float_dtype):
        inv_rho = 1 / self.rho

        steps = torch.arange(num_sample_steps, device=device, dtype=float_dtype)
        sigmas = (
            self.sigma_max**inv_rho
            + steps / (num_sample_steps - 1) * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho

        sigmas = F.pad(sigmas, (0, 1), value=0.0)  # last step is sigma value of 0.
        return sigmas

    @torch.no_grad()
    def sample(
        self,
        denoise_fn,
        batch_size: int,
        image_shape: Tuple[int, ...],
        device,
        float_dtype=torch.float32,
        postproc_fn=None,
    ):
        # Example: postproc_fn = lambda images: images.clamp(-1., 1.)
        postproc_fn = postproc_fn or (lambda x: x)

        shape = (batch_size, *image_shape)

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma

        sigmas = self._sample_schedule(self.num_sample_steps, device, float_dtype)

        gammas = torch.where(
            (sigmas >= self.S_tmin) & (sigmas <= self.S_tmax),
            min(self.S_churn / self.num_sample_steps, math.sqrt(2) - 1),
            0.0,
        )

        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

        # images is noise at the beginning

        init_sigma = sigmas[0]

        images = init_sigma * torch.randn(shape, device=device, dtype=float_dtype)

        # gradually denoise
        sigmas_and_gammas = tqdm(sigmas_and_gammas, desc='elucidated diffusion sampling step')
        for sigma, sigma_next, gamma in sigmas_and_gammas:
            sigma, sigma_next, gamma = map(lambda t: t.item(), (sigma, sigma_next, gamma))

            eps = self.S_noise * torch.randn(shape, device=device, dtype=float_dtype)  # stochastic sampling

            sigma_hat = sigma + gamma * sigma
            images_hat = images + math.sqrt(sigma_hat**2 - sigma**2) * eps

            model_output = self._precond_net_forward(denoise_fn, images_hat, sigma_hat)
            model_output = postproc_fn(model_output)
            denoised_over_sigma = (images_hat - model_output) / sigma_hat

            images_next = images_hat + (sigma_next - sigma_hat) * denoised_over_sigma

            # second order correction, if not the last timestep

            if sigma_next != 0:
                model_output_next = self._precond_net_forward(denoise_fn, images_next, sigma_next)
                model_output_next = postproc_fn(model_output_next)
                denoised_prime_over_sigma = (images_next - model_output_next) / sigma_next
                images_next = images_hat + 0.5 * (sigma_next - sigma_hat) * (
                    denoised_over_sigma + denoised_prime_over_sigma
                )

            images = images_next

        images = postproc_fn(images)
        return images

    # training

    def _loss_weight(self, sigma):
        return (sigma**2 + self.sigma_data**2) * (sigma * self.sigma_data) ** -2

    def _noise_distribution(self, batch_size, device, float_dtype):
        sigmas = (self.P_mean + self.P_std * torch.randn((batch_size,), device=device, dtype=float_dtype)).exp()
        return sigmas

    def p_losses(self, denoise_fn, x_start):
        batch_size, device, dtype = x_start.shape[0], x_start.device, x_start.dtype

        sigmas = self._noise_distribution(batch_size, device, dtype)
        padded_sigmas = sigmas.view(batch_size, *([1] * len(x_start.shape[1:])))

        noise = torch.randn_like(x_start)
        noised_images = x_start + padded_sigmas * noise  # alphas are 1. in the paper

        x_predict = self._precond_net_forward(denoise_fn, noised_images, sigmas)

        losses = self.loss_fn(input=x_predict, target=x_start, reduction='none')
        losses = losses * self._loss_weight(padded_sigmas)
        return losses, x_predict
