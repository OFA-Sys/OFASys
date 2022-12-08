# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from typing import List, Union

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from PIL.Image import Image as pilImage
from torch import Tensor

from ofasys.module.taming.models.vqgan import GumbelVQ
from ofasys.utils.file_utils import cached_path


def custom_to_pil(x):
    x = x.detach()
    x = torch.clamp(x, -1.0, 1.0)
    x = (x + 1.0) / 2.0
    x = x.permute(1, 2, 0).cpu().numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


class VQGANTokenizer(object):
    def __init__(self, vqgan_config_path, vqgan_model_path, code_image_size, vqgan_factor):

        local_config_path = cached_path(vqgan_config_path)
        vqgan_config = OmegaConf.load(local_config_path)
        vqgan = GumbelVQ(**vqgan_config.model.params)

        local_model_path = cached_path(vqgan_model_path)
        sd = torch.load(local_model_path, map_location="cpu")["state_dict"]
        missing, unexpected = vqgan.load_state_dict(sd, strict=False)
        for k, v in vqgan.named_parameters():
            v.requires_grad = False
        self.vqgan = vqgan
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        self.vqgan.to(self.device)
        self.vqgan.eval()
        self.code_image_size = code_image_size
        self.vqgan_factor = vqgan_factor

    def encode(self, x: Tensor, **kwargs) -> Tensor:
        batch_size = x.size()[0]
        x = x.to(self.device)
        with torch.no_grad():
            z, _, [_, _, image_codes] = self.vqgan.encode(x)
            image_codes = image_codes.view(batch_size, -1).detach()
        return image_codes

    def decode(self, tokens: Tensor, return_pil=True, **kwargs) -> Union[List[pilImage], Tensor]:

        tokens = tokens[:, :-1]
        # avoid tokens not in code dict
        l_bound = torch.zeros_like(tokens)
        tokens = torch.where(tokens.lt(l_bound), l_bound, tokens)
        h_bound = torch.ones_like(tokens) * (self.vqgan.vocab_size - 1)
        tokens = torch.where(tokens.gt(h_bound), h_bound, tokens)

        tokens = tokens.view(-1, self.code_image_size // self.vqgan_factor, self.code_image_size // self.vqgan_factor)
        with torch.no_grad():
            images = self.vqgan.decode_code(tokens.contiguous().to(self.device))
        if return_pil:
            images = [custom_to_pil(image) for image in images]
        return images
