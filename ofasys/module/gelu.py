"""
See "Gaussian Error Linear Units (GELUs)" by Dan Hendrycks and Kevin Gimpel with
the corresponding GitHub repo: https://github.com/hendrycks/GELUs
"""

# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import torch


@torch.jit.script
def gelu_accurate(x):
    return x * 0.5 * (1.0 + torch.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))


def gelu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x.float()).type_as(x)
