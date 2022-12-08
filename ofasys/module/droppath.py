""" DropPath

Code impl inspired by
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
"""
# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import torch


class DropPath(torch.nn.Module):
    """
    Drop paths per sample when applied in main path of residual blocks.

    Args:
        drop_prob(float): probability of drop the path per sample. Default: 0.0
        batch_axis(int): the axis of batch (sample). Default: 0
        scale_by_keep(bool): scale the output by keep_prob (1-drop_prob) Default: True

    Input:
        x(Tensor): Input can be of any shape

    Output:
        Tensor: Output is of the same shape as input
    """

    def __init__(self, drop_prob: float = 0.0, batch_axis: int = 0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        if drop_prob < 0 or drop_prob > 1:
            raise ValueError("droppath probability has to be between 0 and 1, " "but got {}".format(drop_prob))
        if batch_axis < 0 or not isinstance(batch_axis, int):
            raise ValueError("droppath batch_axis has to be a natural number, " "but got {}".format(batch_axis))

        self.drop_prob = drop_prob
        self.batch_axis = batch_axis
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        if self.batch_axis >= x.ndim:
            raise ValueError(
                "droppath batch_axis has to be less than input.ndim, "
                "but got {} >= {}".format(self.batch_axis, x.ndim)
            )
        keep_prob = 1 - self.drop_prob
        shape = [1 for i in range(x.ndim)]
        shape[self.batch_axis] = x.shape[self.batch_axis]
        # Update drop_path to be symbolically traceable, slightly faster
        # random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        # if keep_prob > 0.0 and self.scale_by_keep:
        #     random_tensor.div_(keep_prob)
        # return x * random_tensor
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        if keep_prob > 0.0 and self.scale_by_keep:
            x.div_(keep_prob)
        return x * random_tensor

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)
