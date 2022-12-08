# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import numpy as np

from ofasys.utils.audio_feature_transforms import (
    AudioFeatureTransform,
    register_audio_feature_transform,
)


@register_audio_feature_transform("utterance_cmvn")
class UtteranceCMVN(AudioFeatureTransform):
    """Utterance-level CMVN (cepstral mean and variance normalization)"""

    @classmethod
    def from_config_dict(cls, config=None):
        _config = {} if config is None else config
        return UtteranceCMVN(
            _config.get("norm_means", True),
            _config.get("norm_vars", True),
        )

    def __init__(self, norm_means=True, norm_vars=True):
        self.norm_means, self.norm_vars = norm_means, norm_vars

    def __repr__(self):
        return self.__class__.__name__ + f"(norm_means={self.norm_means}, norm_vars={self.norm_vars})"

    def __call__(self, x):
        mean = x.mean(axis=0)
        square_sums = (x**2).sum(axis=0)

        if self.norm_means:
            x = np.subtract(x, mean)
        if self.norm_vars:
            var = square_sums / x.shape[0] - mean**2
            std = np.sqrt(np.maximum(var, 1e-10))
            x = np.divide(x, std)

        return x
