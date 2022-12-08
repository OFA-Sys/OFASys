# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from pathlib import Path

import numpy as np

from ofasys.utils.audio_feature_transforms import (
    AudioFeatureTransform,
    register_audio_feature_transform,
)
from ofasys.utils.file_utils import cached_path


@register_audio_feature_transform("ofa_global_cmvn")
class OFAGlobalCMVN(AudioFeatureTransform):
    """Global CMVN (cepstral mean and variance normalization). The global mean
    and variance need to be pre-computed and stored in NumPy format (.npz)."""

    @classmethod
    def from_config_dict(cls, config=None):
        _config = {} if config is None else config
        local_stats_npz_path = cached_path(_config.get("stats_npz_path"))
        return OFAGlobalCMVN(Path(local_stats_npz_path))

    def __init__(self, stats_npz_path):
        self.stats_npz_path = stats_npz_path
        stats = np.load(stats_npz_path)
        self.mean, self.std = stats["mean"], stats["std"]

    def __repr__(self):
        return self.__class__.__name__ + f'(stats_npz_path="{self.stats_npz_path}")'

    def __call__(self, x):
        x = np.subtract(x, self.mean)
        x = np.divide(x, self.std)
        return x
