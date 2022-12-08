# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from .base import BatchGeneratorOutput, GeneratorOutput, MultiGeneratorOutput
from .diffusion_generator import DiffusionGenerator, MotionOutput
from .sequence_generator import SequenceGenerator, SequenceGeneratorOutput
from .speech_generator import AutoRegressiveSpeechGenerator, SpeechGeneratorOutput

__all__ = [
    "GeneratorOutput",
    "MultiGeneratorOutput",
    "BatchGeneratorOutput",
    "SequenceGenerator",
    "SequenceGeneratorOutput",
    "AutoRegressiveSpeechGenerator",
    "SpeechGeneratorOutput",
    "DiffusionGenerator",
    "MotionOutput",
]
