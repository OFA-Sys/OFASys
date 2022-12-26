# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from .audio import DefaultAudioEmbedPreprocess, DefaultAudioPreprocess
from .box import DefaultBoxPreprocess
from .image import DefaultImagePreprocess
from .image_code import VQGANCodePreprocess
from .motion_6d import Motion6dPreprocess
from .phone import DefaultPhonePreprocess
from .struct import DefaultStructPreprocess
from .text import DefaultTextPreprocess
from .category import CategoryPreprocess
from .video import DefaultVideoPreprocess

__all__ = [
    'DefaultTextPreprocess',
    'CategoryPreprocess',
    'DefaultImagePreprocess',
    'VQGANCodePreprocess',
    'DefaultBoxPreprocess',
    'DefaultAudioPreprocess',
    'DefaultAudioEmbedPreprocess',
    'Motion6dPreprocess',
    'DefaultPhonePreprocess',
    'DefaultVideoPreprocess',
    'DefaultStructPreprocess',
]
