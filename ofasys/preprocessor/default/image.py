# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import os
import warnings
from dataclasses import dataclass, field
from io import BytesIO
from typing import List, Union

import numpy as np
import requests
import torch
from PIL import Image, ImageFile, ImageOps
from torchvision import transforms

from ofasys.configure import register_config
from ofasys.utils.oss import oss_get
from ofasys.utils.transforms import RandomResize
from ofasys.utils.vision_helper import RandomAugment

from ..instruction import ModalityType, Slot
from ..utils import base64decode
from .base import CollateOutput, PreprocessConfig, SafeBasePreprocess

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

DEFAULT_CROP_PCT = 0.875
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
IMAGENET_DPN_MEAN = (124 / 255, 117 / 255, 104 / 255)
IMAGENET_DPN_STD = tuple([1 / (0.0167 * 255)] * 3)


def load_image(image: Union[str, "PIL.Image.Image"]) -> "PIL.Image.Image":
    """
    Loads `image` to a PIL Image.

    Args:
        image (`str` or `PIL.Image.Image`) could be:
            A remote link starts with `http://` or `https://` or `oss://`;
            A base64 string of image;
            A local file path;
            A PIL.Image.Image object

    Returns:
        `PIL.Image.Image`: A PIL Image.
    """
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            # We need to actually check for a real protocol, otherwise it's impossible to
            # use a local file like http_huggingface_co.png.
            image = Image.open(requests.get(image, stream=True).raw)
        elif image.startswith("oss://"):
            fin = oss_get(image)
            image = Image.open(BytesIO(fin.read()))
            del fin
        elif os.path.exists(image):
            image = Image.open(BytesIO(open(image, 'rb').read()))
        else:
            image_bytes = base64decode(image)
            if image_bytes is not None:
                image = Image.open(BytesIO(image_bytes))
            elif os.path.isfile(image):
                image = Image.open(image)
            else:
                raise ValueError(f"Incorrect format used for image.{load_image.__doc__}Got {image}")
    elif isinstance(image, Image.Image):
        image = image
    else:
        raise ValueError(f"Incorrect format used for image.{load_image.__doc__}Got {image}")
    try:
        image = ImageOps.exif_transpose(image)
    except:
        pass
    image = image.convert("RGB")
    return image


@dataclass
class ImagePreprocessConfig(PreprocessConfig):
    patch_image_size: int = field(default=480, metadata={"help": "patch image size"})
    imagenet_default_mean_and_std: bool = field(default=False, metadata={"help": "imagenet normalize"})
    interpolation: str = field(default='bicubic', metadata={"help": "image interpolation"})


@register_config("ofasys.preprocess", "image", ImagePreprocessConfig)
class DefaultImagePreprocess(SafeBasePreprocess):
    def __init__(self, global_dict, cfg: ImagePreprocessConfig):
        super().__init__(global_dict, cfg, ModalityType.IMAGE)
        if cfg.imagenet_default_mean_and_std:
            self.mean = IMAGENET_DEFAULT_MEAN
            self.std = IMAGENET_DEFAULT_STD
        else:
            self.mean = IMAGENET_INCEPTION_MEAN
            self.std = IMAGENET_INCEPTION_STD
        if cfg.interpolation == 'bicubic':
            self.interpolation = Image.BICUBIC
        else:
            raise ValueError(cfg.interpolation)
        self.patch_image_size = cfg.patch_image_size

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.patch_image_size, self.patch_image_size), interpolation=self.interpolation),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )
        self.train_transform = self.transform

    def map(self, slot: Slot) -> Slot:
        super().map(slot)
        image = slot.value
        if isinstance(image, np.ndarray) or isinstance(image, torch.Tensor):
            # assert image.shape[-2:] == (self.patch_image_size, self.patch_image_size)
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image)
        else:
            image = load_image(image)
            if slot.split == 'train':
                image = self.train_transform(image)
            else:
                image = self.transform(image)

        if slot.get_attr('mask_ratio', float):
            mask_ratio = slot.get_attr('mask_ratio', float)
            segment = int(self.patch_image_size * (1 - mask_ratio) / 2.0)
            image[:, segment:-segment, segment:-segment] = 0.0

        slot.value = image

        return slot

    def collate(self, slots: List[Slot]) -> CollateOutput:
        super().collate(slots)
        slots[0].value = torch.stack([slot.value for slot in slots], dim=0)
        slot = slots[0]
        return CollateOutput(slot)


@register_config("ofasys.preprocess", "imagenet", ImagePreprocessConfig)
class ImagenetImagePreprocess(DefaultImagePreprocess):
    def __init__(self, global_dict, cfg: ImagePreprocessConfig):
        super().__init__(global_dict, cfg)
        from timm.data import create_transform

        self.train_transform = create_transform(
            input_size=self.patch_image_size,
            is_training=True,
            color_jitter=0.4,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation=self.cfg.interpolation,
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
            mean=self.mean,
            std=self.std,
        )
        self.train_transform = transforms.Compose(
            self.train_transform.transforms[:3]
            + [
                RandomAugment(
                    2,
                    7,
                    isPIL=True,
                    augs=[
                        'Identity',
                        'AutoContrast',
                        'Equalize',
                        'Brightness',
                        'Sharpness',
                        'ShearX',
                        'ShearY',
                        'TranslateX',
                        'TranslateY',
                        'Rotate',
                    ],
                ),
            ]
            + self.train_transform.transforms[3:]
        )


@register_config("ofasys.preprocess", "imagepretrain", ImagePreprocessConfig)
class ImagePretrainImagePreprocess(DefaultImagePreprocess):
    def __init__(self, global_dict, cfg: ImagePreprocessConfig):
        super().__init__(global_dict, cfg)
        max_range = int(self.patch_image_size * 1.5) + 1
        scales = np.arange(self.patch_image_size, max_range).tolist()

        self.train_transform = transforms.Compose(
            [
                RandomResize(scales, max_size=672),
                transforms.CenterCrop(self.patch_image_size),
                RandomAugment(
                    2,
                    7,
                    isPIL=True,
                    augs=[
                        'Identity',
                        'AutoContrast',
                        'Equalize',
                        'Brightness',
                        'Sharpness',
                        'ShearX',
                        'ShearY',
                        'TranslateX',
                        'TranslateY',
                        'Rotate',
                    ],
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
