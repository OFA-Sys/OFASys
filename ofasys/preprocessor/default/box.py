# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch

from ofasys import ModalityType
from ofasys.configure import register_config
from ofasys.utils import transforms as T

from ..dictionary import Dictionary
from ..instruction import Instruction, Slot
from ..utils import collate_tokens
from .base import CollateOutput, PreprocessConfig, SafeBasePreprocess
from .image import load_image


@dataclass
class BoxPreprocessConfig(PreprocessConfig):
    box_dict_size: int = field(default=1000, metadata={"help": "bounding box dict size"})
    max_image_size: int = field(default=512, metadata={"help": "image size upper bound"})
    # Co-transform image and bounding box  TODO: modify the value correspoinging to image
    patch_image_size: int = field(default=512, metadata={"help": "patch image size"})
    imagenet_default_mean_and_std: bool = field(default=False, metadata={"help": "imagenet normalize"})
    # More image arguments
    # random_resize_upper: Optional[int] = field(default=None, metadata={"random_resize_upper"})
    # random_resize_max_size: Optional[int] = field(default=None, metadata={"random_resize_max_size"})
    # center_crop: bool = field(default=False, metadata={"help": "whether use center_crop"})
    # random_horizontal_flip: bool = field(default=False, metadata={"help": "random_horizontal_flip"})


@register_config("ofasys.preprocess", "box", BoxPreprocessConfig)
class DefaultBoxPreprocess(SafeBasePreprocess):
    def __init__(self, global_dict: Dictionary, cfg: BoxPreprocessConfig):
        super().__init__(global_dict, cfg, ModalityType.BOX)
        self.num_bins = cfg.box_dict_size
        self.max_image_size = cfg.max_image_size
        for i in range(self.num_bins):
            global_dict.add_symbol("<bin>_{}".format(i))
        self.dict_start, self.dict_end = self.global_dict.get_start_end_idx('<bin>')
        assert self.dict_end >= self.dict_start >= 0

        if cfg.imagenet_default_mean_and_std:
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        self.transform = T.Compose(
            [
                T.RandomResize([cfg.patch_image_size], max_size=cfg.patch_image_size),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )

    def instruction_map(self, ist_data: Instruction) -> Instruction:
        slots = ist_data.slots

        def _fetch_modal(mod):
            return [slot for slot in slots if slot.modality == mod]

        image_slot = _fetch_modal(ModalityType.IMAGE)[0]
        box_slot = _fetch_modal(ModalityType.BOX)[0]

        # TODO:remove this after triming tasks of grounding_caption & object_detection
        if box_slot.value is not None and not isinstance(box_slot.value, str):
            return ist_data

        assert image_slot.get_attr('preprocess') is None, (
            f'{self.__class__.__name__} will transform the image and bounding box cooperatively, '
            'which skips the `map` process of the image itself.'
        )

        image = load_image(image_slot.value)
        w, h = image.size
        boxes_target = {"boxes": [], "labels": [], "area": [], "size": torch.tensor([h, w])}

        if slots[0].split == 'test':
            region_coord = '0,0,{},{}'.format(h, w)
        else:
            region_coord = box_slot.value

        x0, y0, x1, y1 = region_coord.strip().split(',')
        boxes_target["boxes"] = torch.tensor([[float(x0), float(y0), float(x1), float(y1)]])
        boxes_target["labels"] = np.array([0])
        boxes_target["area"] = torch.tensor([(float(x1) - float(x0)) * (float(y1) - float(y0))])
        patch_image, patch_boxes = self.transform(image, boxes_target)
        resize_h, resize_w = patch_boxes["size"][0], patch_boxes["size"][1]

        image_slot.value = patch_image
        box_slot.value = patch_boxes["boxes"]
        # TODO: add a `set` method for BasePreprocessor or Instruction
        ist_data.others['__preprocess_decode_kwargs__'] = {
            'w_resize_ratio': resize_w / w,
            'h_resize_ratio': resize_h / h,
        }
        ist_data.others['raw_image'] = image
        return ist_data

    def map(self, slot: Slot) -> Slot:
        super().map(slot)
        patch_boxes = slot.value
        quant_x0 = "<bin>_{}".format(int((patch_boxes[0][0] / self.max_image_size * (self.num_bins - 1)).round()))
        quant_y0 = "<bin>_{}".format(int((patch_boxes[0][1] / self.max_image_size * (self.num_bins - 1)).round()))
        quant_x1 = "<bin>_{}".format(int((patch_boxes[0][2] / self.max_image_size * (self.num_bins - 1)).round()))
        quant_y1 = "<bin>_{}".format(int((patch_boxes[0][3] / self.max_image_size * (self.num_bins - 1)).round()))
        region_coord = "{} {} {} {}".format(quant_x0, quant_y0, quant_x1, quant_y1)
        tokens = self.encode(region_coord)
        if slot.has_attr('add_bos'):
            tokens = torch.cat([torch.LongTensor([self.global_dict.bos()]), tokens])
        if slot.has_attr('add_eos'):
            tokens = torch.cat([tokens, torch.LongTensor([self.global_dict.eos()])])
        slot.value = tokens
        return slot

    def group_key(self, slot: Slot):
        return ModalityType.TEXT

    def encode(self, region_coord):
        tokens = self.global_dict.encode_line(line=region_coord, add_if_not_exist=False, append_eos=False).long()
        return tokens

    def decode(self, tokens, w_resize_ratio, h_resize_ratio):
        region_coord = tokens[:-1] - self.dict_start
        region_coord = region_coord / (self.num_bins - 1) * self.max_image_size
        region_coord[::2] /= w_resize_ratio
        region_coord[1::2] /= h_resize_ratio
        return region_coord

    def collate(self, slots: List[Slot]) -> CollateOutput:
        """
        Inputs:
            samples: List of Tensors after preprocess

        Returns:
            dict:
                src_tokens (Tensor): batched tokens with shape `[batch, seq_len]`
        """
        super().collate(slots)
        slots[0].value = collate_tokens(
            [slot.value for slot in slots],
            pad_idx=self.global_dict.pad(),
            eos_idx=self.global_dict.eos(),
        )
        slot = slots[0]
        if slot.is_src:
            return CollateOutput(slot)
        else:
            input_slot = Slot(
                slot.modality,
                slot.is_src,
                slot.value[:, :-1],
                slot.global_position,
                slot.column_name,
                slot.attributes,
                None,
            )
            target_slot = Slot(
                slot.modality,
                slot.is_src,
                slot.value[:, 1:],
                slot.global_position,
                slot.column_name,
                slot.attributes,
                None,
            )

            # for lagecy compatible
            ntokens = target_slot.value.ne(self.global_dict.pad()).long().sum().item()
            extra_dict = {
                "target": target_slot.value,
                "ntokens": ntokens,
            }
            return CollateOutput(input_slot, target_slot, extra_dict)

    def __call__(self, x):
        raise NotImplementedError
