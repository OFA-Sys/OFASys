# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import random
from itertools import chain
from typing import Any, Dict

import numpy as np
import torch

from ofasys.configure import register_config
from ofasys.preprocessor import Instruction
from ofasys.preprocessor.default.image import load_image
from ofasys.task.base import OFATask, TaskConfig
from ofasys.utils import transforms as T


@register_config("ofasys.task", "object_detection", dataclass=TaskConfig)
class ObjectDetectionTask(OFATask):
    def __init__(self, cfg: TaskConfig, **kwargs):
        super().__init__(cfg, **kwargs)

        if cfg.preprocess.image.imagenet_default_mean_and_std:
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        patch_image_size = cfg.preprocess.image.patch_image_size
        self.transform = T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.LargeScaleJitter(output_size=patch_image_size, aug_scale_min=1.0, aug_scale_max=1.5),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )

    def preprocess(self, data: Dict[str, Any], split: str) -> Instruction:
        base64_str, label = data['img'], data['label']

        image = load_image(base64_str)

        w, h = image.size
        boxes_target = {"boxes": [], "labels": [], "area": [], "size": torch.tensor([h, w])}
        label_list = label.strip().split('&&')
        for label in label_list:
            x0, y0, x1, y1, cat_id, cat = label.strip().split(',', 5)
            boxes_target["boxes"].append([float(x0), float(y0), float(x1), float(y1)])
            boxes_target["labels"].append(cat)
            boxes_target["area"].append((float(x1) - float(x0)) * (float(y1) - float(y0)))
        boxes_target["boxes"] = torch.tensor(boxes_target["boxes"])
        boxes_target["labels"] = np.array(boxes_target["labels"])
        boxes_target["area"] = torch.tensor(boxes_target["area"])
        patch_image, boxes_target = self.transform(image, boxes_target)

        data['img'] = patch_image
        data['boxes_target'] = boxes_target
        return data

    def build_instruction(self, data: Dict[str, Any], split: str) -> Instruction:
        def get_template():
            if len(self.templates) > 1:
                template = random.sample(self.templates, k=1)[0]
            else:
                template = self.templates[0]
            return template

        template = get_template()
        boxes_target = data['boxes_target']
        new_slots = ' [BOX] [TEXT]' * len(boxes_target['boxes'])
        template = template.replace('( [BOX] [TEXT])*', new_slots)
        ist = Instruction(template, split=split, decoder_plain_with_loss=self.cfg.instruction.decoder_plain_with_loss)

        object_data = [
            [box.unsqueeze(0), box_label] for box, box_label in zip(boxes_target["boxes"], boxes_target["labels"])
        ]
        object_data = tuple(chain(*object_data))
        return ist.format(data['img'], *object_data)
