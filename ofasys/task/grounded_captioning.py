# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from typing import Any, Dict

import numpy as np
import torch

from ofasys.configure import register_config
from ofasys.preprocessor.default.image import load_image
from ofasys.task.base import OFATask, TaskConfig
from ofasys.utils import transforms as T


@register_config("ofasys.task", "grounded_captioning", dataclass=TaskConfig)
class GroundedCaptioningTask(OFATask):
    def __init__(self, cfg: TaskConfig, **kwargs):
        super().__init__(cfg, **kwargs)

        if cfg.preprocess.image.imagenet_default_mean_and_std:
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        patch_image_size = cfg.preprocess.image.patch_image_size
        scales = np.arange(patch_image_size, 481).tolist()
        self.transform = T.Compose(
            [
                T.RandomResize(scales, max_size=672),
                T.ObjectCenterCrop((patch_image_size, patch_image_size)),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )

    def preprocess(self, data: Dict[str, Any], split: str) -> Dict[str, Any]:
        base64_str, region_coord = data['img'], data['region_coord']

        image = load_image(base64_str)
        w, h = image.size
        boxes_target = {"boxes": [], "labels": [], "area": [], "size": torch.tensor([h, w])}
        x0, y0, x1, y1 = region_coord.strip().split(',')
        boxes_target["boxes"] = torch.tensor([[float(x0), float(y0), float(x1), float(y1)]])
        boxes_target["labels"] = np.array([0])
        boxes_target["area"] = torch.tensor([(float(x1) - float(x0)) * (float(y1) - float(y0))])
        patch_image, patch_boxes = self.transform(image, boxes_target)

        data['img'] = patch_image
        data['patch_boxes'] = patch_boxes["boxes"]
        return data
