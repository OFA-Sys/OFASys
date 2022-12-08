# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import random
from dataclasses import dataclass, field
from typing import Any, Dict

from ofasys.configure import register_config
from ofasys.task.base import OFATask, TaskConfig
from ofasys.utils.file_utils import cached_path


@dataclass
class ImageTextMatchingConfig(TaskConfig):
    all_captions: str = field(
        default='oss://shuangqing-multimodal/ofa/data/all_captions.txt',
        metadata={"help": 'directory for negative samples'},
    )


@register_config("ofasys.task", "image_text_matching", dataclass=ImageTextMatchingConfig)
class ImageTextMatchingTask(OFATask):
    def __init__(self, cfg: ImageTextMatchingConfig, **kwargs):
        super().__init__(cfg, **kwargs)
        local_path = cached_path(cfg.all_captions)
        self.all_caption_list = [row.strip() for row in open(local_path) if row.strip() != '']

    def preprocess(self, data: Dict[str, Any], split: str) -> Dict[str, Any]:
        if random.random() < 0.5:
            caption = data['caption']
            label = 'yes'
        else:
            caption = random.choice(self.all_caption_list)
            label = 'no'
        data['caption'] = caption
        data['label'] = label
        return data
