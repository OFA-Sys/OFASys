# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from ofasys.configure import register_config
from ofasys.task.caption import CaptionTask, TaskConfig


@register_config("ofasys.task", "video_caption", dataclass=TaskConfig)
class VideoCaptionTask(CaptionTask):
    pass
