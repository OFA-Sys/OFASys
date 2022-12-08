# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import warnings

from ofasys.configure import register_config
from ofasys.task.base import OFATask, TaskConfig

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


@register_config("ofasys.task", "video_classify", dataclass=TaskConfig)
class VideoClassifyTask(OFATask):
    pass
