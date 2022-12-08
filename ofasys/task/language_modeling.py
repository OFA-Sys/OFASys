# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from ofasys.configure import register_config
from ofasys.task.base import OFATask, TaskConfig


@register_config("ofasys.task", "language_modeling", dataclass=TaskConfig)
class LanguageModelingTask(OFATask):
    pass
