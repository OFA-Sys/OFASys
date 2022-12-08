# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import codecs
import logging
import os
import sys
from enum import Enum, unique


# We need to setup root logger before importing any libraries.
logging.basicConfig(
    format="%(asctime)s - %(name)s@%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("ofasys")


@unique
class ModalityType(Enum):
    TEXT = 1
    IMAGE = 2
    BOX = 3
    AUDIO = 4
    MOTION = 5
    PHONE = 6
    VIDEO = 7
    STRUCT = 8

    @classmethod
    def parse(cls, mark):
        for mod in ModalityType:
            if mark == mod.name:
                return cls(mod.value)
        return None


from . import (
    adaptor,
    configure,
    engine,
    io,
    metric,
    model,
    module,
    preprocessor,
    task,
    utils,
)
from .configure import BaseDataclass, TrainerConfig, register_config, ConfigStore
from .engine import Trainer
from .hub_interface import OFASys
from .model import BaseModel, GeneralistModel
from .preprocessor import Instruction, Slot
from .task import OFATask as Task
from .templates import TaskTemplates

__all__ = [
    'ModalityType',
    'BaseDataclass',
    'Instruction',
    'ConfigStore',
    'register_config',
    'io',
    'preprocessor',
    'module',
    'adaptor',
    'model',
    'engine',
    'utils',
    'task',
    'metric',
    'configure',
    'OFASys',
    'TaskTemplates',
    'Task',
    'BaseModel',
    'GeneralistModel',
    'Trainer',
    'TrainerConfig',
]
