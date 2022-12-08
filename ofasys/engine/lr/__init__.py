# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from omegaconf import DictConfig

from ofasys.configure import auto_import, registry

from .fairseq_lr_scheduler import FairseqLRScheduler, LegacyFairseqLRScheduler

(
    build_lr_scheduler_,
    register_lr_scheduler,
    LR_SCHEDULER_REGISTRY,
    LR_SCHEDULER_DATACLASS_REGISTRY,
) = registry.setup_registry("--lr-scheduler", base_class=FairseqLRScheduler, default="fixed")


def build_lr_scheduler(cfg: DictConfig, optimizer):
    return build_lr_scheduler_(cfg, optimizer)


auto_import(__file__)
