# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from ofasys import Trainer, TrainerConfig
from ofasys.configure import ConfigStore, options
from ofasys.configure.utils import convert_namespace_to_omegaconf
from ofasys.distributed import utils as distributed_utils


def main(cfg: TrainerConfig) -> None:
    tasks = [t() for t in ConfigStore().build("ofasys.task")]
    tasks = list(sorted(tasks, key=lambda x: x.name))
    assert len(tasks) > 0
    model = ConfigStore().build("ofasys.model", "unify")()
    trainer = Trainer(cfg)
    trainer.fit(model, tasks)


if __name__ == "__main__":
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)
    cfg = convert_namespace_to_omegaconf(args)
    distributed_utils.call_main(cfg, main)
