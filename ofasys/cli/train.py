# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import base64
from ofasys import Trainer, TrainerConfig
from ofasys.configure import ConfigStore, options
from ofasys.configure.utils import convert_namespace_to_omegaconf
from ofasys.distributed import utils as distributed_utils
from omegaconf import OmegaConf
from ofasys.model.ofa import GeneralistModelConfig


def main(cfg: TrainerConfig) -> None:
    tasks = [t() for t in ConfigStore().build("ofasys.task")]
    tasks = list(sorted(tasks, key=lambda x: x.name))
    assert len(tasks) > 0
    node: GeneralistModelConfig = ConfigStore().get("ofasys.model", "unify")
    if hasattr(cfg.model, 'extra_models'):
        node.config.extra_models = OmegaConf.merge(
            node.config.extra_models, cfg.model.extra_models)
    model = node.target(node.config)
    trainer = Trainer(cfg)
    trainer.fit(model, tasks)


if __name__ == "__main__":
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)
    ofa_config = OmegaConf.create(base64.b64decode(args.ofasys_complete_config.encode()).decode())
    cfg = convert_namespace_to_omegaconf(args)
    cfg.model = ofa_config.model
    distributed_utils.call_main(cfg, main)
