# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import contextlib
import copy
import gc
import logging
import math
import os
import random
import sys
import time
from argparse import Namespace
from dataclasses import asdict
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.profiler as profiler
from omegaconf import DictConfig, OmegaConf

from ofasys.configure import ConfigStore, TrainerConfig
from ofasys.configure.utils import convert_namespace_to_omegaconf
from ofasys.distributed import DistributedModelDispatcher, fsdp_enable_wrap, fsdp_wrap
from ofasys.distributed import utils as distributed_utils
from ofasys.engine import lr as lr_scheduler
from ofasys.engine import optim
from ofasys.engine.ema import EMA
from ofasys.logging import meters, metrics, progress_bar
from ofasys.model import BaseModel
from ofasys.module import utils
from ofasys.preprocessor.dictionary import Dictionary
from ofasys.task import OFATask
from ofasys.utils import checkpoint_utils
from ofasys.utils.file_io import PathManager
from ofasys.utils.oss import oss_exists

from .nan_detector import NanDetector

logger = logging.getLogger(__name__)


class Trainer(object):
    """Main class for data parallel training.

    This class supports synchronous distributed data parallel training,
    where multiple workers each have a full model replica and gradients
    are accumulated across workers before each update. We use
    :class:`~torch.nn.parallel.DistributedDataParallel` to handle
    communication of the gradients across workers.
    """

    def __init__(self, cfg: TrainerConfig = None):
        if cfg is None:
            cfg = TrainerConfig.from_yaml(
                os.path.join(
                    os.path.dirname(__file__),
                    '..',
                    'config',
                    'default_trainer.yaml',
                )
            )
        self.cfg = cfg

    @classmethod
    def from_yaml(cls, yaml_path):
        return Trainer(TrainerConfig.from_yaml(yaml_path))

    def fit(self, model: BaseModel, tasks: List[OFATask]):
        cfg = self.cfg
        if torch.cuda.device_count() == 0:
            cfg.common.fp16 = False
            cfg.common.cpu = True

        metrics.reset()

        if cfg.common.log_file is not None:
            handler = logging.FileHandler(filename=cfg.common.log_file)
            logger.addHandler(handler)

        random.seed(cfg.common.seed)
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

        if distributed_utils.is_master(cfg.distributed_training):
            checkpoint_utils.verify_checkpoint_directory(cfg.checkpoint.save_dir)

        # Print args
        _print_conf(cfg)

        if cfg.checkpoint.write_checkpoints_asynchronously:
            try:
                import iopath  # noqa: F401
            except ImportError:
                logging.exception(
                    "Asynchronous checkpoint writing is specified but iopath is not installed: `pip install iopath`"
                )
                return

        # Setup task
        self.global_dict = Dictionary()
        for task in tasks:
            task.initialize(self.global_dict, is_train=True)

        # Build model
        OFATask.upgrade_model_adaptor_cfg(tasks, model.cfg)
        model.initialize(self.global_dict)
        OFATask._model = model

        # Wrap with FSDP
        if cfg.distributed_training.ddp_backend == "fully_sharded":
            with fsdp_enable_wrap(cfg.distributed_training):
                model = fsdp_wrap(model)

        logger.info(f"Model Structure:\n{model}")
        logger.info("tasks: {}".format([task.__class__.__name__ for task in tasks]))
        logger.info("model: {}".format(model.__class__.__name__))
        logger.info("criterions: {}".format([task.criterion.__class__.__name__ for task in tasks]))
        logger.info(
            "num. shared model params: {:,} (num. trained: {:,})".format(
                sum(p.numel() for p in model.parameters() if not getattr(p, "expert", False)),
                sum(p.numel() for p in model.parameters() if not getattr(p, "expert", False) and p.requires_grad),
            )
        )

        logger.info(
            "num. expert model params: {} (num. trained: {})".format(
                sum(p.numel() for p in model.parameters() if getattr(p, "expert", False)),
                sum(p.numel() for p in model.parameters() if getattr(p, "expert", False) and p.requires_grad),
            )
        )

        if len(tasks) > 1:
            cfg.checkpoint.no_epoch_checkpoints = True
            cfg.checkpoint.save_interval = -1
            logger.info(
                "multi-task training does not support save by epoch yet, no_epoch_checkpoints is True by default."
            )

        # setup trainer
        assert cfg.common.model_parallel_size == 1, "Do not support MegatronTrainer for now."
        self._setup(model, tasks)
        logger.info("training on {} devices (GPUs/TPUs)".format(cfg.distributed_training.distributed_world_size))

        # Load the latest checkpoint if one is available and restore the
        # corresponding train iterator
        extra_state = checkpoint_utils.load_checkpoint(
            cfg.checkpoint,
            self,
            # don't cache epoch iterators for sharded datasets
            disable_iterator_cache=True,
        )

        self.criterion_wrapper()
        self.lr_reinitialize()
        lr = self.get_lr()

        train_meter = meters.StopwatchMeter()
        train_meter.start()

        valid_losses, should_stop = train(cfg, self, extra_state)

        train_meter.stop()
        logger.info("done training in {:.1f} seconds".format(train_meter.sum))

        # ioPath implementation to wait for all asynchronous file writes to complete.
        if cfg.checkpoint.write_checkpoints_asynchronously:
            logger.info("ioPath PathManager waiting for all asynchronous checkpoint " "writes to finish.")
            PathManager.async_close()
            logger.info("ioPath PathManager finished waiting.")

    def _setup(self, model, tasks):
        assert not hasattr(self, 'tasks')
        self.tasks = tasks
        cfg = self.cfg
        if isinstance(cfg, Namespace):
            logger.warning("argparse.Namespace configuration is deprecated! Automatically converting to OmegaConf")
            cfg = convert_namespace_to_omegaconf(cfg)

        self.cfg = cfg
        self.max_epoch = self.cfg.optimization.max_epoch or math.inf

        # catalog shared parameters
        shared_params = _catalog_shared_params(model)
        self.cuda = torch.cuda.is_available() and not cfg.common.cpu

        if self.cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if self.is_fsdp:
            import fairscale

            if self.cfg.common.bf16:
                raise ValueError("FullyShardedDataParallel is not compatible with --bf16 or --memory-efficient-bf16")
            if self.cfg.distributed_training.zero_sharding != "none":
                raise ValueError(
                    "FullyShardedDataParallel is not compatible with --zero-sharding option (it's already built in)"
                )
            if max(self.cfg.optimization.update_freq) > 1 and fairscale.__version__ < "0.4.0":
                raise RuntimeError(
                    "Please update to fairscale 0.4.0 or newer when combining "
                    "--update-freq with FullyShardedDataParallel"
                )
        else:
            if hasattr(self.cfg.distributed_training, "cpu_offload") and self.cfg.distributed_training.cpu_offload:
                raise ValueError("--cpu-offload requires --ddp-backend=fully_sharded")

        # copy model to current device/dtype
        self._model = model
        if not self.is_fsdp:
            if cfg.common.fp16:
                assert not cfg.common.amp, "Cannot use fp16 and AMP together"
                self._model = self._model.half()
            elif cfg.common.bf16:
                self._model = self._model.to(dtype=torch.bfloat16)
            elif cfg.common.amp:
                self._amp_retries = 0
        if (
            not cfg.distributed_training.pipeline_model_parallel
            # the DistributedModelDispatcher wrapper will handle moving to device,
            # so only handle cases which don't use the wrapper
            and not self.use_distributed_wrapper
        ):
            self._model = self._model.to(device=self.device)
        self.pipeline_model_parallel = cfg.distributed_training.pipeline_model_parallel
        self.last_device = None
        if self.cuda and self.pipeline_model_parallel:
            self.last_device = torch.device(cfg.distributed_training.pipeline_devices[-1])

        # check that shared parameters are preserved after device transfer
        for shared_param in shared_params:
            ref = _get_module_by_path(self._model, shared_param[0])
            for path in shared_param[1:]:
                logger.info("detected shared parameter: {} <- {}".format(shared_param[0], path))
                _set_module_by_path(self._model, path, ref)

        self._dummy_batch = {}  # indicates we don't have a dummy batch at first for each task
        self._lr_scheduler = None
        self._num_updates = 0
        self._optim_history = None
        self._optimizer = None
        self._warn_once = set()
        self._wrapped_model = None
        self._ema = None

        if self.cuda and self.data_parallel_world_size > 1:
            self._grad_norm_buf = torch.cuda.DoubleTensor(self.data_parallel_world_size)
        else:
            self._grad_norm_buf = None

        # get detailed cuda environment
        if self.cuda:
            self.cuda_env = utils.CudaEnvironment()
            if self.data_parallel_world_size > 1:
                self.cuda_env_arr = distributed_utils.all_gather_list(
                    self.cuda_env, group=distributed_utils.get_global_group()
                )
            else:
                self.cuda_env_arr = [self.cuda_env]
            if self.data_parallel_rank == 0:
                utils.CudaEnvironment.pretty_print_cuda_env_list(self.cuda_env_arr)
        else:
            self.cuda_env = None
            self.cuda_env_arr = None

        metrics.log_start_time("wall", priority=790, round=1)

        self._start_time = time.time()
        self._previous_training_time = 0
        self._cumulative_training_time = None

    def reinitialize(self):
        """Reinitialize the Trainer, typically after model params change."""
        self._lr_scheduler = None
        self._optimizer = None
        self._wrapped_model = None

    def criterion_wrapper(self):
        for task in self.tasks:
            if not self.is_fsdp:
                if self.cfg.common.fp16:
                    assert not self.cfg.common.amp, "Cannot use fp16 and AMP together"
                    task.criterion = task.criterion.half()
                elif self.cfg.common.bf16:
                    task.criterion = task.criterion.to(dtype=torch.bfloat16)
            if (
                not self.cfg.distributed_training.pipeline_model_parallel
                # the DistributedModelDispatcher wrapper will handle moving to device,
                # so only handle cases which don't use the wrapper
                and not self.use_distributed_wrapper
            ):
                task.criterion = task.criterion.to(device=self.device)

    @property
    def data_parallel_world_size(self):
        if self.cfg.distributed_training.distributed_world_size == 1:
            return 1
        return distributed_utils.get_data_parallel_world_size()

    @property
    def data_parallel_process_group(self):
        return distributed_utils.get_data_parallel_group()

    @property
    def data_parallel_rank(self):
        if self.cfg.distributed_training.distributed_world_size == 1:
            return 0
        return distributed_utils.get_data_parallel_rank()

    @property
    def is_data_parallel_master(self):
        # NOTE: this returns true for all model parallel replicas with data
        # parallel rank 0
        return self.data_parallel_rank == 0

    @property
    def use_distributed_wrapper(self) -> bool:
        return (self.data_parallel_world_size > 1 and not self.cfg.optimization.use_bmuf) or (
            self.is_fsdp and self.cfg.distributed_training.cpu_offload
        )

    @property
    def should_save_checkpoint_on_current_rank(self) -> bool:
        """Indicates whether to save checkpoints on the current DDP rank."""
        if (self.is_fsdp and self.cfg.distributed_training.use_sharded_state) or getattr(
            self.cfg.model, "base_layers", 0
        ) > 0:
            return True
        else:
            return self.is_data_parallel_master

    @property
    def always_call_state_dict_during_save_checkpoint(self) -> bool:
        if self.is_fsdp and not self.cfg.distributed_training.use_sharded_state:
            # FSDP calls communication collective when consolidating checkpoints
            return True
        else:
            return False

    @property
    def checkpoint_suffix(self) -> str:
        """Suffix to add to the checkpoint file name."""
        if self.is_fsdp and self.cfg.distributed_training.use_sharded_state:
            return self.cfg.checkpoint.checkpoint_suffix + "-shard{0}".format(self.data_parallel_rank)
        else:
            return self.cfg.checkpoint.checkpoint_suffix or ""

    @property
    def model(self):
        if self._wrapped_model is None:
            if self.use_distributed_wrapper:
                self._wrapped_model = DistributedModelDispatcher(
                    self.cfg.distributed_training,
                    self._model,
                    process_group=self.data_parallel_process_group,
                    device=self.device,
                )
            else:
                self._wrapped_model = self._model
        return self._wrapped_model

    @property
    def ema(self):
        if self._ema is None:
            self._build_ema()
        return self._ema

    def _build_ema(self):
        if self.cfg.ema.store_ema:
            self._ema = EMA(self._model, self.cfg.ema, self.device)
            logger.info("Exponential Moving Average Shadow Model is initialized.")

    @property
    def optimizer(self):
        if self._optimizer is None:
            self._build_optimizer()
        return self._optimizer

    @property
    def lr_scheduler(self):
        if self._lr_scheduler is None:
            self._build_optimizer()  # this will initialize self._lr_scheduler
        return self._lr_scheduler

    def _build_optimizer(self):
        params = list(
            filter(
                lambda p: p.requires_grad,
                chain(self.model.parameters(), *[task.criterion.parameters() for task in self.tasks]),
            )
        )

        if self.is_fsdp and self.cfg.common.fp16:
            # FullyShardedDataParallel always uses MemoryEfficientFP16 wrapper,
            # mostly for the grad scaling. But if we don't have the
            # --memory-efficient-fp16 flag set, then we're effectively doing
            # regular --fp16 and can allow the use of optimizers that would
            # otherwise be unsupported by MemoryEfficientFP16Optimizer.
            allow_unsupported = not self.cfg.common.memory_efficient_fp16
            self._optimizer = optim.MemoryEfficientFP16Optimizer.build_optimizer(
                self.cfg, params, allow_unsupported=allow_unsupported
            )
        elif self.cfg.common.fp16 or self.cfg.common.bf16 or self.cfg.common.amp:
            if self.cuda and torch.cuda.get_device_capability(0)[0] < 7:
                logger.info(
                    "NOTE: your device does NOT support faster training with --fp16 "
                    "or --amp, please switch to FP32 which is likely to be faster"
                )
            if self.cfg.common.memory_efficient_fp16 or self.cfg.common.memory_efficient_bf16:
                self._optimizer = optim.MemoryEfficientFP16Optimizer.build_optimizer(self.cfg, params)
            elif self.cfg.common.amp:
                self._optimizer = optim.AMPOptimizer.build_optimizer(self.cfg, params)
            else:
                self._optimizer = optim.FP16Optimizer.build_optimizer(self.cfg, params)
        else:
            if self.cuda and torch.cuda.get_device_capability(0)[0] >= 7:
                logger.info("NOTE: your device may support faster training with --fp16 or --amp")
            self._optimizer = optim.build_optimizer(self.cfg.optimizer, params)

        if self.is_fsdp:
            assert not self.cfg.optimization.use_bmuf, "--ddp-backend=fully_sharded is not compatible with BMUF"
            assert self._optimizer.supports_flat_params, (
                "--ddp-backend=fully_sharded is only compatible with pointwise "
                "optimizers (e.g., Adam, AdamW, Adadelta, Adamax, SGD, etc.). "
                "However, the sharding will result in slightly different results when "
                "using non-pointwise optimizers (e.g., Adagrad, Adafactor, LAMB)"
            )

        if self.cfg.optimization.use_bmuf:
            self._optimizer = optim.FairseqBMUF(
                self.cfg.bmuf,
                self._optimizer,
            )

        if self.cfg.distributed_training.zero_sharding == "os":
            if (
                self.cfg.common.fp16
                and not self.cfg.common.memory_efficient_fp16
                and not self.cfg.common.memory_efficient_bf16
            ) and not self.cfg.common.fp16_no_flatten_grads:
                raise ValueError(
                    "ZeRO is incomptabile with fp16 and flattened grads. Please use --fp16-no-flatten-grads"
                )
            else:
                optim.shard_(self._optimizer, self.data_parallel_process_group)

        # We should initialize the learning rate scheduler immediately after
        # building the optimizer, so that the initial learning rate is set.
        self._lr_scheduler = lr_scheduler.build_lr_scheduler(
            self.cfg.lr_scheduler,
            self.optimizer,
        )
        self._lr_scheduler.step_update(0)

    @property
    def is_fsdp(self):
        return self.cfg.distributed_training.ddp_backend == "fully_sharded"

    def consolidate_optimizer(self):
        """For OSS, we need to consolidate the state dict."""
        if self.cfg.checkpoint.no_save_optimizer_state:
            return
        self._gathered_optim_state = None
        if hasattr(self.optimizer.optimizer, "consolidate_state_dict"):
            self.optimizer.optimizer.consolidate_state_dict()
        elif self.is_fsdp and not self.model.use_sharded_state:
            st = self.model.gather_full_optim_state_dict(self.optimizer)  # only returns on rank 0
            self._gathered_optim_state = st

    def state_dict(self):
        state_dict = {
            "args": None,  # legacy
            "cfg": (
                OmegaConf.to_container(self.cfg, resolve=True, enum_to_str=True)
                if OmegaConf.is_config(self.cfg)
                else self.cfg
            ),
            "configstore": {
                "model": {name: asdict(node.config) for name, node in ConfigStore().get_dict("ofasys.model").items()},
                "task": {name: asdict(node.config) for name, node in ConfigStore().get_dict("ofasys.task").items()},
            },
            "model": self.model.state_dict(),
            "optimizer_history": (self._optim_history or [])
            + [
                {
                    "optimizer_name": self.optimizer.__class__.__name__,
                    "lr_scheduler_state": self.lr_scheduler.state_dict(),
                    "num_updates": self.get_num_updates(),
                }
            ],
            # "task_state": self.task.state_dict() if self.task is not None else {},
            "extra_state": {
                "metrics": metrics.state_dict(),
                "previous_training_time": self.cumulative_training_time(),
            },
            "global_dict_indices": self.model.global_dict.indices,
        }
        if self.cfg.ema.store_ema:
            # Save EMA model state as extra state
            state_dict["extra_state"]["ema"] = self.ema.get_model().state_dict()
            if self.cfg.ema.ema_fp32:
                # Save EMA params in fp32
                state_dict["extra_state"]["ema_fp32_params"] = self.ema.fp32_params
        if not self.cfg.checkpoint.no_save_optimizer_state:
            if self._gathered_optim_state is not None:
                state_dict["last_optimizer_state"] = self._gathered_optim_state
                self._gathered_optim_state = None
            else:
                state_dict["last_optimizer_state"] = self.optimizer.state_dict()
        if self.is_fsdp:
            # save meta data for recombining checkpoint upon loading
            state_dict["fsdp_metadata"] = self.model.local_metadata_dict()
        return state_dict

    def save_checkpoint(self, filename, extra_state):
        """Save all training state in a checkpoint file."""
        logger.info(f"Saving checkpoint to {filename}")
        # call state_dict on all ranks in case it needs internal communication
        state_dict = utils.move_to_cpu(self.state_dict())
        state_dict["extra_state"].update(extra_state)
        if self.should_save_checkpoint_on_current_rank:
            checkpoint_utils.torch_persistent_save(
                state_dict,
                filename,
                async_write=self.cfg.checkpoint.write_checkpoints_asynchronously,
            )
        logger.info(f"Finished saving checkpoint to {filename}")

    def load_checkpoint(
        self,
        filename,
        reset_optimizer=False,
        reset_lr_scheduler=False,
        optimizer_overrides=None,
        reset_meters=False,
    ):
        """
        Load all training state from a checkpoint file.
        rank = 0 will load the checkpoint, and then broadcast it to all
        other ranks.
        """
        use_ema_weights_to_init_param = False
        use_latest_weights_to_init_ema = False
        extra_state, self._optim_history, last_optim_state = None, [], None

        logger.info(f"Preparing to load checkpoint {filename}")
        is_distributed = self.data_parallel_world_size > 1
        if filename.startswith('oss://'):
            bexists = oss_exists(filename)
        else:
            bexists = PathManager.isfile(filename)
        if bexists:
            load_on_all_ranks = (
                self.cfg.checkpoint.load_checkpoint_on_all_dp_ranks
                # FSDP requires loading checkpoint shards on all ranks
                or (self.is_fsdp and self.cfg.distributed_training.use_sharded_state)
                or getattr(self.cfg.model, "base_layers", 0) > 0
            )

            if load_on_all_ranks or self.data_parallel_rank == 0:
                state = checkpoint_utils.load_checkpoint_to_cpu(filename, load_on_all_ranks=load_on_all_ranks)
                last_optim_state = state.get("last_optimizer_state", None)

                # If doing zero_sharding, do not broadcast global optimizer
                # state. Later we will broadcast sharded states to each rank
                # to avoid memory from exploding.
                if (
                    not load_on_all_ranks
                    and self.cfg.distributed_training.zero_sharding == "os"
                    and "last_optimizer_state" in state
                    and is_distributed
                ):
                    state["last_optimizer_state"] = "SHARDED"
            else:
                last_optim_state = None
                state = None

            if is_distributed and not load_on_all_ranks:
                state = distributed_utils.broadcast_object(
                    state,
                    src_rank=0,
                    group=self.data_parallel_process_group,
                    dist_device=self.device,
                )
                if self.data_parallel_rank > 0:
                    last_optim_state = state.get("last_optimizer_state", None)

            # load model parameters
            try:
                # these code is for compatible with old adaptor emb code. TODO: remove these
                rm_keys = []
                for key in state['model'].keys():
                    if (
                        (key.startswith('encoder.adaptor.') or key.startswith('decoder.adaptor.'))
                        and key.endswith('.embed_tokens.weight')
                        and len(key.split('.')) == 5
                    ):
                        rm_keys.append(key)
                for key in rm_keys:
                    del state['model'][key]

                self.model.update_embedding(state)

                if use_ema_weights_to_init_param and "extra_state" in state and "ema" in state["extra_state"]:
                    logger.info(
                        "use_ema_weights_to_init_param = True, will use EMA weights in "
                        "the ckpt to init the model param..."
                    )
                    ema_state_dict = (
                        state["extra_state"]["ema_fp32_params"]
                        if "ema_fp32_params" in state["extra_state"]
                        else state["extra_state"]["ema"]
                    )
                    logger.info(self.model.load_state_dict(ema_state_dict, strict=True, model_cfg=self.cfg.model))
                else:
                    logger.info(
                        self.model.load_state_dict(state["model"], strict=not self.is_fsdp, model_cfg=self.cfg.model)
                    )
                # save memory for later steps
                if not (
                    self.cfg.ema.store_ema
                    and (
                        use_latest_weights_to_init_ema
                        or not ("extra_state" in state and "ema" in state["extra_state"])
                    )
                ):
                    del state["model"]

            except Exception as e:
                logger.info(str(e))
                raise Exception(
                    "Cannot load model parameters from checkpoint {}; "
                    "please ensure that the architectures match.".format(filename)
                )
            extra_state = state["extra_state"]
            self._optim_history = state["optimizer_history"]

        if last_optim_state is not None and not reset_optimizer:
            # rebuild optimizer after loading model, since params may have changed
            self._build_optimizer()

            # only reload optimizer and lr_scheduler if they match
            last_optim = self._optim_history[-1]
            assert last_optim["optimizer_name"] == self.optimizer.__class__.__name__, (
                "Optimizer does not match; please reset the optimizer "
                f"(--reset-optimizer). {last_optim['optimizer_name']} vs "
                f"{self.optimizer.__class__.__name__}"
            )

            if not reset_lr_scheduler:
                self.lr_scheduler.load_state_dict(last_optim["lr_scheduler_state"])

            if self.is_fsdp and not self.model.use_sharded_state:
                # if use_sharded_state, the last_optim_state is already sharded,
                # skip this
                last_optim_state = self.model.get_shard_from_optim_state_dict(last_optim_state)
            elif not load_on_all_ranks and is_distributed:
                last_optim_state = self.optimizer.broadcast_global_state_dict(last_optim_state)

            self.optimizer.load_state_dict(last_optim_state, optimizer_overrides)

            self.set_num_updates(last_optim["num_updates"])

        if extra_state is not None:
            if "previous_training_time" in extra_state:
                self._previous_training_time = extra_state["previous_training_time"]
                self._start_time = time.time()

            self.lr_step_update()

            if "metrics" in extra_state and not reset_meters:
                metrics.load_state_dict(extra_state["metrics"])

                # reset TimeMeters, since their start times don't make sense anymore
                for meter in metrics.get_meters("default"):
                    if isinstance(meter, meters.TimeMeter):
                        meter.reset()

            if self.cfg.ema.store_ema:
                if use_latest_weights_to_init_ema or "ema" not in extra_state:
                    if "ema" not in extra_state:
                        logger.warn(
                            "EMA not found in checkpoint. But store_ema is True. "
                            "EMA is re-initialized from checkpoint."
                        )
                    elif use_latest_weights_to_init_ema:
                        logger.info("use_latest_weights_to_init_ema = True. EMA is re-initialized from checkpoint.")
                    self.ema.restore(state["model"], build_fp32_params=self.cfg.ema.ema_fp32)
                    del state["model"]
                else:
                    logger.info("Loading EMA from checkpoint")
                    self.ema.restore(extra_state["ema"], build_fp32_params=False)

                    if self.cfg.ema.ema_fp32:
                        if "ema_fp32_params" in extra_state:
                            logger.info("Loading EMA fp32 params from checkpoint")
                            self.ema.build_fp32_params(extra_state["ema_fp32_params"])
                        else:
                            logger.info("Building EMA fp32 params from EMA model in checkpoint")
                            self.ema.build_fp32_params()

            logger.info("Loaded checkpoint {} ({} updates)".format(filename, self.get_num_updates()))

        else:
            logger.info("No existing checkpoint found {}".format(filename))

        return extra_state

    def init_train_iterator(self, itr_state):
        for task in self.tasks:
            task.init_data_iterator(
                split='train',
                group=self.data_parallel_process_group,
                itr_state=itr_state.get(task.name, None),
            )

    def begin_epoch(self, epoch):
        """Called at the beginning of each epoch."""
        logger.info("begin training epoch {}".format(epoch))
        self.lr_step_begin_epoch(epoch)

    def begin_valid_epoch(self, epoch):
        """Called at the beginning of each validation epoch."""

        # task specific setup per validation epoch
        for task in self.tasks:
            task.begin_valid_epoch(epoch, self.get_model())

    def reset_dummy_batch(self, batch):
        self._dummy_batch = batch

    @metrics.aggregate("train")
    def train_step(self, samples, raise_oom=False):
        """Do forward, backward and parameter update."""
        self._set_seed()
        self.model.train()
        self.zero_grad()

        metrics.log_start_time("train_wall", priority=800, round=1)

        task_logging_outputs = []
        task_sample_size = []
        for task_id, task in enumerate(self.tasks):
            metrics.log_custom(
                meters.StrMeter,
                '{}/epoch'.format(task.__class__.__name__),
                task.data_iterators['train'].epoch_str,
                priority=1 + task_id,
            )
            # If EMA is enabled through store_ema=True
            # and task.uses_ema is True, pass the EMA model as a keyword
            # argument to the task.
            extra_kwargs = {}
            if self.cfg.ema.store_ema and getattr(task, "uses_ema", False):
                extra_kwargs["ema_model"] = self.ema.get_model()

            # forward and backward pass
            logging_outputs, sample_size, ooms = [], 0, 0
            for i, sample in enumerate(samples[task_id]):  # delayed update loop
                sample, is_dummy_batch = self._prepare_sample(task, sample)

                def maybe_no_sync():
                    """
                    Whenever *samples* contains more than one mini-batch, we
                    want to accumulate gradients locally and only call
                    all-reduce in the last backwards pass.
                    """
                    if (
                        self.data_parallel_world_size > 1
                        and hasattr(self.model, "no_sync")
                        and i < len(samples[task_id]) - 1
                        # The no_sync context manager results in increased memory
                        # usage with FSDP, since full-size gradients will be
                        # accumulated on each GPU. It's typically a better tradeoff
                        # to do the extra communication with FSDP.
                        and not self.is_fsdp
                    ):
                        return self.model.no_sync()
                    else:
                        return contextlib.ExitStack()  # dummy contextmanager

                try:
                    with maybe_no_sync():
                        # forward and backward
                        loss, sample_size_i, logging_output = task.train_step(
                            sample=sample,
                            model=self.model,
                            optimizer=self.optimizer,
                            update_num=self.get_num_updates(),
                            ignore_grad=is_dummy_batch,
                            **extra_kwargs,
                        )
                        del loss

                    logging_outputs.append(logging_output)
                    sample_size += sample_size_i

                    # emptying the CUDA cache after the first step can
                    # reduce the chance of OOM
                    if self.cuda and self.get_num_updates() == 0:
                        torch.cuda.empty_cache()
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        self._log_oom(e)
                        if raise_oom:
                            raise e
                        logger.warning("attempting to recover from OOM in forward/backward pass")
                        ooms += 1
                        self.zero_grad()
                        if self.cuda:
                            torch.cuda.empty_cache()
                        if self.cfg.distributed_training.distributed_world_size == 1:
                            return None
                    else:
                        raise e

            if is_dummy_batch:
                if torch.is_tensor(sample_size):
                    sample_size.zero_()
                else:
                    sample_size *= 0.0

            if torch.is_tensor(sample_size):
                sample_size = sample_size.float()
            else:
                sample_size = float(sample_size)

            # gather logging outputs from all replicas
            if self._sync_stats():
                train_time = self._local_cumulative_training_time()
                logging_outputs, (sample_size, ooms, total_train_time,) = self._aggregate_logging_outputs(
                    task, logging_outputs, sample_size, ooms, train_time, ignore=is_dummy_batch
                )
                self._cumulative_training_time = total_train_time / self.data_parallel_world_size
            task_logging_outputs.append(logging_outputs)
            task_sample_size.append(sample_size)

        overflow = False
        try:
            with torch.autograd.profiler.record_function("reduce-grads"):
                # reduce gradients across workers
                self.optimizer.all_reduce_grads(self.model)

            with torch.autograd.profiler.record_function("multiply-grads"):
                # multiply gradients by (data_parallel_size / sample_size) since
                # DDP normalizes by the number of data parallel workers for
                # improved fp16 precision.
                # Thus we get (sum_of_gradients / sample_size) at the end.
                # In case of fp16, this step also undoes loss scaling.
                # (Debugging note: Some optimizers perform this scaling on the
                # fly, so inspecting model.parameters() or optimizer.params may
                # still show the original, unscaled gradients.)
                numer = (
                    self.data_parallel_world_size if not self.cfg.optimization.use_bmuf or self._sync_stats() else 1
                )
                self.optimizer.multiply_grads(numer / (sum(task_sample_size) or 1.0))
                # self.optimizer.multiply_grads(numer / (sample_size or 1.0))
                # Note: (sample_size or 1.0) handles the case of a zero gradient, in a
                # way that avoids CPU/device transfers in case sample_size is a GPU or
                # TPU object. The assumption is that the gradient itself is also 0.

            with torch.autograd.profiler.record_function("clip-grads"):
                # clip grads
                grad_norm = self.clip_grad_norm(self.cfg.optimization.clip_norm)

            # check that grad norms are consistent across workers
            if not self.cfg.optimization.use_bmuf and self.cfg.distributed_training.ddp_backend != "slow_mo":
                self._check_grad_norms(grad_norm)
            if not torch.isfinite(grad_norm).all():
                # in case of AMP, if gradients are Nan/Inf then
                # optimizer step is still required
                if self.cfg.common.amp:
                    overflow = True
                else:
                    # check local gradnorm single GPU case, trigger NanDetector
                    raise FloatingPointError("gradients are Nan/Inf")

            with torch.autograd.profiler.record_function("optimizer"):
                # take an optimization step
                self.tasks[0].optimizer_step(self.optimizer, model=self.model, update_num=self.get_num_updates())
                if self.cfg.common.amp and overflow:
                    if self._amp_retries == self.cfg.common.amp_batch_retries:
                        logger.info("AMP: skipping this batch.")
                        self._amp_retries = 0
                    else:
                        self._amp_retries += 1
                        # recursion to feed in same batch
                        return self.train_step(samples, raise_oom)

        except FloatingPointError:
            # re-run the forward and backward pass with hooks attached to print
            # out where it fails
            raise
            self.zero_grad()
            with NanDetector(self.get_model()):
                for _, sample in enumerate(samples):
                    sample, _ = self._prepare_sample(task, sample)
                    task.train_step(
                        sample,
                        self.model,
                        self.optimizer,
                        self.get_num_updates(),
                        ignore_grad=False,
                        **extra_kwargs,
                    )
            raise
        except OverflowError as e:
            overflow = True
            logger.info(f"NOTE: gradient overflow detected, ignoring gradient, {str(e)}")
            grad_norm = torch.tensor(0.0).cuda()
            self.zero_grad()
        except RuntimeError as e:
            if "out of memory" in str(e):
                self._log_oom(e)
                logger.error("OOM during optimization, irrecoverable")
            raise e

        # Some distributed wrappers (e.g., SlowMo) need access to the optimizer
        # after the step
        if hasattr(self.model, "perform_additional_optimizer_actions"):
            if hasattr(self.optimizer, "fp32_params"):
                self.model.perform_additional_optimizer_actions(self.optimizer.optimizer, self.optimizer.fp32_params)
            else:
                self.model.perform_additional_optimizer_actions(self.optimizer.optimizer)

        logging_output = None
        if not overflow or self.cfg.distributed_training.ddp_backend == "slow_mo":
            self.set_num_updates(self.get_num_updates() + 1)

            if self.cfg.ema.store_ema:
                # Step EMA forward with new model.
                self.ema.step(
                    self.get_model(),
                    self.get_num_updates(),
                )
                metrics.log_scalar(
                    "ema_decay",
                    self.ema.get_decay(),
                    priority=10000,
                    round=5,
                    weight=0,
                )

            if self.cuda and self.cuda_env is not None:
                # log minimum free memory over the iteration
                gb_used = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
                torch.cuda.reset_peak_memory_stats()
                gb_free = self.cuda_env.total_memory_in_GB - gb_used
                metrics.log_scalar("gb_free", gb_free, priority=1500, round=1, weight=0)

            # log stats
            logging_output = self._reduce_and_log_stats(self.tasks, task_logging_outputs, task_sample_size, grad_norm)

            # clear CUDA cache to reduce memory fragmentation
            if (
                self.cuda
                and self.cfg.common.empty_cache_freq > 0
                and (
                    (self.get_num_updates() + self.cfg.common.empty_cache_freq - 1) % self.cfg.common.empty_cache_freq
                )
                == 0
            ):
                torch.cuda.empty_cache()

        if self.cfg.common.fp16 or self.cfg.common.amp:
            metrics.log_scalar(
                "loss_scale",
                (self.optimizer.scaler.loss_scale if self.cfg.common.fp16 else self.optimizer.scaler.get_scale()),
                priority=700,
                round=4,
                weight=0,
            )

        metrics.log_stop_time("train_wall")
        return logging_output

    @metrics.aggregate("valid")
    def valid_step(self, task, sample, raise_oom=False):
        """Do forward pass in evaluation mode."""
        # If EMA is enabled through store_ema=True
        # and task.uses_ema is True, pass the EMA model as a keyword
        # argument to the task.
        extra_kwargs = {}
        if self.cfg.ema.store_ema and getattr(task, "uses_ema", False):
            extra_kwargs["ema_model"] = self.ema.get_model()

        with torch.no_grad():
            self.model.eval()

            sample, is_dummy_batch = self._prepare_sample(task, sample, is_valid=True)

            try:
                _loss, sample_size, logging_output = task.valid_step(sample, self.model, **extra_kwargs)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self._log_oom(e)
                    if not raise_oom:
                        logger.warning("ran out of memory in validation step, retrying batch")
                        for p in self.model.parameters():
                            if p.grad is not None:
                                p.grad = None  # free some memory
                        if self.cuda:
                            torch.cuda.empty_cache()
                        return self.valid_step(sample, raise_oom=True)
                raise e

            logging_outputs = [logging_output]
            if is_dummy_batch:
                if torch.is_tensor(sample_size):
                    sample_size.zero_()
                else:
                    sample_size *= 0.0

        # gather logging outputs from all replicas
        if self.data_parallel_world_size > 1:
            logging_outputs, (sample_size,) = self._aggregate_logging_outputs(
                task,
                logging_outputs,
                sample_size,
                ignore=is_dummy_batch,
            )

        # log validation stats
        logging_output = self._reduce_and_log_stats([task], [logging_outputs], [sample_size])

        return logging_output

    def zero_grad(self):
        self.optimizer.zero_grad()

    def total_num_updates(self):
        assert len(self.tasks) > 0
        if len(self.tasks) > 1:
            assert self.cfg.optimization.max_update != 0
            return self.cfg.optimization.max_update
        else:
            epoch_itr = self.tasks[0].data_iterators['train']
            if self.max_epoch > 0 and self.max_epoch != math.inf:
                max_update = epoch_itr.total_num_updates(self.max_epoch)
            else:
                max_update = math.inf
            if self.cfg.optimization.max_update > 0:
                max_update = min(max_update, self.cfg.optimization.max_update)
            return max_update

    def lr_reinitialize(self):
        self.lr_reinit(self.total_num_updates(), self.get_num_updates())

    def lr_step_begin_epoch(self, epoch):
        """Adjust the learning rate at the beginning of the epoch."""
        self.lr_scheduler.step_begin_epoch(epoch)
        # prefer updating the LR based on the number of steps
        return self.lr_step_update()

    def lr_reinit(self, total_updates, num_updates):
        self.lr_scheduler.reinit(total_updates, num_updates)

    def lr_step(self, epoch, val_loss=None):
        """Adjust the learning rate at the end of the epoch."""
        self.lr_scheduler.step(epoch, val_loss)
        # prefer updating the LR based on the number of steps
        return self.lr_step_update()

    def lr_step_update(self):
        """Update the learning rate after each update."""
        new_lr = self.lr_scheduler.step_update(self.get_num_updates())
        if isinstance(new_lr, dict):
            for k, v in new_lr.items():
                metrics.log_scalar(f"lr_{k}", v, weight=0, priority=300)
            new_lr = new_lr.get("default", next(iter(new_lr.values())))
        else:
            metrics.log_scalar("lr", new_lr, weight=0, priority=300)
        return new_lr

    def get_lr(self):
        """Get the current learning rate."""
        return self.optimizer.get_lr()

    def get_model(self):
        """Get the (non-wrapped) model instance."""
        return self._model

    def get_num_updates(self):
        """Get the number of parameters updates."""
        return self._num_updates

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        self._num_updates = num_updates
        self.lr_step_update()
        metrics.log_scalar("num_updates", self._num_updates, weight=0, priority=200)

    def clip_grad_norm(self, clip_norm):
        def agg_norm_fn(total_norm):
            total_norm = total_norm.cuda().float() ** 2
            total_norm = distributed_utils.all_reduce(total_norm, group=self.data_parallel_process_group)
            return total_norm**0.5

        should_agg_norm = self.is_fsdp and (
            self.data_parallel_process_group is not None or torch.distributed.is_initialized()
        )
        return self.optimizer.clip_grad_norm(clip_norm, aggregate_norm_fn=agg_norm_fn if should_agg_norm else None)

    def cumulative_training_time(self):
        if self._cumulative_training_time is None:
            # single GPU
            return self._local_cumulative_training_time()
        else:
            return self._cumulative_training_time

    def _local_cumulative_training_time(self):
        """Aggregate training time in seconds."""
        return time.time() - self._start_time + self._previous_training_time

    def _fp_convert_sample(self, sample):
        def apply_half(t):
            if t.dtype is torch.float32:
                return t.to(dtype=torch.half)
            return t

        def apply_bfloat16(t):
            if t.dtype is torch.float32:
                return t.to(dtype=torch.bfloat16)
            return t

        if self.cfg.common.fp16:
            sample = utils.apply_to_sample(apply_half, sample)

        if self.cfg.common.bf16:
            sample = utils.apply_to_sample(apply_bfloat16, sample)

        return sample

    def _prepare_sample(self, task, sample, is_dummy=False, is_valid=False):
        if sample == "DUMMY":
            raise Exception(
                "Trying to use an uninitialized 'dummy' batch. This usually indicates "
                "that the total number of batches is smaller than the number of "
                "participating GPUs. Try reducing the batch size or using fewer GPUs."
            )

        if sample is None or len(sample) == 0:
            task_dummy_batch = self._dummy_batch[task.name]
            assert task_dummy_batch is not None and len(task_dummy_batch) > 0, "Invalid dummy batch: {}".format(
                task_dummy_batch
            )
            sample, _ = self._prepare_sample(task, task_dummy_batch, is_dummy=True, is_valid=is_valid)
            return sample, True

        # Given that PCIe/NVLink bandwidth is significantly smaller than DRAM bandwidth
        # it makes sense to do the format conversion on the CPU and then transfer
        # a smaller buffer to the device. This also saves GPU memory capacity.

        if self.cfg.common.on_cpu_convert_precision:
            sample = self._fp_convert_sample(sample)

        if self.cuda:
            if self.pipeline_model_parallel:
                if 'target' in sample:
                    sample['target'] = utils.move_to_cuda(sample['target'], device=self.last_device)
            else:
                sample = utils.move_to_cuda(sample)

        if not self.cfg.common.on_cpu_convert_precision:
            sample = self._fp_convert_sample(sample)

        # TODO: do not deepcopy sample everytime
        self._dummy_batch[task.name] = copy.deepcopy(sample)

        return sample, False

    def _set_seed(self):
        # Set seed based on args.seed and the update number so that we get
        # reproducible results when resuming from checkpoints
        seed = self.cfg.common.seed + self.get_num_updates()
        utils.set_torch_seed(seed)

    def _sync_stats(self):
        # Return True if it's using multiple GPUs and DDP or multiple GPUs with
        # BMUF and it's a bmuf sync with warmup iterations completed before.
        if self.data_parallel_world_size == 1:
            return False
        elif self.cfg.optimization.use_bmuf:
            return (self.get_num_updates() + 1) % self.cfg.bmuf.global_sync_iter == 0 and (
                self.get_num_updates() + 1
            ) > self.cfg.bmuf.warmup_iterations
        else:
            return True

    def _log_oom(self, exc):
        msg = "OOM: Ran out of memory with exception: {}".format(exc)
        logger.warning(msg)
        if torch.cuda.is_available() and hasattr(torch.cuda, "memory_summary"):
            for device_idx in range(torch.cuda.device_count()):
                logger.warning(torch.cuda.memory_summary(device=device_idx))
        sys.stderr.flush()

    def _aggregate_logging_outputs(
        self,
        task,
        logging_outputs: List[Dict[str, Any]],
        *extra_stats_to_sum,
        ignore=False,
    ):
        if task.criterion.logging_outputs_can_be_summed():
            return self._fast_stat_sync_sum(logging_outputs, *extra_stats_to_sum, ignore=ignore)
        else:
            return self._all_gather_list_sync(logging_outputs, *extra_stats_to_sum, ignore=ignore)

    def _all_gather_list_sync(
        self,
        logging_outputs: List[Dict[str, Any]],
        *extra_stats_to_sum,
        ignore=False,
    ):
        """
        Sync logging outputs across workers. all_gather_list_sync is
        suitable when logging outputs are complex types.
        """
        if ignore:
            logging_outputs = []
        results = list(
            zip(
                *distributed_utils.all_gather_list(
                    [logging_outputs] + list(extra_stats_to_sum),
                    max_size=getattr(self.cfg.common, "all_gather_list_size", 16384),
                    group=self.data_parallel_process_group,
                )
            )
        )
        logging_outputs, extra_stats_to_sum = results[0], results[1:]
        logging_outputs = list(chain.from_iterable(logging_outputs))
        extra_stats_to_sum = [sum(s) for s in extra_stats_to_sum]
        return logging_outputs, extra_stats_to_sum

    def _fast_stat_sync_sum(
        self,
        logging_outputs: List[Dict[str, Any]],
        *extra_stats_to_sum,
        ignore=False,
    ):
        """
        Sync logging outputs across workers. fast_stat_sync_sum is
        faster than all_gather_list_sync, but is only suitable when
        logging outputs are scalars and can be summed. Note that
        *logging_outputs* cannot contain any nested dicts/lists.
        """
        data = {}
        for i, stat in enumerate(extra_stats_to_sum):
            data["extra_stats_" + str(i)] = stat
        if len(logging_outputs) > 0:
            log_keys = list(logging_outputs[0].keys())
            for k in log_keys:
                if not ignore:
                    v = sum(log[k] for log in logging_outputs if k in log)
                else:
                    v = logging_outputs[0][k]
                    v = torch.zeros_like(v) if torch.is_tensor(v) else 0
                data["logging_outputs_" + k] = v
        else:
            log_keys = None

        data = distributed_utils.all_reduce_dict(data, device=self.device, group=self.data_parallel_process_group)

        extra_stats_to_sum = [data["extra_stats_" + str(i)] for i in range(len(extra_stats_to_sum))]
        if log_keys is not None:
            logging_outputs = [{k: data["logging_outputs_" + k] for k in log_keys}]
        else:
            logging_outputs = []
        return logging_outputs, extra_stats_to_sum

    def _check_grad_norms(self, grad_norm):
        """Check that grad norms are consistent across workers."""
        if self._grad_norm_buf is not None:
            self._grad_norm_buf.zero_()
            self._grad_norm_buf[self.data_parallel_rank] = grad_norm
            distributed_utils.all_reduce(self._grad_norm_buf, group=self.data_parallel_process_group)

            def is_consistent(tensor):
                max_abs_diff = torch.max(torch.abs(tensor - tensor[0]))
                return (
                    (torch.isfinite(tensor).all() and (max_abs_diff / (tensor[0] + 1e-6) < 1e-6).all())
                    or (self.cfg.common.amp and not torch.isfinite(tensor).all())
                    # in case of amp non-finite grads are fine
                )

            if not is_consistent(self._grad_norm_buf):
                pretty_detail = "\n".join(
                    "rank {:3d} = {:.8f}".format(r, n) for r, n in enumerate(self._grad_norm_buf.tolist())
                )
                error_detail = "grad_norm across the workers:\n{}\n".format(pretty_detail)
                # use FloatingPointError to trigger NanDetector
                raise FloatingPointError(
                    "Fatal error: gradients are inconsistent between workers. "
                    "Try --ddp-backend=legacy_ddp. "
                    "Or are you mixing up different generation of GPUs in training?"
                    + "\n"
                    + "-" * 80
                    + "\n{}\n".format(error_detail)
                    + "-" * 80
                )

    def _reduce_and_log_stats(self, tasks, task_logging_outputs, task_sample_size, grad_norm=None):
        tasks, task_logging_outputs, task_sample_size = list(tasks), list(task_logging_outputs), list(task_sample_size)

        if grad_norm is not None and (not torch.is_tensor(grad_norm) or torch.isfinite(grad_norm)):
            metrics.log_speed("ups", 1.0, priority=100, round=2)
            metrics.log_scalar("gnorm", grad_norm, priority=400, round=3)
            if self.cfg.optimization.clip_norm > 0:
                metrics.log_scalar(
                    "clip",
                    torch.where(
                        grad_norm > self.cfg.optimization.clip_norm,
                        grad_norm.new_tensor(100),
                        grad_norm.new_tensor(0),
                    ),
                    priority=500,
                    round=1,
                )

        with metrics.aggregate() as agg:
            for task, logging_outputs, sample_size in zip(tasks, task_logging_outputs, task_sample_size):
                if logging_outputs is not None:
                    task.reduce_metrics(logging_outputs, task.criterion)
                    del logging_outputs

                # support legacy interface
                logging_output = agg.get_smoothed_values()
                logging_output["sample_size"] = sample_size
                for key_to_delete in ["ppl", "wps", "wpb", "bsz"]:
                    if key_to_delete in logging_output:
                        del logging_output[key_to_delete]
            return logging_output


def _catalog_shared_params(module, memo=None, prefix=""):
    if memo is None:
        first_call = True
        memo = {}
    else:
        first_call = False
    for name, param in module._parameters.items():
        if param is None:
            continue
        param_prefix = prefix + ("." if prefix else "") + name
        if param not in memo:
            memo[param] = []
        memo[param].append(param_prefix)
        logger.debug(
            'name: %s | shape: %s | %s'
            % (param_prefix, str(param.shape), 'requires_grad' if param.requires_grad else 'frozen')
        )

    for name, m in module._modules.items():
        if m is None:
            continue
        submodule_prefix = prefix + ("." if prefix else "") + name
        _catalog_shared_params(m, memo, submodule_prefix)
    if first_call:
        return [x for x in memo.values() if len(x) > 1]


def _get_module_by_path(module, path):
    path = path.split(".")
    for name in path:
        module = getattr(module, name)
    return module


def _set_module_by_path(module, path, value):
    path = path.split(".")
    for name in path[:-1]:
        module = getattr(module, name)
    setattr(module, path[-1], value)


def should_stop_early(cfg: DictConfig, valid_loss: float) -> bool:
    # skip check if no validation was done in the current epoch
    if valid_loss is None:
        return False
    if cfg.checkpoint.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if cfg.checkpoint.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, "best", None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= cfg.checkpoint.patience:
            logger.info(
                "early stop since valid performance hasn't improved for last {} runs".format(cfg.checkpoint.patience)
            )
            return True
        else:
            return False


def train(cfg, trainer, extra_state):
    if extra_state is None or cfg.checkpoint.reset_dataloader:
        start_step = 0
    else:
        start_step = extra_state.get("global_step", 0)

    progress = progress_bar.progress_bar(
        range(trainer.total_num_updates()),
        start=start_step,
        log_format=cfg.common.log_format,
        log_file=cfg.common.log_file,
        log_interval=cfg.common.log_interval,
        tensorboard_logdir=(
            cfg.common.tensorboard_logdir if distributed_utils.is_master(cfg.distributed_training) else None
        ),
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
        wandb_project=(cfg.common.wandb_project if distributed_utils.is_master(cfg.distributed_training) else None),
        wandb_run_name=os.environ.get("WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)),
        azureml_logging=(
            cfg.common.azureml_logging if distributed_utils.is_master(cfg.distributed_training) else False
        ),
    )
    progress.update_config(_flatten_config(cfg))

    def get_samples():
        samples = []
        for task in trainer.tasks:
            samples.append(task.get_sample('train'))
        return samples

    def train(i, samples):
        with metrics.aggregate("train_inner"), torch.autograd.profiler.record_function("train_step-%d" % i):
            with metrics.aggregate("train_inner"):
                log_output = trainer.train_step(samples)

            if log_output is not None:  # not OOM, overflow, ...
                # log mid-epoch stats
                num_updates = trainer.get_num_updates()
                if num_updates % cfg.common.log_interval == 0:
                    stats = get_training_stats(metrics.get_smoothed_values("train_inner"))
                    progress.log(stats, tag="train_inner", step=num_updates)

                    # reset mid-epoch stats after each log interval
                    # the end-of-epoch stats will still be preserved
                    metrics.reset_meters("train_inner")

        valid_losses, should_stop = validate_and_save(cfg, trainer, step=i)
        return valid_losses, should_stop

    for i in progress:
        samples = get_samples()
        if (
            distributed_utils.get_global_rank() == 0
            and False  # cfg.common.profile # TODO: add a config to control whether to export profile file
            and i == 5
        ):
            logger.info("STARTING PROFILER")
            with profiler.profile(profile_memory=True, with_stack=True, record_shapes=True) as prof:
                valid_losses, should_stop = train(i, samples)
            torch.cuda.synchronize()
            with open(os.path.join(cfg.checkpoint.save_dir, "memory_usage.txt"), 'w') as sourceFile:
                print(
                    prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_memory_usage", row_limit=10),
                    file=sourceFile,
                )
            prof.export_chrome_trace(os.path.join(cfg.checkpoint.save_dir, "profiler_trace.json"))
        else:
            valid_losses, should_stop = train(i, samples)
        if should_stop:
            break

        for task in trainer.tasks:
            if task.data_iterators['train'].end_of_epoch:
                task.end_epoch()
                task.data_iterators['train'].next_epoch()
                task.begin_epoch()

    return valid_losses, should_stop


def _flatten_config(cfg: DictConfig):
    config = OmegaConf.to_container(cfg)
    # remove any legacy Namespaces and replace with a single "args"
    namespace = None
    for k, v in list(config.items()):
        if isinstance(v, Namespace):
            namespace = v
            del config[k]
    if namespace is not None:
        config["args"] = vars(namespace)
    return config


def _print_conf(omgeaconf, width=80):
    conf = _flatten_config(omgeaconf)

    def _conf_to_text(conf, level=0):
        lines = []
        prefix = "║" + "  " * level
        for key in sorted(conf.keys()):
            value = conf[key]
            if key == "_name" and value is None:
                continue
            if isinstance(value, dict):
                if key in {
                    "task",
                    "scoring",
                    "generation",
                    "eval_lm",
                    "dataset",
                    "common_eval",
                }:
                    continue
                lines.append(prefix + f"{key}:")
                lines.extend(_conf_to_text(value, level + 1))
            else:
                if key in {"tokenizer", "bpe", "model", "criterion"}:
                    continue
                lines.append(prefix + f"{key}: {value}")
        return lines

    lines = [f'╔{" fairseq Configuration ":═^{width-1}}']
    lines.extend(_conf_to_text(conf))
    lines.append("╚" + "═" * (width - 1))

    logger.info(f"fairseq managed configuration:\n" + "\n".join(lines))


def validate_and_save(
    cfg: DictConfig,
    trainer: Trainer,
    step: int = 0,
) -> Tuple[List[Optional[float]], bool]:
    num_updates = trainer.get_num_updates()
    max_update = cfg.optimization.max_update or math.inf
    tasks = trainer.tasks
    n_tasks = len(tasks)

    # Stopping conditions (and an additional one based on validation loss later on)
    should_stop = False
    if num_updates >= max_update:
        should_stop = True
        logger.info(f"Stopping training due to " f"num_updates: {num_updates} >= max_update: {max_update}")

    training_time_hours = trainer.cumulative_training_time() / (60 * 60)
    if cfg.optimization.stop_time_hours > 0 and training_time_hours > cfg.optimization.stop_time_hours:
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"cumulative_training_time: {training_time_hours} > "
            f"stop_time_hours: {cfg.optimization.stop_time_hours} hour(s)"
        )

    do_save = False
    do_validate = [False] * n_tasks

    for i, task in enumerate(tasks):
        epoch_itr = task.data_iterators['train']
        do_save = (
            do_save
            or (
                epoch_itr.end_of_epoch
                and cfg.checkpoint.save_interval > 0
                and epoch_itr.epoch % cfg.checkpoint.save_interval == 0
            )
            or should_stop
            or (
                cfg.checkpoint.save_interval_updates > 0
                and num_updates > 0
                and num_updates % cfg.checkpoint.save_interval_updates == 0
                and num_updates >= cfg.dataset.validate_after_updates
            )
        )
        do_validate[i] = (
            (
                (not epoch_itr.end_of_epoch and do_save)  # validate during mid-epoch saves
                or (
                    epoch_itr.end_of_epoch
                    and cfg.dataset.validate_interval > 0
                    and epoch_itr.epoch % cfg.dataset.validate_interval == 0
                )
                or should_stop
                or (
                    cfg.dataset.validate_interval_updates > 0
                    and num_updates > 0
                    and num_updates % cfg.dataset.validate_interval_updates == 0
                )
            )
            and not cfg.dataset.disable_validation
            and num_updates >= cfg.dataset.validate_after_updates
            and task.cfg.dataset.valid_data
        )

    valid_losses = [None] * n_tasks
    mean_scores, valid_tasks = 0.0, 0

    for i, task in enumerate(tasks):
        if do_validate[i]:
            valid_losses[i] = validate(cfg, trainer, task, task.data_iterators['train'])
            mean_scores += valid_losses[i][0] if valid_losses[i][0] is not None else 0
            valid_tasks += 1

    mean_scores = mean_scores / valid_tasks if valid_tasks > 0 else None

    should_stop |= should_stop_early(cfg, mean_scores)

    if do_save or should_stop:
        epoch_itrs = {}
        for task in tasks:
            epoch_itrs[task.name] = task.data_iterators['train']
        checkpoint_utils.save_checkpoint(cfg.checkpoint, trainer, epoch_itrs, mean_scores, step=step)
        gc.collect()

    return valid_losses, should_stop


def get_training_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    stats["wall"] = round(metrics.get_meter("default", "wall").elapsed_time, 0)
    return stats


def validate(
    cfg: DictConfig,
    trainer: Trainer,
    task: OFATask,
    epoch_itr,
) -> List[Optional[float]]:
    """Evaluate the model on the validation set(s) and return the losses."""

    if task.cfg.dataset.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(task.cfg.dataset.fixed_validation_seed)

    trainer.begin_valid_epoch(epoch_itr.epoch)
    valid_losses = []
    for subset in ['valid']:
        logger.info('begin validation on "{}" {}'.format(subset, task.__class__.__name__))

        # Initialize data iterator
        if subset not in task.data_iterators:
            task.init_data_iterator(subset, group=trainer.data_parallel_process_group)
        else:
            task.data_iterators[subset].next_epoch()
        itr = task.data_iterators[subset].cur_epoch_itr
        progress = progress_bar.progress_bar(
            itr,
            log_format=cfg.common.log_format,
            log_interval=cfg.common.log_interval,
            epoch=epoch_itr.epoch,
            prefix=f"valid on '{subset}' subset",
            tensorboard_logdir=(
                cfg.common.tensorboard_logdir if distributed_utils.is_master(cfg.distributed_training) else None
            ),
            default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
            wandb_project=(
                cfg.common.wandb_project if distributed_utils.is_master(cfg.distributed_training) else None
            ),
            wandb_run_name=os.environ.get("WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)),
        )

        # create a new root metrics aggregator so validation metrics
        # don't pollute other aggregators (e.g., train meters)
        with metrics.aggregate(new_root=True) as agg:
            for i, sample in enumerate(progress):
                if cfg.dataset.max_valid_steps is not None and i > cfg.dataset.max_valid_steps:
                    break
                trainer.valid_step(task, sample)

        # log validation stats
        if hasattr(task, 'get_valid_stats'):
            stats = task.get_valid_stats(cfg, trainer, agg.get_smoothed_values())
        else:
            stats = agg.get_smoothed_values()
        stats = get_valid_stats(cfg, trainer, stats)

        if hasattr(task, "post_validate"):
            task.post_validate(trainer.get_model(), stats, agg)

        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        if cfg.checkpoint.best_checkpoint_metric in stats:
            valid_losses.append(stats[cfg.checkpoint.best_checkpoint_metric])
        else:
            valid_losses.append(None)
    return valid_losses


def get_valid_stats(cfg: DictConfig, trainer: Trainer, stats: Dict[str, Any]) -> Dict[str, Any]:
    stats["num_updates"] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, "best"):
        key = "best_{0}".format(cfg.checkpoint.best_checkpoint_metric)
        best_function = max if cfg.checkpoint.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            stats[cfg.checkpoint.best_checkpoint_metric],
        )
    return stats
