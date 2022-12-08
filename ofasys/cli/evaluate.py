# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

# flake8: noqa: E402
import json
import logging
import os
import sys
from itertools import chain

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig

import ofasys
from ofasys.configure import ConfigStore, options
from ofasys.configure.utils import convert_namespace_to_omegaconf
from ofasys.distributed import utils as distributed_utils
from ofasys.logging import progress_bar
from ofasys.module import utils
from ofasys.module.utils import reset_logging
from ofasys.utils import checkpoint_utils

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("ofasys.evaluate")


def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t


def merge_results(task, cfg, logger, score_cnt, score_sum, results):
    if task.cfg._name in ('image_gen', 'text_to_speech'):
        if cfg.distributed_training.distributed_world_size > 1:
            dist.all_reduce(score_sum.data)
            dist.all_reduce(score_cnt.data)
        if score_cnt.item() > 0:
            logger.info(
                "score_sum: {}, score_cnt: {}, score: {}".format(
                    score_sum, score_cnt, round(score_sum.item() / score_cnt.item(), 4)
                )
            )
    else:
        gather_results = None
        if cfg.distributed_training.distributed_world_size > 1:
            gather_results = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gather_results, results)
            dist.all_reduce(score_sum.data)
            dist.all_reduce(score_cnt.data)
        if score_cnt.item() > 0:
            logger.info(
                "score_sum: {}, score_cnt: {}, score: {}".format(
                    score_sum, score_cnt, round(score_sum.item() / score_cnt.item(), 4)
                )
            )

        if cfg.distributed_training.distributed_world_size == 1 or dist.get_rank() == 0:
            os.makedirs(cfg.checkpoint.save_dir, exist_ok=True)
            output_path = os.path.join(cfg.checkpoint.save_dir, "test_predict.json")
            gather_results = list(chain(*gather_results)) if gather_results is not None else results
            with open(output_path, 'w') as fw:
                json.dump(gather_results, fw, indent=2)


def main(cfg: DictConfig, **kwargs):
    reset_logging()
    logger.info(cfg)

    assert cfg.dataset.batch_size is None  # use ofasys batch_size insteal of fairseq

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_fp16 = cfg.common.fp16
    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    if use_cuda:
        torch.cuda.set_device(cfg.distributed_training.device_id)

    # Load ensemble
    overrides = eval(cfg.common_eval.model_overrides)
    logger.info("loading model(s) from {}".format(cfg.checkpoint.restore_file))

    tasks = [t() for t in ConfigStore().build("ofasys.task")]
    assert len(tasks) == 1
    task = tasks[0]
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        utils.split_paths(cfg.checkpoint.restore_file),
        task,
        arg_overrides=overrides,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
    )
    assert len(models) == 1

    # Move models to GPU
    for model, ckpt_path in zip(models, utils.split_paths(cfg.checkpoint.restore_file)):
        if kwargs['ema_eval']:
            logger.info("loading EMA weights from {}".format(ckpt_path))
            model.load_state_dict(checkpoint_utils.load_ema_from_checkpoint(ckpt_path)['model'])
        model.eval()
        if use_fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        split='test',
        group=distributed_utils.get_data_parallel_group(),
    ).cur_epoch_itr
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )

    logger.info("start inference")
    results = []
    score_sum = torch.FloatTensor([0])
    score_cnt = torch.FloatTensor([0])
    if use_cuda:
        score_sum, score_cnt = score_sum.cuda(), score_cnt.cuda()
    for sample in progress:
        if "net_input" not in sample:
            continue
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        sample = utils.apply_to_sample(apply_half, sample) if cfg.common.fp16 else sample
        with torch.no_grad():
            hyps = task.inference(models[0], sample)
            # result, scores = eval_step(task, generator, models, sample, **kwargs)
        # result = []
        # for sample_id, hypo in zip(sample['id'], hyps):
        #     results.append({"sample_id": sample_id, "hypo": hypo})
        # results += result
        # score_sum += sum(scores) if scores is not None else 0
        # score_cnt += len(scores) if scores is not None else 0
        progress.log({"sentences": sample["nsentences"]})

    # merge_results(task, cfg, logger, score_cnt, score_sum, results)


def cli_main():
    parser = options.get_generation_parser()
    parser.add_argument("--ema-eval", action='store_true', help="Use EMA weights to make evaluation.")
    parser.add_argument(
        "--beam-search-vqa-eval",
        action='store_true',
        help=(
            "Use beam search for vqa evaluation (faster inference speed but sub-optimal "
            "result), if not specified, we compute scores for each answer in the "
            "candidate set, which is slower but can obtain best result."
        ),
    )
    args = options.parse_args_and_arch(parser)
    cfg = convert_namespace_to_omegaconf(args)
    distributed_utils.call_main(cfg, main, ema_eval=args.ema_eval, beam_search_vqa_eval=args.beam_search_vqa_eval)


if __name__ == "__main__":
    cli_main()
