# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import functools
import json
import logging
import random
import warnings
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import torch

from ofasys import ModalityType
from ofasys.adaptor import OFAAdaptorConfig, default_adaptor
from ofasys.configure import BaseDataclass, ChoiceEnum, ConfigStore, register_config
from ofasys.engine.optim.amp_optimizer import AMPOptimizer
from ofasys.generator import (
    AutoRegressiveSpeechGenerator,
    BatchGeneratorOutput,
    DiffusionGenerator,
    MotionOutput,
    MultiGeneratorOutput,
    SequenceGenerator,
    SequenceGeneratorOutput,
    SpeechGeneratorOutput,
)
from ofasys.io.reader import EpochBatchIterator
from ofasys.io.reader.utils import parse_template
from ofasys.logging import metrics
from ofasys.metric import BaseMetric
from ofasys.module import utils
from ofasys.preprocessor import (
    GeneralPreprocess,
    Instruction,
    PreprocessConfig,
    Slot,
    default_preprocess,
)
from ofasys.utils import search

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig(BaseDataclass):
    train_data: str = field(
        default="",
        metadata={
            "help": "comma separated path to data list, will be iterated upon during" "epochs in round-robin manner"
        },
    )
    valid_data: str = field(
        default="",
        metadata={"help": "the valid dataset path"},
    )
    test_data: str = field(
        default="",
        metadata={"help": "the valid dataset path"},
    )
    selected_cols: str = field(
        default="",
        metadata={"help": "selected cols"},
    )
    use_hf_datasets: bool = field(default=False, metadata={"help": "whether to use huggingface datasets"})
    sample_ratios: Any = field(
        default=1,
        metadata={"help": "the sample ratio between each dataset."},
    )
    update_freq: Union[int, List[int]] = field(
        default_factory=lambda: [1],
        metadata={"help": "update parameters every N_i batches, when in epoch i"},
    )
    micro_batch_size: int = field(
        default=32,
        metadata={"help": "number of examples in a batch"},
    )
    micro_valid_batch_size: Optional[int] = field(
        default=None,
        metadata={"help": "number of examples in a valid batch"},
    )
    fixed_validation_seed: Optional[int] = field(default=7, metadata={"help": "specified random seed for validation"})
    num_workers: int = field(default=2, metadata={"help": "how many subprocesses to use for data loading"})
    prefetch_factor: int = field(default=5, metadata={"help": "Number of batches to preload"})
    common_io_capacity: int = field(
        default=1024,
        metadata={"help": "common-io capacity"},
    )
    common_io_num_threads: int = field(
        default=2,
        metadata={"help": "common-io number of threads"},
    )
    seperator: str = field(
        default='\t',
        metadata={"help": "tsv seperator"},
    )
    oss_buffer_capacity: int = field(default=64, metadata={"help": "oss reader initial buffer capacity, unit: Kb"})
    header: bool = field(default=False, metadata={"help": "whether tsv file has headers of column name"})
    cached: bool = field(
        default=False,
        metadata={"help": "whether uses cached reader"},
    )
    shuffle: bool = field(
        default=True,
        metadata={
            "help": "Whether to shuffle the training dataset at the beginning of "
            "an epoch, only support cached=True for now."
        },
    )
    text_bin_length: int = field(
        default=1024,
        metadata={"help": "Length of text in TextBinReader"},
    )
    interleaved_multiple_reader: bool = field(
        default=False,
        metadata={"help": "Use interleaved arrangement instead of concatenation when mixing multiple readers"},
    )


@dataclass
class InstructionConfig(BaseDataclass):
    template: Optional[str] = field(default=None, metadata={"help": "template"})
    mode: ChoiceEnum(['auto', 'manual']) = field(
        default='auto', metadata={"help": "instruction mode, not finished implementation"}
    )
    decoder_plain_with_loss: bool = field(
        default=False, metadata={"help": "whether plain text has loss in decoder's instruction"}
    )


MetricConfigs = ConfigStore().make_dataclass("ofasys.metric", "MetricConfigs", __name__)


CriterionConfigs = ConfigStore().make_dataclass("ofasys.criterion", "CriterionConfigs", __name__)


@dataclass
class EvaluationConfig(BaseDataclass):
    metrics: MetricConfigs = field(
        default_factory=MetricConfigs,
        metadata={"help": "A list of metric"},
    )
    generator_args: str = field(
        default='{"beam":5,"max_len_b":32,"no_repeat_ngram_size":3}',
        metadata={
            "help": 'generation args for BLUE or CIDEr scoring, e.g., '
            '\'{"beam": 4, "lenpen": 0.6}\', as JSON string'
        },
    )
    eval_print_samples: bool = field(default=False, metadata={"help": "print sample generations during validation"})
    output_dir: str = field(default='', metadata={"help": "path to save inference results"})


@dataclass
class TaskConfig(BaseDataclass):
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    instruction: InstructionConfig = field(default_factory=InstructionConfig)
    criterion: CriterionConfigs = field(default_factory=CriterionConfigs)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    max_source_positions: int = field(default=1024, metadata={"help": "max number of tokens in the source sequence"})
    max_target_positions: int = field(default=1024, metadata={"help": "max number of tokens in the target sequence"})
    max_src_length: int = field(default=128, metadata={"help": "the maximum src sequence length"})
    max_tgt_length: int = field(default=30, metadata={"help": "the maximum target sequence length"})
    max_object_length: int = field(default=30, metadata={"help": "the maximum object sequence length"})
    constraint_range: Optional[str] = field(default=None, metadata={"help": "constraint range"})
    scst: bool = field(default=False, metadata={"help": "Self-critical sequence training"})
    scst_args: str = field(
        default='{}',
        metadata={"help": 'generation args for Self-critical sequence training, as JSON string'},
    )

    diffuser_args: str = field(
        default='{"scheduler": "DDIMScheduler", "num_inference_steps": 50}',
        metadata={"help": "args for the diffuser scheduler, as JSON string"},
    )

    def update(self, **kwargs):
        if 'name' in kwargs:
            self._name = kwargs['name']
        if 'instruction' in kwargs:
            self.instruction.template = kwargs['instruction']
        if 'micro_batch_size' in kwargs:
            self.dataset.micro_batch_size = kwargs['micro_batch_size']


@register_config("ofasys.task", "default", dataclass=TaskConfig)
class OFATask:
    def __init__(self, cfg: TaskConfig = None, **kwargs):
        """
        A Task in OFA-Sys describes an execution logic specifying which parts of the model should be involved
        in dealing with certain input-output mapping. It contains a declarative multi-modal instruction
        and a logical plan that supplements model implementation details for a task for certain datasets.
        Task contains *Metrics*, *Preprocessor*, *Criterion* , and *data_iterators*.

        Args:
            cfg (TaskConfig): configuration for Task, including dataset config, preprocess config, instruction config,
            criterion config and evaluation config.

        """
        self.cfg = TaskConfig() if cfg is None else cfg
        self.cfg.update(**kwargs)
        self._generator = None
        self.diffuser_args = json.loads(cfg.diffuser_args)  # accessed by the diffusion criterion and generator

        self.datasets = {}
        self.data_iterators: Dict[str, EpochBatchIterator] = {}
        self.templates = parse_template(self.cfg.instruction.template)
        warning_for_bos_eos(self.templates)
        self.target_modality = self.infer_target_modality(self.templates[0]) if self.templates is not None else None
        self.target_preprocess = (
            self.infer_target_preprocess(self.templates[0]) if self.templates is not None else None
        )

    def initialize(self, global_dict, **kwargs):
        self.global_dict = global_dict
        if kwargs.get('is_train', True):
            update_preprocess_config_by_template(self.cfg.preprocess, self.templates, self.name)
        self.general_preprocess = self.build_preprocess(self.cfg.preprocess, global_dict)
        self.metrics = self.build_metrics(self.cfg.evaluation.metrics)
        if kwargs.get('is_train', True):
            self.criterion = self.build_criterion(self.cfg.criterion)

    @classmethod
    def upgrade_model_adaptor_cfg(cls, tasks, model_cfg):
        active_adaptors_name = collect_adaptor_name_from_tasks(tasks)
        update_adaptor_config_by_names(model_cfg.adaptor, active_adaptors_name)

    @property
    def generator(self):
        if self._generator is None:
            gen_args = json.loads(self.cfg.evaluation.generator_args)
            if self.target_modality == ModalityType.TEXT:
                # Constrained generation
                assert self.target_preprocess is not None
                gen_args["constraint_trie"] = self.general_preprocess.name2pre[self.target_preprocess].constraint_trie
                gen_args["constraint_range"] = self.cfg.constraint_range
            self._generator = self.build_generator(target_modality=self.target_modality, **gen_args)
        return self._generator

    @generator.setter
    def generator(self, generator):
        self._generator = generator

    @property
    def scst_generator(self):
        if self._scst_generator is None:
            scst_args = json.loads(self.cfg.scst_args)
            self._scst_generator = self.build_generator(target_modality=self.target_modality, **scst_args)
        return self._scst_generator

    @property
    def name(self):
        if self.cfg._name:
            return self.cfg._name
        else:
            return self.__class__.__name__

    def add_dataset(self, dataset, split="train"):
        assert self.datasets.get(split, None) is None, f"{split} dataset already exists in task {self.name}"
        self.datasets[split] = dataset

    def add_train_dataset(self, dataset):
        self.add_dataset(dataset, split="train")

    def add_valid_dataset(self, dataset):
        self.add_dataset(dataset, split="valid")

    def add_test_dataset(self, dataset):
        self.add_dataset(dataset, split="test")

    def infer_target_modality(self, instruction: Union[str, Instruction]):
        if not isinstance(instruction, Instruction):
            instruction = Instruction(instruction)
        target_slot: Slot = Slot.get_target_slot_from_slots(instruction.slots)
        return target_slot.modality

    def infer_target_preprocess(self, instruction: Union[str, Instruction]):
        if not isinstance(instruction, Instruction):
            instruction = Instruction(instruction)
        target_slot: Slot = Slot.get_target_slot_from_slots(instruction.slots)
        return (
            target_slot.get_attr('preprocess')
            if target_slot.get_attr('preprocess')
            else default_preprocess[target_slot.modality]
        )

    def preprocess(self, data: Dict[str, Any], split: str) -> Dict[str, Any]:
        """
        Preprocess raw input data for a certain dataset.

        Args:
            data (Dict): input data.
            split (str): data split: train, valid, or test.

        Returns:

        """
        return data

    def build_instruction(self, data: Dict[str, Any], split: str) -> Instruction:
        """
        Initialize an Instruction using a sampled template and format with input data.

        Args:
            data (Dict): input data.
            split (str): data split: train, valid, or test.

        Returns:
            formatted instruction.

        """

        def get_template():
            if len(self.templates) > 1:
                template = random.sample(self.templates, k=1)[0]
            else:
                template = self.templates[0]
            return template

        template = get_template()
        ist = Instruction(template, split=split, decoder_plain_with_loss=self.cfg.instruction.decoder_plain_with_loss)
        return ist.format(**data)

    def build_preprocess(self, cfg: PreprocessConfig, global_dict):
        """
        Build GeneralPreprocess.

        Args:
            cfg: config object for Preprocess.

        Returns:
            GeneralPreprocess object.
        """
        return GeneralPreprocess(cfg, global_dict)

    def build_criterion(self, cfg: CriterionConfigs):
        """
        Build criterion for the task. If not assigned ,
        :class:`LabelSmoothedCrossEntropyCriterion` will be use as default.

        Note:
            NOT support criterion with parameters yet.

        Args:
            cfg (CriterionConfigs): config object for Criterion.

        Returns:
            Criterion object.
        """
        criterion = None
        for config_field in fields(cfg):
            if config_field.name.startswith('_'):
                continue
            config = getattr(cfg, config_field.name)
            if config.is_active:
                criterion = ConfigStore().get("ofasys.criterion", config_field.name).target(self, config)
                break
        if criterion is None:
            config = getattr(cfg, "cross_entropy")
            criterion = ConfigStore().get("ofasys.criterion", "cross_entropy").target(self, config)
            logger.info(f"No criterion is specified for {self.name}, CrossEntropyCriterion will be used by default.")
        assert not utils.has_parameters(criterion), "NOT support criterion with parameters yet."
        return criterion

    def build_metrics(self, cfg: MetricConfigs) -> List[BaseMetric]:
        """
        Build all metrics for the task.

        Args:
            cfg (MetricConfig): config object for Metrics

        Returns:
            List of metrics.
        """
        metrics = []
        for config_field in fields(cfg):
            if config_field.name.startswith('_'):
                continue
            metric_config = getattr(cfg, config_field.name)
            if metric_config.target_field is None:
                continue
            metrics.append(ConfigStore().get("ofasys.metric", config_field.name).target(metric_config))
        return metrics

    def preprocess_data_and_instruction(self, data, split):
        ist_data = self.preprocess(data, split)
        if ist_data is None:
            return None
        instruction = self.build_instruction(ist_data, split)
        return self.general_preprocess(instruction)

    def get_batch_iterator(self, split, group=None, epoch=1):
        if split != 'train' and self.cfg.dataset.micro_valid_batch_size is not None:
            micro_batch_size = self.cfg.dataset.micro_valid_batch_size
        else:
            micro_batch_size = self.cfg.dataset.micro_batch_size
        if split == 'train':
            update_freq = self.cfg.dataset.update_freq
            if isinstance(update_freq, int):
                update_freq = [update_freq]
        else:
            update_freq = None

        if split == 'train':
            data_paths = self.cfg.dataset.train_data
            shuffle = self.cfg.dataset.shuffle
        elif split == 'valid':
            data_paths = self.cfg.dataset.valid_data
            shuffle = False
        elif split == 'test':
            data_paths = self.cfg.dataset.test_data
            shuffle = False
        else:
            raise ValueError("Unsupported data split: " + split)

        epoch_itr = EpochBatchIterator(
            self.cfg.dataset,
            data_paths=data_paths,
            dataset=self.datasets.get(split, None),
            split=split,
            process_fn=functools.partial(self.preprocess_data_and_instruction, split=split),
            collate_fn=self.general_preprocess.collate,
            update_freq=update_freq,
            batch_size=micro_batch_size,
            num_workers=self.cfg.dataset.num_workers,
            prefetch_factor=self.cfg.dataset.prefetch_factor,
            group=group,
            epoch=epoch,
            seed=self.cfg.dataset.fixed_validation_seed,
            shuffle=shuffle,
        )
        return epoch_itr

    def init_data_iterator(self, split, group=None, itr_state=None):
        assert split not in self.data_iterators
        epoch = itr_state['epoch'] if itr_state is not None else 1
        self.data_iterators[split] = self.get_batch_iterator(split, group=group, epoch=epoch)
        if itr_state is not None:
            self.data_iterators[split].load_state_dict(itr_state)
        if split == 'train':
            self.begin_epoch()

    def get_sample(self, split):
        epoch_itr = self.data_iterators[split]
        return next(epoch_itr.cur_epoch_itr)

    def begin_epoch(self):
        """Hook function called before the start of each epoch."""
        epoch = self.data_iterators['train'].epoch
        logger.info(f"Start iterating over task {self.name}, epoch {epoch}")

    def end_epoch(self):
        epoch = self.data_iterators['train'].epoch
        logger.info(f"End iterating over task {self.name}, epoch {epoch}")

    def begin_valid_epoch(self, epoch, model):
        """Hook function called before the start of each validation epoch."""
        pass

    def build_sequence_generator(self, **gen_kwargs):
        """
        Build a :class:`~ofasys.utils.SequenceGenerator` instance for this
        task.

        Args:
            models (List[~ofasys.model.OFAModel]): ensemble of models
            gen_kwargs (Dict[str, Any]): extra options to pass
                through to SequenceGenerator
        """
        # General generate argument
        beam_size = gen_kwargs.pop("beam", 5)
        return_n_best = gen_kwargs.pop("return_n_best", 1)
        max_len_a = gen_kwargs.pop("max_len_a", 0)
        max_len_b = gen_kwargs.pop("max_len_b", 200)
        max_len = gen_kwargs.pop("max_len", 256)
        min_len = gen_kwargs.pop("min_len", 1)
        normalize_scores = gen_kwargs.pop("normalize_scores", False)
        len_penalty = gen_kwargs.pop("lenpen", 1)
        unk_penalty = gen_kwargs.pop("unkpen", 0)
        temperature = gen_kwargs.pop("temperature", 1.0)
        no_repeat_ngram_size = gen_kwargs.pop("no_repeat_ngram_size", 0)

        # Choose search strategy. Defaults to Beam Search.
        sampling = gen_kwargs.pop("sampling", False)
        sampling_topk = gen_kwargs.pop("sampling_topk", -1)
        sampling_topp = gen_kwargs.pop("sampling_topp", -1.0)
        diverse_beam_groups = gen_kwargs.pop("diverse_beam_groups", -1)
        diverse_beam_strength = gen_kwargs.pop("diverse_beam_strength", 0.5)
        diversity_rate = gen_kwargs.pop("diversity_rate", -1)
        match_source_len = gen_kwargs.pop("match_source_len", False)
        constrained = gen_kwargs.pop("constrained", False)
        constraints = gen_kwargs.pop("constraints", None)

        if (
            sum(
                int(cond)
                for cond in [
                    sampling,
                    diverse_beam_groups > 0,
                    match_source_len,
                    diversity_rate > 0,
                ]
            )
            > 1
        ):
            raise ValueError("Provided Search parameters are mutually exclusive.")
        assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
        assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

        if sampling:
            search_strategy = search.Sampling(self.target_dictionary, sampling_topk, sampling_topp)
        elif diverse_beam_groups > 0:
            search_strategy = search.DiverseBeamSearch(
                self.target_dictionary, diverse_beam_groups, diverse_beam_strength
            )
        elif match_source_len:
            # this is useful for tagging applications where the output
            # length should match the input length, so we hardcode the
            # length constraints for simplicity
            search_strategy = search.LengthConstrainedBeamSearch(
                self.target_dictionary,
                min_len_a=1,
                min_len_b=0,
                max_len_a=1,
                max_len_b=0,
            )
        elif diversity_rate > -1:
            search_strategy = search.DiverseSiblingsSearch(self.target_dictionary, diversity_rate)
        elif constrained:
            search_strategy = search.LexicallyConstrainedBeamSearch(self.target_dictionary, constraints)
        else:
            search_strategy = search.BeamSearch(self.target_dictionary)

        return SequenceGenerator(
            self.target_dictionary,
            beam_size=beam_size,
            return_n_best=return_n_best,
            max_len_a=max_len_a,
            max_len_b=max_len_b,
            max_len=max_len,
            min_len=min_len,
            normalize_scores=normalize_scores,
            len_penalty=len_penalty,
            unk_penalty=unk_penalty,
            temperature=temperature,
            no_repeat_ngram_size=no_repeat_ngram_size,
            search_strategy=search_strategy,
            **gen_kwargs,
        )

    def build_speech_generator(self, **gen_kwargs):
        return AutoRegressiveSpeechGenerator(
            self.source_dictionary,
            stats_npz_path="http://ofasys.oss-cn-zhangjiakou.aliyuncs.com/tasks/tts/gcmvn_stats.npz",
            max_iter=gen_kwargs.pop("max_iter", 1500),
            eos_prob_threshold=gen_kwargs.pop("eos_prob_threshold", 0.5),
        )

    def build_diffusion_generator(self, **gen_kwargs):
        return DiffusionGenerator(
            self.general_preprocess,
            self.diffuser_args,
            **gen_kwargs,
        )

    def build_generator(self, target_modality, **gen_kwargs):
        if target_modality == ModalityType.MOTION:
            return self.build_diffusion_generator(**gen_kwargs)
        elif target_modality == ModalityType.AUDIO:
            return self.build_speech_generator(**gen_kwargs)
        elif target_modality == ModalityType.TEXT:
            return self.build_sequence_generator(**gen_kwargs)
        elif target_modality == ModalityType.BOX:
            assert gen_kwargs["min_len"] == 4 and gen_kwargs["max_len"] == 4
            return self.build_sequence_generator(**gen_kwargs)
        elif target_modality == ModalityType.IMAGE:
            assert gen_kwargs["min_len"] == gen_kwargs["max_len"]
            return self.build_sequence_generator(**gen_kwargs)
        else:
            raise NotImplementedError

    def train_step(self, sample, model, optimizer, update_num, ignore_grad=False):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch from preprocessor.
            model (~ofasys.model.OFAModel): the model
            optimizer (~ofasys.engine.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        self.criterion.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                sample = model.update_sample(sample)
                loss, sample_size, logging_output = self.criterion(model, sample, update_num=update_num)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

    def evaluate(self, model, sample, **kwargs):
        """
        Do inference, and use every metrics to evaluate the inference result.

        Args:
            model (~ofasys.model.OFAModel): the model
            sample (dict): the mini-batch from preprocessor.

        Returns:
            A dict contains compute results from each *Metric*

        """
        hyps = self.inference(model, sample, **kwargs)

        def prepare_for_metric(outputs):
            predict_results = []
            if self.target_modality == ModalityType.TEXT:
                for multi_outputs in outputs:
                    if isinstance(multi_outputs, List):
                        predict_results.append(multi_outputs[0].text)
                    else:
                        predict_results.append(multi_outputs.text)
            elif self.target_modality == ModalityType.IMAGE:
                # todo: is it reasonable to set the valid_batch_size of the image_gen task to 1 ?
                assert len(outputs) == 1
                if isinstance(outputs[0], List):
                    for out in outputs[0]:
                        predict_results.append(out.image)
                else:
                    predict_results.append(outputs[0].image)
            elif self.target_modality == ModalityType.AUDIO:
                for out in outputs:
                    predict_results.append(out.waveform.detach().cpu().numpy().astype(np.float32))
            elif self.target_modality == ModalityType.MOTION:
                for out in outputs:
                    predict_results.append(out.bvh)
            elif self.target_modality == ModalityType.BOX:
                for multi_outputs in outputs:
                    if isinstance(multi_outputs, List):
                        predict_results.append(multi_outputs[0].box)
                    else:
                        predict_results.append(multi_outputs.box)
            elif self.target_modality == ModalityType.CATEGORY:
                for multi_outputs in outputs:
                    if isinstance(multi_outputs, List):
                        predict_results.append(multi_outputs[0].text)
                    else:
                        predict_results.append(multi_outputs.text)
            else:
                raise NotImplementedError
            return predict_results

        hyps = prepare_for_metric(hyps)

        logging_output = {}
        for metric in self.metrics:
            refs = sample[metric.cfg.target_field]
            if self.cfg.evaluation.eval_print_samples and self.target_modality == ModalityType.TEXT:
                logger.info("example hypothesis: " + str(hyps[0]))
                logger.info("example reference: " + str(refs[0]))
            logging_output.update(metric.compute(hyps, refs))
        return logging_output

    def valid_step(self, sample, model):
        """
        Do forward and return the loss as computed by *criterion* for the given *model* and *sample*.
        If the task has any metrics, will also call ``evaluate()``.

        Args:
            sample (dict): the mini-batch from preprocessor.
            model (~ofasys.model.OFAModel): the model

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        self.criterion.eval()
        sample = model.update_sample(sample)
        loss, sample_size, logging_output = self.criterion(model, sample)
        if len(self.metrics) > 0:
            logging_output.update(self.evaluate(model, sample))
        elif self.cfg.evaluation.output_dir:
            self.evaluate(model, sample)
        return loss, sample_size, logging_output

    def optimizer_step(self, optimizer, model, update_num):
        optimizer.step()

    def reduce_metrics(self, logging_outputs, criterion):
        if not any("ntokens" in log for log in logging_outputs):
            warnings.warn("ntokens not found in Criterion logging outputs, cannot log wpb or wps")
        else:
            ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
            metrics.log_scalar("wpb", ntokens, priority=180, round=1)
            metrics.log_speed("wps", ntokens, priority=90, round=1)

        if not any("nsentences" in log for log in logging_outputs):
            warnings.warn("nsentences not found in Criterion logging outputs, cannot log bsz")
        else:
            nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
            metrics.log_scalar("bsz", nsentences, priority=190, round=1)

        criterion.reduce_metrics(logging_outputs, self.name)

        for metric in self.metrics:
            metric.report(logging_outputs)

    def inference(self, model, sample, **kwargs):
        """
        Generate result for given *sample*, and convert the gen_outputs to raw data format
        using ``preprocessor.decode()``.

        Args:
            model (~ofasys.model.OFAModel): the model
            sample (dict): the mini-batch from preprocessor.

        Returns:

        """
        model.eval()
        target_slot = Slot.get_target_slot_from_sample(sample)

        gen_outputs = self.inference_step(generator=self.generator, model=model, sample=sample, **kwargs)

        outputs = self.postprocess(gen_outputs, target_slot=target_slot, **sample)
        return outputs

    def postprocess_for_image_code(self, outputs: BatchGeneratorOutput, **sample):
        preprocessor = self.general_preprocess.name2pre["image_vqgan"]
        adaptor = OFATask._model.decoder.adaptor.name2adaptor["image_vqgan"]

        for idx, single_output in enumerate(outputs):
            if isinstance(single_output, List):
                single_output: MultiGeneratorOutput
                image_codes = (
                    torch.cat([sub_output.tokens.unsqueeze(0) for sub_output in single_output])
                    - preprocessor.code_index_start
                )
                images = adaptor.tokenizer.decode(image_codes, return_pil=True)
                for i, sub_output in enumerate(single_output):
                    sub_output.image = images[i]
                if 'text' in sample:
                    text = sample['text'][idx]
                    logger.info(f'input query {text}, rerank with clip')
                    indices = preprocessor.rerank_with_clip(images, text)
                    new_outputs = []
                    for i in range(len(single_output)):
                        new_outputs.append(single_output[indices[i]])
                    outputs[idx] = new_outputs
            else:
                single_output: SequenceGeneratorOutput
                image_codes = single_output.tokens.unsqueeze(0) - preprocessor.code_index_start
                images = adaptor.tokenizer.decode(image_codes, return_pil=True)
                single_output.image = images[0]
        return outputs

    def postprocess(self, outputs, target_slot: Slot, **sample):
        if target_slot.modality == ModalityType.IMAGE:
            return self.postprocess_for_image_code(outputs, **sample)
        return self.general_preprocess.postprocess(outputs, **sample)

    def inference_step(self, generator, model, sample, **kwargs):
        """
        Generate result for given *sample*.

        Args:
            generator: object of decoding strategy.
            model (~ofasys.model.OFAModel): the model
            sample (dict): the mini-batch from preprocessor.
        Returns:

        """
        with torch.no_grad():
            return generator.generate(model, sample, **kwargs)

    def state_dict(self):
        # TODO: add self.data_iterators to state_dict
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        # TODO: load self.data_iterators from state_dict
        pass

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.cfg.max_source_positions, self.cfg.max_target_positions)

    @staticmethod
    def logging_outputs_can_be_summed(criterion) -> bool:
        """
        Whether the logging outputs returned by `train_step` and `valid_step` can
        be summed across workers prior to calling `aggregate_logging_outputs`.
        Setting this to True will improve distributed training speed.
        """
        return criterion.logging_outputs_can_be_summed()

    @property
    def source_dictionary(self):
        """Return the source :class:`~Dictionary`."""
        return self.global_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~Dictionary`."""
        return self.global_dict

    @property
    def tgt_dict(self):
        return self.global_dict

    @property
    def src_dict(self):
        return self.global_dict

    @property
    def bpe(self):
        return self.general_preprocess.bpe


def update_adaptor_config_by_names(cfg: OFAAdaptorConfig, adaptor_name_activated: Set[str]):
    for adaptor_name in cfg.__annotations__:
        if adaptor_name.startswith('_'):
            continue
        if adaptor_name in adaptor_name_activated:
            setattr(getattr(cfg, adaptor_name), 'is_active', True)
    return


def collect_adaptor_name_from_tasks(tasks: list) -> Set[str]:
    encoder_adaptors = set()
    decoder_adaptors = set()
    for task in tasks:
        for template in task.templates:
            ist = Instruction(template)
            for slot in ist.slots:
                if slot.is_src:
                    encoder_adaptors.add(
                        slot.get_attr('adaptor') if slot.has_attr('adaptor') else default_adaptor[slot.modality]
                    )
                else:
                    decoder_adaptors.add(
                        slot.get_attr('adaptor') if slot.has_attr('adaptor') else default_adaptor[slot.modality]
                    )
    logger.info(f"Encoder adaptor {'，'.join(encoder_adaptors)} be activated!")
    logger.info(f"Decoder adaptor {'，'.join(decoder_adaptors)} be activated!")
    return encoder_adaptors | decoder_adaptors


def update_preprocess_config_by_template(cfg: PreprocessConfig, templates: List[str], task_name):
    assert templates is not None, f"{task_name}'s templates is None"
    all_preprocess_name = set()
    for template in templates:
        ist = Instruction(template)
        for slot in ist.slots:
            all_preprocess_name.add(
                slot.get_attr('preprocess') if slot.has_attr('preprocess') else default_preprocess[slot.modality]
            )
    for pre_name in cfg.__annotations__:
        if pre_name in all_preprocess_name:
            setattr(getattr(cfg, pre_name), 'is_active', True)
        else:
            setattr(getattr(cfg, pre_name), 'is_active', False)
    logger.info(f"Preprocess {'，'.join(all_preprocess_name)} of Task:{task_name} be activated!")


def warning_for_bos_eos(templates: List[str]):
    if templates is None:
        return
    att_warnings = set()
    token_warnings = set()
    for template in templates:
        ist = Instruction(template)
        for slot in ist.slots:
            if slot.has_attr('add_bos'):
                att_warnings.add('add_bos')
            if slot.has_attr('add_eos'):
                att_warnings.add('add_eos')
            if isinstance(slot.value, str):
                if '<BOS>' in slot.value:
                    token_warnings.add('<BOS>')
                if '<EOS>' in slot.value:
                    token_warnings.add('<EOS>')
    if att_warnings:
        logger.warning(f"Attributs {', '.join(att_warnings)} will be ignored!")
    if token_warnings:
        logger.warning(f"Tokens {', '.join(token_warnings)} will be treated as plain text!")
