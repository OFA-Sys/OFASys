# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import logging
from dataclasses import asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from dacite import Config, from_dict
from torch import nn

from ofasys import ModalityType
from ofasys.adaptor import OFAAdaptorConfig
from ofasys.configure import ConfigStore
from ofasys.module import utils
from ofasys.module.utils import apply_to_sample
from ofasys.preprocessor import (
    Dictionary,
    Instruction,
    PreprocessConfig,
    Slot,
    default_preprocess,
)
from ofasys.task.base import OFATask, TaskConfig
from ofasys.utils import checkpoint_utils

logger = logging.getLogger(__name__)


class OFASys(nn.Module):
    def __init__(self, tasks, cfg, model, task_name=None, seed=42):
        """OFASys provides an easy-to-use inferface that allows users to load ckpt
        and use different instructions for inference.

        .. note::
            We do not recommend calling the ``__init__`` function directly.
            Call ``from_pretrained`` instead.

        Args:
            tasks: the list of tasks.
            cfg: configuration object.
            model: model object.
            task_name: if not None, use specified task, rather than OFATask.
            seed: random seed.
        """
        super().__init__()
        self.cfg = cfg
        self.tasks = tasks
        self.model = model
        self.default_task = tasks[task_name] if task_name is not None and task_name in self.tasks else tasks["default"]
        self.current_task = None
        OFATask._model = model
        np.random.seed(seed)
        utils.set_torch_seed(seed)
        self.model.prepare_for_inference_(cfg)
        # this is useful for determining the device
        self.register_buffer("_float_tensor", torch.tensor([0], dtype=torch.float))

    @property
    def device(self):
        return self._float_tensor.device

    @property
    def dtype(self):
        return self._float_tensor.dtype

    def cuda(self, device=None):
        cuda_object = super().cuda(device=device)
        try:
            audio_preprocessor = cuda_object.task.general_preprocess.name2pre["audio"]
            audio_preprocessor.vocoder = audio_preprocessor.vocoder.cuda(device=device)
        except (KeyError, AttributeError):
            pass
        return cuda_object

    def cpu(self):
        cpu_object = super().cpu()
        try:
            audio_preprocessor = cpu_object.task.general_preprocess.name2pre["audio"]
            audio_preprocessor.vocoder = audio_preprocessor.vocoder.cpu()
        except (KeyError, AttributeError):
            pass
        return cpu_object

    def half(self):
        half_object = super().half()
        try:
            audio_preprocessor = half_object.task.general_preprocess.name2pre["audio"]
            audio_preprocessor.vocoder = audio_preprocessor.vocoder.half()
        except (KeyError, AttributeError):
            pass
        return half_object

    def double(self):
        double_object = super().double()
        try:
            audio_preprocessor = double_object.task.general_preprocess.name2pre["audio"]
            audio_preprocessor.vocoder = audio_preprocessor.vocoder.double()
        except (KeyError, AttributeError):
            pass
        return double_object

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        task_name=None,
        initialize_all_tasks=False,
    ):
        """
        Load pretrained OFASys ckpt and config from the given path.

        args:
            model_path: pretrained ckpt path.
            task_name: if not None, the specified task will be used, rather than OFATask.
            initialize_all_tasks: if True, all pretraining tasks will be initialized.

        Returns:
            OFASys:

        """
        state = checkpoint_utils.load_checkpoint_to_cpu(model_path)
        upgrade_state_dict(state)
        cfg = state["cfg"]

        tasks = {}
        global_dict = Dictionary()
        task_cfg_default = TaskConfig()
        
        if len(state['configstore']['task']) == 1:
            task_name, task_dict = list(state['configstore']['task'].items())[0]
            node = ConfigStore().get("ofasys.task", task_name)
            task_cfg = from_dict(data_class=node.config.__class__, data=task_dict, config=Config(cast=[Enum]))
            task_name = taskname_alias.get(task_name, task_name)
            tasks[task_name] = node.target(task_cfg)
            tasks[task_name].initialize(global_dict, is_train=False)
        else:
            for cur_task, task_dict in state['configstore']['task'].items():
                node = ConfigStore().get("ofasys.task", cur_task)
                task_cfg = from_dict(data_class=node.config.__class__, data=task_dict, config=Config(cast=[Enum]))
                update_preprocess_cfg_by_another_cfg(task_cfg_default.preprocess, task_cfg.preprocess)
                cur_task = taskname_alias.get(cur_task, cur_task)
                if initialize_all_tasks or (task_name is not None and task_name == cur_task):
                    tasks[cur_task] = node.target(task_cfg)
                    tasks[cur_task].initialize(global_dict, is_train=False)
    
            tasks['default'] = OFATask(task_cfg_default)
            tasks['default'].initialize(global_dict, is_train=False)
    
        model_name, model_dict = list(state["configstore"]["model"].items())[0]
        node = ConfigStore().get("ofasys.model", model_name)
        model_cfg = from_dict(data_class=node.config.__class__, data=model_dict, config=Config(cast=[Enum]))
        update_adaptor_config_by_ckpt(model_cfg.adaptor, state)
        model = node.target(model_cfg)
        model.initialize(global_dict)
        model.load_state_dict(state["model"], strict=True, model_cfg=model_cfg)

        return cls(tasks, cfg, model, task_name=task_name)

    def __call__(
        self,
        instructions_or_tasks: Union[str, Instruction, List[Union[str, Instruction]]],
        data: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        closed_set: Optional[Dict[str, Any]] = None,
        batch_size: int = 1,
        **gen_kwargs,
    ):
        return self.inference(
            instructions_or_tasks, data=data, closed_set=closed_set, batch_size=batch_size, **gen_kwargs
        )

    def inference(
        self,
        instructions_or_tasks: Union[str, Instruction, List[Union[str, Instruction]]],
        data: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        closed_set: Optional[Dict[str, Any]] = None,
        batch_size: int = 1,
        beam_size: Optional[int] = None,  # for sequence generator
        max_len: Optional[int] = None,
        min_len: Optional[int] = None,
        len_penalty: Optional[float] = None,
        unk_penalty: Optional[float] = None,
        temperature: Optional[float] = None,
        sampling: Optional[bool] = None,
        sampling_topk: Optional[int] = None,
        sampling_topp: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        return_n_best: Optional[int] = None,
        max_iter: Optional[int] = None,  # for speech generator
        **extra_gen_kwargs,
    ):
        """
        Perform free-style inference according to the instruction using the loaded ckpt.
        Generator parameters will be transparently passed in.
        Single sample or list of samples are both supported.

        Args:
            instructions_or_tasks: formatted instruction object, or template string, or task name, or List of them.
            data: data to fill in slots in instrcution.
            closed_set: perform a constraint generation on the given candidates set (default: None).
            batch_size: batch size of data (default: 1).
            beam_size: beam width (default: 5).
            max_len: the maximum length of the generated output  (not including end-of-sentence) (default: 256)
            min_len: the minimum length of the generated output  (not including end-of-sentence) (default: 1)
            len_penalty: length penalty, where <1.0 favors shorter, >1.0 favors longer sentences (default: 1.0)
            unk_penalty: unknown word penalty, where <0 produces more unks, >0 produces fewer (default: 0.0)
            temperature:  temperature, where values>1.0 produce more uniform samples and values <1.0 produce sharper
                samples (default: 1.0)
            sampling: whether use sampling instead of beam search (default: false)
            sampling_topk: sample from the k most likely tokens at each step (default: -1).
            sampling_topp:  sample among the smallest set of tokens whose cumulative probability mass exceeds p
                at each step (default: -1.0)
            no_repeat_ngram_size: prevent decoding of ngrams that have already appeared (default: 3).
            return_n_best: return best n results (default: -1, which indicates beam_size)
            max_iter: max iteration steps for SpeechGenerator (default: 1500).
            output_shape: output shape for DiffusionGenerator (default: None).
        """
        gen_kwargs = extra_gen_kwargs
        user_gen_kwargs = {
            # for sequence generator
            "beam": beam_size,
            "max_len": max_len,
            "min_len": min_len,
            "lenpen": len_penalty,
            "unkpen": unk_penalty,
            "temperature": temperature,
            "sampling": sampling,
            "sampling_topk": sampling_topk,
            "sampling_topp": sampling_topp,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "return_n_best": return_n_best,
            # for speech generator
            "max_iter": max_iter,
        }
        user_gen_kwargs = dict(filter(lambda x: x[1], user_gen_kwargs.items()))
        gen_kwargs.update(user_gen_kwargs)

        is_list = isinstance(instructions_or_tasks, list) or isinstance(data, list)
        if is_list:
            return self.inference_multi(
                instructions_or_tasks, data=data, closed_set=closed_set, batch_size=batch_size, **gen_kwargs
            )
        else:
            return self.inference_single(instructions_or_tasks, data=data, closed_set=closed_set, **gen_kwargs)

    @property
    def task(self):
        if self.current_task is None:
            return self.default_task
        return self.current_task

    @task.setter
    def task(self, cur_task):
        self.current_task = cur_task

    def build_instruction(
        self,
        instruction_or_template: Union[str, Instruction],
        data: Optional[Dict[str, Any]] = None,
        split: str = 'test',
    ):
        """
        Fill template with input data.
        """
        if isinstance(instruction_or_template, str):
            instruction_or_template = Instruction(instruction_or_template, split=split)
        assert instruction_or_template.split == split, f"instruction_or_template.split must be {split}"
        if data is None:
            data = {}
        else:
            data = self.task.preprocess(data, split)
        return instruction_or_template.format(**data)

    def build_sample(self, instructions: Union[Instruction, List[Instruction]]):
        """
        Convert instruction into batched input data by calling the Generalpreprocess.
        """
        if not isinstance(instructions, list):
            instructions = [instructions]

        processed_instructions = [self.task.general_preprocess(ist) for ist in instructions]
        sample = self.task.general_preprocess.collate(processed_instructions)
        sample = apply_to_sample(
            lambda t: t.to(dtype=self.dtype if t.dtype == torch.float32 else t.dtype, device=self.device), sample
        )
        return sample

    def prepare_for_generation(
        self, instruction: Instruction, closed_set: Optional[Dict[str, Any]] = None, **gen_kwargs
    ):
        """
        Parse the instruction and init the generator object for the target slot.
        """
        target_slot: Slot = Slot.get_target_slot_from_slots(instruction.slots)
        preprocess = self.task.general_preprocess.get_preprocess(target_slot)
        self.task.target_preprocess = (
            target_slot.get_attr('preprocess')
            if target_slot.get_attr('preprocess') is not None
            else default_preprocess[target_slot.modality]
        )

        if target_slot.modality == ModalityType.TEXT:
            preprocess.prepare_for_generation(closed_set)
            if closed_set is not None and "beam" not in gen_kwargs:
                gen_kwargs["beam"] = 1
            if "no_repeat_ngram_size" not in gen_kwargs:
                if "beam" not in gen_kwargs or gen_kwargs["beam"] > 1:
                    gen_kwargs["no_repeat_ngram_size"] = 3
            gen_kwargs["constraint_trie"] = self.task.general_preprocess.name2pre['text'].constraint_trie
        elif target_slot.modality == ModalityType.IMAGE:
            if "beam" not in gen_kwargs:
                gen_kwargs["beam"] = 20
            if "return_n_best" not in gen_kwargs:
                gen_kwargs["return_n_best"] = -1
            if "sampling" not in gen_kwargs:
                gen_kwargs["sampling"] = True
            gen_kwargs["min_len"] = 1024
            gen_kwargs["max_len"] = 1024
            constraint_start = preprocess.code_index_start
            constraint_end = preprocess.code_index_start + preprocess.num_codes
            gen_kwargs["constraint_range"] = f"({constraint_start},{constraint_end})"
        elif target_slot.modality == ModalityType.BOX:
            gen_kwargs["min_len"] = 4
            gen_kwargs["max_len"] = 4
        elif target_slot.modality == ModalityType.MOTION:
            gen_kwargs["device"] = self.device
            gen_kwargs["dtype"] = self.dtype

        self.task.generator = self.task.build_generator(target_modality=target_slot.modality, **gen_kwargs)

    def inference_single(
        self,
        instruction_or_task: Union[str, Instruction],
        data: Optional[Dict[str, Any]] = None,
        closed_set: Optional[Dict[str, Any]] = None,
        **gen_kwargs,
    ):
        if (
            isinstance(instruction_or_task, str)
            and instruction_or_task != "default"
            and instruction_or_task in self.tasks
        ):
            self.task = self.tasks[instruction_or_task]
            instruction_or_template = self.task.templates[0]
        else:
            self.task = self.default_task
            instruction_or_template = instruction_or_task

        instruction = self.build_instruction(instruction_or_template, data, split='test')
        self.prepare_for_generation(instruction, closed_set, **gen_kwargs)
        sample = self.build_sample(instruction)
        outputs = self.task.inference(self.model, sample)
        return outputs[0]

    def inference_multi(
        self,
        instructions_or_tasks: Union[str, Instruction, List[Union[str, Instruction]]],
        data: Optional[List[Dict[str, Any]]] = None,
        closed_set: Optional[Dict[str, Any]] = None,
        batch_size: int = 1,
        **gen_kwargs,
    ):
        if isinstance(instructions_or_tasks, list):
            if data is None:
                return [
                    self.inference_single(item, closed_set=closed_set, **gen_kwargs) for item in instructions_or_tasks
                ]
            else:
                assert len(instructions_or_tasks) == len(
                    data
                ), "The length of `instructions_or_templates` and `data` must match."
                return [
                    self.inference_single(item, data_item, closed_set=closed_set, **gen_kwargs)
                    for item, data_item in zip(instructions_or_tasks, data)
                ]
        else:
            if (
                isinstance(instructions_or_tasks, str)
                and instructions_or_tasks != "default"
                and instructions_or_tasks in self.tasks
            ):
                self.task = self.tasks[instructions_or_tasks]
                instructions_or_templates = self.task.templates[0]
            else:
                self.task = self.default_task
                instructions_or_templates = instructions_or_tasks

            instructions = [
                self.build_instruction(instructions_or_templates, data_item, split='test') for data_item in data
            ]
            self.prepare_for_generation(instructions[0], closed_set, **gen_kwargs)

            batch = []
            total_outputs = []
            for i, item in enumerate(instructions):
                batch.append(item)
                if (i + 1) % batch_size == 0:
                    sample = self.build_sample(batch)
                    outputs = self.task.inference(self.model, sample)
                    total_outputs.extend(outputs)
                    batch = []

            if len(batch) > 0:
                sample = self.build_sample(batch)
                outputs = self.task.inference(self.model, sample)
                total_outputs.extend(outputs)
            return total_outputs


# TODO: change it in ofasys/task/xxx.py
taskname_alias = {
    'gigaword': 'text_summary',
    'refcoco': 'image_grounding',
    'dart': 'table2text',
    'snli_ve': 'text_entailment',
    'diffusion': 'motion_diffusion',
}


def upgrade_state_dict(state):
    # TODO: unify ckpt format
    try:
        del state['configstore']['model']['unify']['adaptor']['image_vqgan']['vqgan_model_path']
    except:
        pass
    try:
        del state['configstore']['model']['unify']['adaptor']['image_vqgan']['vqgan_config_path']
    except:
        pass
    if 'image_gen' in state['configstore']['task']:
        del state['configstore']['task']['image_gen']['evaluation']['output_dir']
    for task_cfg in state['configstore']['task'].values():
        task_cfg['dataset']['num_workers'] = 0
        if 'location' in task_cfg['preprocess']:
            task_cfg['preprocess']['box'] = task_cfg['preprocess']['location']
            del task_cfg['preprocess']['location']
        if 'active_preprocessors' in state:
            for key, val in task_cfg['preprocess'].items():
                if key != '_name':
                    val['is_active'] = key in state['active_preprocessors']


def update_adaptor_config_by_ckpt(cfg: OFAAdaptorConfig, state):
    keys = map(lambda x: x.split('.', 3), state['model'].keys())
    adaptor_names = set()
    for key in keys:
        if len(key) >= 3 and key[0] in ('encoder', 'decoder') and key[1] == 'adaptor':
            adaptor_names.add(key[2])
    adaptor_names_activated = set()
    for adaptor_name in cfg.__annotations__:
        if adaptor_name.startswith('_'):
            continue
        if adaptor_name in adaptor_names:
            setattr(getattr(cfg, adaptor_name), 'is_active', True)
            adaptor_names_activated.add(adaptor_name)
        else:
            setattr(getattr(cfg, adaptor_name), 'is_active', False)
    logger.info(f"Adaptor {'，'.join(adaptor_names_activated)} be activated!")
    return


def update_preprocess_cfg_by_another_cfg(tgt_cfg: PreprocessConfig, src_cfg: PreprocessConfig):
    for pre_name, cfg in src_cfg.__dataclass_fields__.items():
        if pre_name.startswith('_'):
            continue
        if getattr(src_cfg, pre_name).is_active:
            setattr(getattr(tgt_cfg, pre_name), 'is_active', True)
        # src_pre_cfg = getattr(src_cfg, pre_name)
        # tgt_pre_cfg = getattr(tgt_cfg, pre_name)
        #
        # if src_pre_cfg.is_active:
        #     if not tgt_pre_cfg.is_active:
        #         tgt_pre_cfg = from_dict(
        #             data_class=tgt_pre_cfg.__class__, data=asdict(src_pre_cfg), config=Config(cast=[Enum])
        #         )
        #         setattr(tgt_cfg, pre_name, tgt_pre_cfg)
    return
