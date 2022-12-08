# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import hashlib
import os
import random
import warnings
from typing import Any, Dict, List, Union

from PIL import Image, ImageFile

from ofasys.configure import register_config
from ofasys.generator import MultiGeneratorOutput, SequenceGeneratorOutput
from ofasys.preprocessor import Instruction
from ofasys.preprocessor.default.base import PreprocessSkipException
from ofasys.preprocessor.default.video import (
    DefaultVideoPreprocess,
    video_tensor_normalize_reverse,
)
from ofasys.preprocessor.instruction import Slot
from ofasys.task.base import OFATask, TaskConfig

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


@register_config("ofasys.task", "video_generation", dataclass=TaskConfig)
class VideoGenerationTask(OFATask):
    def preprocess(self, data: Dict[str, Any], split: str) -> Instruction:
        instruction: Instruction = super().preprocess(data, split)
        slot_by_name: Dict[str, Slot] = {}
        target_slot: Slot = None
        video_src_preprocessor: DefaultVideoPreprocess = None
        for slot in instruction.slots:
            if slot.is_src:
                if slot.column_name == 'video':
                    video_hash = int(hashlib.md5(slot.value.encode()).hexdigest(), base=16)
                try:
                    self.general_preprocess.get_preprocess(slot).map(slot)
                except PreprocessSkipException:
                    return instruction
                if isinstance(self.general_preprocess.get_preprocess(slot), DefaultVideoPreprocess):
                    video_src_preprocessor = self.general_preprocess.get_preprocess(slot)
                slot_by_name[slot.column_name] = slot
            else:
                target_slot = slot
        assert video_src_preprocessor is not None
        assert target_slot is not None
        denorm_value = video_tensor_normalize_reverse(
            slot_by_name['video'].value, video_src_preprocessor.mean, video_src_preprocessor.std
        )
        condition_index = (random.randint(0, denorm_value.size(1) - 1) + video_hash) % denorm_value.size(1)

        slot_by_name['video'].value[:, condition_index:, :, :] *= 0.0
        target_slot.value = denorm_value[:, condition_index, :, :]
        return instruction

    def build_sequence_generator(self, **gen_kwargs):
        preprocess = self.general_preprocess.name2pre["image_vqgan"]

        constraint_start = preprocess.code_index_start
        constraint_end = preprocess.code_index_start + preprocess.num_codes
        gen_kwargs.update(**{"constraint_range": f"({constraint_start},{constraint_end})"})

        return super().build_sequence_generator(**gen_kwargs)

    def inference(self, model, sample, **kwargs):
        outputs = super().inference(model, sample, **kwargs)

        if self.cfg.evaluation.output_dir:
            os.makedirs(self.cfg.evaluation.output_dir, exist_ok=True)

            single_output: SequenceGeneratorOutput
            multi_output: Union[MultiGeneratorOutput, SequenceGeneratorOutput]
            for caption, multi_output in zip(sample['cap'], outputs):
                if isinstance(multi_output, List):
                    for i, single_output in enumerate(multi_output):
                        image_name = os.path.join(self.cfg.evaluation.output_dir, caption + f"_{i}")
                        single_output.save_image(image_name)
                else:
                    single_output = multi_output
                    image_name = os.path.join(self.cfg.evaluation.output_dir, caption)
                    single_output.save_image(image_name)
