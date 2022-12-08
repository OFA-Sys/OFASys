# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import os
from typing import Any, Dict, List, Union

from ofasys.configure import register_config
from ofasys.generator import MultiGeneratorOutput, SequenceGeneratorOutput
from ofasys.preprocessor.default.text import remove_punctuation
from ofasys.task.base import OFATask, TaskConfig


@register_config("ofasys.task", "image_gen", dataclass=TaskConfig)
class ImageGenTask(OFATask):
    def preprocess(self, data: Dict[str, Any], split: str) -> Dict[str, Any]:
        caption = data['cap']
        if '&&' in caption:
            caption = caption.split('&&')[0]
        caption = remove_punctuation(caption).strip().lower()
        caption_token_list = caption.strip().split()
        caption = ' '.join(caption_token_list[: self.cfg.max_src_length])
        data['cap'] = caption
        return data

    def build_sequence_generator(self, **gen_kwargs):
        preprocess = self.general_preprocess.name2pre["image_vqgan"]

        constraint_start = preprocess.code_index_start
        constraint_end = preprocess.code_index_start + preprocess.num_codes
        gen_kwargs.update(**{"constraint_range": f"({constraint_start},{constraint_end})"})

        return super().build_sequence_generator(**gen_kwargs)

    def inference(self, model, sample, **kwargs):
        # todo(fix it): we usually set valid_batch_size to 1 for image_gen  task.
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

        return outputs
