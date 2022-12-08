# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from typing import Any, Dict, List, Union

from ofasys.configure import register_config
from ofasys.generator import MultiGeneratorOutput, SequenceGeneratorOutput
from ofasys.preprocessor.default.text import remove_punctuation
from ofasys.task.base import OFATask, TaskConfig


@register_config("ofasys.task", "speech_to_text", dataclass=TaskConfig)
class Speech2TextTask(OFATask):
    def preprocess(self, data: Dict[str, Any], split: str) -> Dict[str, Any]:
        if data.get('text', None) is not None:
            if split == 'train':
                text = data['text']
                text = remove_punctuation(text).strip()
                text_token_list = text.strip().split()
                text = ' '.join(text_token_list[: self.cfg.max_tgt_length])
                data['text'] = text

        return data

    def inference(self, model, samples, **kwargs):
        hyps = super().inference(model, samples, **kwargs)

        multi_hyps: Union[SequenceGeneratorOutput, MultiGeneratorOutput]
        for multi_hyps in hyps:
            if isinstance(multi_hyps, List):
                for hyp in multi_hyps:
                    hyp.text = remove_punctuation(hyp.text).strip()
            else:
                multi_hyps.text = remove_punctuation(multi_hyps.text).strip()
        return hyps
