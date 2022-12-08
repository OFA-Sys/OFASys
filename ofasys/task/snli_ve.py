# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import re
from typing import Any, Dict, List, Union

from ofasys.configure import register_config
from ofasys.generator import MultiGeneratorOutput, SequenceGeneratorOutput
from ofasys.preprocessor.default.text import remove_punctuation
from ofasys.task.base import OFATask, TaskConfig


@register_config("ofasys.task", "snli_ve", dataclass=TaskConfig)
class SNLIVEGenTask(OFATask):
    def __init__(self, cfg: TaskConfig, **kwargs):
        super().__init__(cfg, **kwargs)

    def initialize(self, global_dict, **kwargs):
        super().initialize(global_dict, **kwargs)
        text_pre = self.general_preprocess.name2pre['text']
        self.label2ans_d = {v: k for k, v in text_pre.ans2label_dict.items()}

    def preprocess(self, data: Dict[str, Any], split: str) -> Dict[str, Any]:
        def pre_caption(caption, max_words):
            caption = (
                caption.lower().lstrip(",.!?*#:;~").replace('-', ' ').replace('/', ' ').replace('<person>', 'person')
            )
            caption = re.sub(r"\s{2,}", ' ', caption).rstrip('\n').strip(' ')
            # truncate caption
            caption_words = caption.split(' ')
            if len(caption_words) > max_words:
                caption = ' '.join(caption_words[:max_words])
            return caption

        if data.get('label', None) is not None:
            data['label'] = self.label2ans_d[data['label']]

        data['hyp'] = pre_caption(data['hyp'], self.cfg.max_src_length)
        data['cap'] = pre_caption(data['cap'], self.cfg.max_src_length)
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
