# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from typing import List, Union

from ofasys.configure import register_config
from ofasys.generator import MultiGeneratorOutput, SequenceGeneratorOutput
from ofasys.preprocessor.default.text import remove_punctuation
from ofasys.task.base import OFATask, TaskConfig


@register_config("ofasys.task", "glue", dataclass=TaskConfig)
class GLUEGenTask(OFATask):
    def __init__(self, cfg: TaskConfig, **kwargs):
        super().__init__(cfg, **kwargs)

    def initialize(self, global_dict, **kwargs):
        super().initialize(global_dict, **kwargs)
        text_pre = self.general_preprocess.name2pre['text']
        self.label2ans_d = {v: k for k, v in text_pre.ans2label_dict.items()}

    def preprocess(self, data, split):
        if data.get('label', None) is not None:
            data['label'] = self.label2ans_d[data['label']]

        for key in data.keys():
            if key == 'label':
                continue
            data[key] = ' '.join(data[key].lower().strip().split()[: self.cfg.max_src_length])
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
