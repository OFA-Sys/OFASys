# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from typing import Any, Dict, List, Union

from ofasys.configure import register_config
from ofasys.generator import MultiGeneratorOutput, SequenceGeneratorOutput
from ofasys.metric.bleu import fix_tokenization
from ofasys.task.base import OFATask, TaskConfig


@register_config("ofasys.task", "gigaword", dataclass=TaskConfig)
class GigawordTask(OFATask):
    def preprocess(self, data: Dict[str, Any], split: str) -> Dict[str, Any]:
        data['src'] = data['src'].lower().replace('<unk>', 'unk')
        if data.get('tgt', None) is not None:
            data['tgt'] = data['tgt'].lower().replace('<unk>', 'unk')
            data['ref'] = fix_tokenization(data['tgt']).replace('<unk>', ' unk').replace('1', '#')
        return data

    def inference(self, model, samples, **kwargs):
        hyps = super().inference(model, samples, **kwargs)

        multi_hyps: Union[SequenceGeneratorOutput, MultiGeneratorOutput]
        for multi_hyps in hyps:
            if isinstance(multi_hyps, List):
                for hyp in multi_hyps:
                    hyp.text = fix_tokenization(hyp.text).replace('<unk>', ' unk').replace('1', '#')
            else:
                multi_hyps.text = fix_tokenization(multi_hyps.text).replace('<unk>', ' unk').replace('1', '#')
        return hyps
