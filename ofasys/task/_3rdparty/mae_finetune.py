from typing import List, Dict, Optional
from ofasys.configure import register_config
from ofasys.generator.base import BatchGeneratorOutput
from ofasys.generator.sequence_generator import SequenceGeneratorOutput
from ofasys.model.ofa import OFAEncoderDecoderExecutor
from ofasys.preprocessor.instruction import Slot
from ..base import TaskConfig
from ofasys.task.base import OFATask
import torch
from torch import nn
from torch import Tensor
from ofasys import ModalityType

class SimpleSequenceGenerator(nn.Module):
    @torch.no_grad()
    def generate(self, model, sample, **kwargs):
        """
            Generate function. Should be overridden by all subclasses.
        """
        model.eval()
        net_input = sample["net_input"]
        decoder_outs = model.forward(slots=net_input['slots'])
        #print('decoder_outs', decoder_outs[0].shape)
        tokens = torch.argmax(decoder_outs[0], dim=-1).unsqueeze(-1)
        return [SequenceGeneratorOutput(
            tokens=tokens,
            score=None,
            attention=None,
            positional_scores=None
        )]

@register_config("ofasys.task", "mae_finetune", dataclass=TaskConfig)
class MAEFinetuneTask(OFATask):
    def __init__(self, cfg: TaskConfig):
        super().__init__(cfg)
        self.executor = OFAEncoderDecoderExecutor(
            encoder_name='transformer_encoder',
            decoder_name='pooling_decoder')
        self.generator = SimpleSequenceGenerator()

    def inference_step(self, generator, model, sample, **kwargs):
        with model.executor_context(self.executor):
            return super().inference_step(generator, model, sample, **kwargs)

    def valid_step(self, sample, model):
        with model.executor_context(self.executor):
            return super().valid_step(sample, model)

    def train_step(self, sample, model, optimizer, update_num, ignore_grad=False):
        with model.executor_context(self.executor):
            return super().train_step(sample, model, optimizer, update_num, ignore_grad)