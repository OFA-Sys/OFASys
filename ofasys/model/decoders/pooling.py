from torch import Tensor
from torch.nn import Linear
from typing import List, Optional, Dict
from dataclasses import dataclass, field
from ofasys.adaptor.general import OFAGeneralAdaptor
from ofasys.configure.config_store import register_config
from ofasys.configure.configs import BaseDataclass
from ofasys.model.base_decoder import BaseDecoder
from ofasys.module.layer_norm import LayerNorm
from ofasys.preprocessor.instruction import Slot
import torch.nn.functional as F
import logging
logger = logging.getLogger(__name__)


@dataclass
class OFAPoolingModelConfig(BaseDataclass):
    embed_dim: int = field(
        default=512,
        metadata={"help": "embedding dimension"},
    )
    normalize_before: bool = field(
        default=False,
        metadata={"help": "encoder drop path rate"},
    )
    normalize_after: bool = field(
        default=False,
        metadata={"help": "decoder drop path rate"},
    )

    pooling_position_begin: int = field(
        default=0,
        metadata={"help": "attention scale factor"},
    )

    pooling_position_end: int = field(
        default=0,
        metadata={"help": "attention scale factor"},
    )


@register_config("ofasys.model.extra_decoders", "pooling", dataclass=OFAPoolingModelConfig)
class OFAPoolingModel(BaseDecoder):

    def __init__(
        self,
        cfg: OFAPoolingModelConfig,
        dictionary,
        adaptor,
        no_encoder_attn=False,
    ):
        self.cfg = cfg
        super().__init__(dictionary)
        self.pooling_position_begin: int = cfg.pooling_position_begin
        self.pooling_position_end: int = cfg.pooling_position_end
        self.layernorm_before: Optional[LayerNorm] = None
        self.layernorm_after: Optional[LayerNorm] = None
        self.adaptor = [adaptor]
        if self.cfg.normalize_before:
            self.layernorm_before = LayerNorm(cfg.embed_dim)
        if self.cfg.normalize_after:
            self.layernorm_after = LayerNorm(cfg.embed_dim)

    def forward(
        self,
        slots: List[Slot],
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str,
                                         Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        return_all_hiddens: bool = False,
        return_all_attention_weights: bool = False,
    ):
        assert encoder_out is not None and len(encoder_out["encoder_out"]) > 0
        x = encoder_out["encoder_out"][0]
        x = x.transpose(0, 1).contiguous()
        # assert x.size()[1] == bsz, f"Expected enc.shape == (t, {bsz}, c) got {enc.shape}"
        if self.layernorm_before is not None:
            x = self.layernorm_before(x)
        if self.pooling_position_end == -1:
            x = x[:, self.pooling_position_begin:, :].mean(dim=1)
        else:
            x = x[:, self.pooling_position_begin:self.pooling_position_end + 1, :].mean(dim=1)
        if self.layernorm_after is not None:
            x = self.layernorm_after(x)
        extra = {}
        extra['last_hidden_state'] = x
        if not features_only:
            adaptor_output, extra = self.adaptor[0].forward_output(x, extra, slots)
            return adaptor_output, extra
        return x, extra
        #return [x]

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return 1

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        logits = net_output[0].float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of OFA."""

        prefix = name + "." if name != "" else ""
        for param_name, param_tensor in self.state_dict().items():
            if (prefix + param_name) not in state_dict:
                state_dict[prefix + param_name] = self.state_dict()[param_name]
                logger.info('not found in checkpoint: %s%s' % (prefix, param_name))

        return state_dict
