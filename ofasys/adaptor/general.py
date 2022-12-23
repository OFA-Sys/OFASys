# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import logging
from dataclasses import fields
from typing import Any, Dict, List

import torch
from torch import Tensor

from ofasys import ModalityType
from ofasys.adaptor.base import AdaptorOutput, BaseAdaptor, Slot
from ofasys.configure import ConfigStore, auto_import
from ofasys.module import Embedding
from ofasys.utils.logging_utils import master_logging

logger = logging.getLogger(__name__)

auto_import(__file__)


# ConfigStore().make_dataclass implements the same function as the following code
# @dataclass
# class OFAAdaptorConfig(BaseDataclass):
#     text: TextAdaptorConfig = TextAdaptorConfig()
#     image_resnet: ImageResnetAdaptorConfig = ImageResnetAdaptorConfig()
#     image_vqgan: ImageVqganAdaptorConfig = ImageVqganAdaptorConfig()
# where ['text', 'image_resnet', 'image_vqgan'] are registered under 'ofasys.adaptor'
OFAAdaptorConfig = ConfigStore().make_dataclass(
    "ofasys.adaptor",
    "OFAAdaptorConfig",
    __name__,
    ['text', 'image_resnet', 'image_patch_embed', 'image_vqgan', 'box'],
)
default_adaptor = {
    ModalityType.TEXT: 'text',
    ModalityType.IMAGE: 'image_resnet',
    ModalityType.BOX: 'text',
    ModalityType.AUDIO: 'audio_fbank',
    ModalityType.PHONE: 'text',
    ModalityType.VIDEO: 'video_image_sequence',
    ModalityType.MOTION: 'text',
    ModalityType.STRUCT: 'text',
    ModalityType.CATEGORY: 'text',
}


class OFAGeneralAdaptor(torch.nn.Module):
    def __init__(self, cfg, dictionary, is_src):
        """
        General adaptor will dispatch slot to its adaptor (or default adaptor for its Modality).
        General will init each adaptor (if is_activate).
        Like ** BaseAdaptor **, GeneralAdaptor can work for both IO Adaptors (**forward** for Input Adaptor
        and **forward_output** for Output Adaptor).

        Args:
            cfg : model config.
            dictionary (Dictionary): global vocab.
            is_src (bool): where is the adaptor used for .
        """
        super().__init__()
        # build public network across modality adaptors
        self.embed_tokens = self.build_embedding(cfg, dictionary)
        self.cfg = cfg
        self.is_src = is_src

        self.name2adaptor: Dict[str, BaseAdaptor] = {}
        for config_field in fields(cfg.adaptor):
            if config_field.name.startswith('_'):
                continue
            # TODO: activate adaptors according to encoder and decoder automatically
            if config_field.name == 'image_vqgan' and is_src:
                continue
            if config_field.name == 'image_resnet' and not is_src:
                continue
            if config_field.name == 'video_image_sequence' and not is_src:
                continue
            if config_field.name == 'image_vit' and not is_src:
                continue

            config = getattr(cfg.adaptor, config_field.name)
            # parse adaptor's config from model's config by BaseAdaptorConfig.parse_from_model_cfg
            config.parse_from_model_cfg(cfg)
            if config.is_active is False:
                continue

            self.name2adaptor[config_field.name] = (
                ConfigStore()
                .get("ofasys.adaptor", config_field.name)
                .target(self.embed_tokens, dictionary, is_src, self, config)
            )
            setattr(self, config_field.name, self.name2adaptor[config_field.name])

        # build concat networks
        embed_dim = cfg.encoder_embed_dim if is_src else cfg.decoder_embed_dim
        self.num_attention_heads = cfg.encoder_attention_heads if is_src else cfg.decoder_attention_heads
        self.pos_scaling = float(embed_dim / cfg.encoder_attention_heads * cfg.attn_scale_factor) ** -0.5
        if not self.cfg.entangle_position_embedding:
            self.pos_q_linear = torch.nn.Linear(embed_dim, embed_dim)
            self.pos_k_linear = torch.nn.Linear(embed_dim, embed_dim)

    def get_adaptor(self, slot: Slot) -> BaseAdaptor:
        """
        Get Adaptor for the given Slot. If the Slot is not assigned with a adaptor name in the Instruction,
        we will use the default Adaptor for its modality.

        Args:
            slot (Slot): preprocessed input data.

        Returns:
            BaseAdaptor:
                Adaptor for Slot.
        """
        if slot.get_attr('adaptor'):
            return self.name2adaptor[slot.get_attr('adaptor')]
        else:
            return self.name2adaptor[default_adaptor[slot.modality]]

    def forward(self, slots: List[Slot], **kwargs):
        """
        When work as GeneranlInputAdaptor, GeneralAdaptor will dispatch each slot to its adaptor, then
        gather all AdaptorOutputs and concatenate them to one AdaptorOutput by using ``self.concat()``.
        return a tuple instead of AdaptorOutput as checkpoint_activations need iterable object.

        Args:
            slots: preprocessed input slots.
        Returns:
            tuple:
                concatenated embedding.
        """
        # Apply each adaptor in order of ModalityType, in order to guarantee the
        # numerical consistency with older versions
        modality_outputs = [None for _ in range(len(slots))]
        cnt = 0
        modal_mask = [None for _ in range(len(slots))]
        for mod in ModalityType:
            for i, slot in enumerate(slots):
                if slot.modality == mod:
                    adaptor = self.get_adaptor(slot)
                    modality_outputs[i] = adaptor(slot, **kwargs)
                    #modal ffn
                    if self.cfg.modal_ffn:
                        modal_mask_item = torch.zeros_like(modality_outputs[i].masks, dtype=torch.int64)
                        modal_mask_item = modal_mask_item + int(mod.value) - 1
                        modal_mask[i] = modal_mask_item
                    cnt += 1
            if cnt == len(slots):
                break
        assert cnt == len(slots), cnt
        # modality_outputs = []
        # for slot in slots:
        #     modality_outputs.append(self.mod2adaptor[slot.modality](slot, **kwargs))
        output = self.concat(modality_outputs)
        # return output
        if self.cfg.modal_ffn:
            modal_mask = torch.cat(modal_mask, dim=-1)
        return output.embed, output.masks, output.pos_embed, output.self_attn_bias, modal_mask

    def forward_output(self, x: Tensor, extra: Dict[str, Any], slots: List[Slot], **kwargs):
        """
        When work as GeneralOutputAdaptor, GeneralAdaptor will dispatch hidden states from model
        to the target Output Adaptor ( by calling method ``forward_output()``).

        Note:
            Only one Output Adaptor is supported now, which means we only allow one Slot in
            the target sequence of the Instruction.

        Args:
            x (Tensor): hidden states from model in the shape of
             ``(batch_size, seq_length, embed_dim)``
            extra (Dict[str, Any]): extra model output information.
            slots (List[Slot]):  input preprocessed data.

        Returns:
            tuple:
                - x (Tensor): modality data in Tensor form.
                - extra (Dict[str, Any]): model output with any modality-specific information.

        """
        output_slot = None
        for slot in slots:
            if not slot.is_src:
                assert output_slot is None, 'supports only one target slot'
                output_slot = slot

        assert output_slot
        adaptor = self.get_adaptor(output_slot)
        return adaptor.forward_output(x, extra, slot=output_slot)

    _embed_tokens = None

    def build_embedding(self, cfg, dictionary):
        """

        Args:
            cfg: model config.
            dictionary (Dictionary): global vocab.

        Returns:
            Embedding:
                global embedding matrix.
        """

        if OFAGeneralAdaptor._embed_tokens is not None:
            return OFAGeneralAdaptor._embed_tokens

        assert cfg.share_all_embeddings
        assert cfg.encoder_embed_dim == cfg.decoder_embed_dim
        assert not cfg.decoder_embed_path or cfg.decoder_embed_path == cfg.encoder_embed_path
        assert cfg.freeze_encoder_embedding == cfg.freeze_decoder_embedding
        assert cfg.max_source_positions == cfg.max_target_positions

        embed_tokens = Embedding(
            num_embeddings=len(dictionary), embedding_dim=cfg.encoder_embed_dim, padding_idx=dictionary.pad()
        )
        cfg.share_decoder_input_output_embed = True
        if cfg.freeze_encoder_embedding:
            embed_tokens.weight.requires_grad = False
        OFAGeneralAdaptor._embed_tokens = embed_tokens
        return embed_tokens

    def build_abs_pos_bias(self, pos_embed):
        batch_size, seq_length = pos_embed.size(0), pos_embed.size(1)
        if not self.cfg.entangle_position_embedding:
            pos_q = (
                self.pos_q_linear(pos_embed).view(batch_size, seq_length, self.num_attention_heads, -1).transpose(1, 2)
                * self.pos_scaling
            )
            pos_k = (
                self.pos_k_linear(pos_embed).view(batch_size, seq_length, self.num_attention_heads, -1).transpose(1, 2)
            )
            abs_pos_bias = torch.matmul(pos_q, pos_k.transpose(2, 3))
        else:
            abs_pos_bias = torch.zeros(
                batch_size,
                self.num_attention_heads,
                seq_length,
                seq_length,
                dtype=pos_embed.dtype,
                device=pos_embed.device,
            )
        return abs_pos_bias

    def concat(self, modality_outputs: List[AdaptorOutput]) -> AdaptorOutput:
        """
        Concatenate all adaptor outputs into a large AdaptorOutput in order.

        Args:
            modality_outputs (List[AdaptorOutput]): AdaptorOutput from different slots.

        Returns:
            AdaptorOutput:
                concatenated AdaptorOuptut, which will be fed into the computation model.
        """
        output = AdaptorOutput(
            torch.cat(tuple(map(lambda x: x.embed, modality_outputs)), dim=1),
            torch.cat(tuple(map(lambda x: x.masks, modality_outputs)), dim=1),
            torch.cat(tuple(map(lambda x: x.pos_embed, modality_outputs)), dim=1),
            None,
        )
        if not self.cfg.use_self_attn_bias:
            return output

        output.self_attn_bias = []
        abs_pos_bias = self.build_abs_pos_bias(output.pos_embed)
        num_layers = self.cfg.encoder.layers if self.is_src else self.cfg.decoder.layers
        num_rel_pos_tables = 1 if self.cfg.share_attn_bias else num_layers

        for idx in range(num_rel_pos_tables):
            self_attn_bias = abs_pos_bias.clone()
            start_pos = 0
            for modality_output in modality_outputs:
                seq_length = modality_output.seq_length
                end_pos = start_pos + seq_length
                if modality_output.self_attn_bias[idx] is not None:
                    self_attn_bias[:, :, start_pos:end_pos, start_pos:end_pos] += modality_output.self_attn_bias[idx]
                start_pos = end_pos
            assert start_pos == output.seq_length
            output.self_attn_bias.append(self_attn_bias)

        return output

    @property
    def embed_tokens(self):
        return self._embed_tokens

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of ofa."""
        #  delete the keys in state_dict which do not use in the adaptors
        for adaptor_name in self.cfg.adaptor.__annotations__:
            if adaptor_name.startswith('_'):
                continue
            if adaptor_name not in self.name2adaptor:
                prefix = name + '.' + adaptor_name
                keys = [key for key in state_dict.keys() if key.startswith(prefix)]
                if keys:
                    if len(keys) > 5:
                        logger.info(
                            f'{adaptor_name} exists in checkpoints but unused, the following keys are deleted (5 out of {len(keys)} are shown): {keys[:5]}'
                        )
                    else:
                        logger.info(
                            f'{adaptor_name} exists in checkpoints but unused, the following keys are deleted: {keys}'
                        )
                    for key in keys:
                        del state_dict[key]

        for adaptor_name in self.name2adaptor:
            self.name2adaptor[adaptor_name].upgrade_state_dict_named(state_dict, "{}.{}".format(name, adaptor_name))
        return state_dict

    def update_sample(self, sample):
        for adaptor_name in self.name2adaptor:
            self.name2adaptor[adaptor_name].update_sample(sample)
        return sample
