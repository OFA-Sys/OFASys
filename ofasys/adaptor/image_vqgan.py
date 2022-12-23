# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from dataclasses import dataclass, field
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ofasys.adaptor.base import AdaptorOutput, BaseAdaptor, BaseAdaptorConfig, Slot
from ofasys.configure import register_config
from ofasys.module import Embedding
from ofasys.preprocessor import Dictionary
from ofasys.preprocessor.tokenizer.vqgan import VQGANTokenizer


def make_vqgan_code_bucket_position(bucket_size, num_relative_distance):
    coords_h = torch.arange(bucket_size)
    coords_w = torch.arange(bucket_size)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += bucket_size - 1  # shift to start from 0
    relative_coords[:, :, 1] += bucket_size - 1
    relative_coords[:, :, 0] *= 2 * bucket_size - 1
    relative_position_index = torch.zeros(size=(bucket_size * bucket_size + 1,) * 2, dtype=relative_coords.dtype)
    relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
    relative_position_index[0, 0:] = num_relative_distance - 3
    relative_position_index[0:, 0] = num_relative_distance - 2
    relative_position_index[0, 0] = num_relative_distance - 1
    return relative_position_index


@dataclass
class ImageVqganAdaptorConfig(BaseAdaptorConfig):
    code_image_size: int = field(
        default=256,
        metadata={"help": "code image size"},
    )
    code_bucket_size: int = field(
        default=42,
        metadata={"help": "image bucket size"},
    )
    vqgan_factor: int = field(default=8, metadata={"help": "vqgan factor"})
    vqgan_model_path: str = field(
        default="oss://ofasys/tasks/image_gen/vqgan/last.ckpt",
        metadata={"help": "path of vqgan model"},
    )
    vqgan_config_path: str = field(
        default="oss://ofasys/tasks/image_gen/vqgan/model.yaml",
        metadata={"help": "path of vqgan config"},
    )
    use_encode: bool = field(default=True, metadata={"help": "where to use tokenizer.encode in map"})
    code_entry_prefix: str = field(default='code', metadata={"help": "prefix of code entry in the global_dict"})


@register_config("ofasys.adaptor", "image_vqgan", ImageVqganAdaptorConfig)
class ImageVqganAdaptor(BaseAdaptor):
    def __init__(
        self,
        embed_tokens: Embedding,
        dictionary: Dictionary,
        is_src: bool,
        general_adaptor,
        cfg: ImageVqganAdaptorConfig,
    ):
        super().__init__(embed_tokens, dictionary, is_src, general_adaptor, cfg)

        self.window_size = cfg.code_image_size // cfg.vqgan_factor

        self.embed_code_positions = Embedding(cfg.code_bucket_size**2 + 1, cfg.embed_dim)

        code_num_rel_dis = (2 * cfg.code_bucket_size - 1) * (2 * cfg.code_bucket_size - 1) + 3
        code_rp_bucket = make_vqgan_code_bucket_position(cfg.code_bucket_size, code_num_rel_dis)
        code_position_idx = (
            torch.arange(self.window_size).unsqueeze(0).expand(self.window_size, self.window_size)
            + torch.arange(self.window_size).unsqueeze(1) * cfg.code_bucket_size
            + 1
        )
        code_position_idx = torch.cat([torch.tensor([0]), code_position_idx.view(-1)])
        code_position_idx = torch.cat([code_position_idx, torch.tensor([1024] * 768)])

        num_rel_pos_tables = 1 if self.cfg.share_attn_bias else self.num_layers
        self.code_rel_pos_table_list = nn.ModuleList(
            [Embedding(code_num_rel_dis, cfg.num_attention_heads, zero_init=True) for _ in range(num_rel_pos_tables)]
        )
        self.tokenizer = VQGANTokenizer(
            vqgan_config_path=cfg.vqgan_config_path,
            vqgan_model_path=cfg.vqgan_model_path,
            code_image_size=cfg.code_image_size,
            vqgan_factor=cfg.vqgan_factor,
        )
        # TODO: change this when merge split_adaptor branch
        # self.code_index_start = self.dictionary.index("<{}_0>".format(cfg.code_entry_prefix))
        self.code_index_start = self.dictionary.index("<code>_0")
        self.register_buffer("code_rp_bucket", code_rp_bucket)
        self.register_buffer("code_position_idx", code_position_idx)

    def get_rel_pos_bias(self, batch_size, seq_length, idx, **kwargs):
        code_position_idx = self.code_position_idx[:seq_length]
        rp_bucket = self.code_rp_bucket[code_position_idx][:, code_position_idx]
        values = F.embedding(rp_bucket, self.code_rel_pos_table_list[idx].weight)
        return values

    def update_sample(self, sample: Dict):
        """
        preprocess sample on gpu.

        Args:
            sample (Dict): preprocessed data named dict

        Returns:
            Dict:
                sample: add vqgan encoded images to slot.value
        """

        if self.cfg.use_encode and sample.get('target', None) is None:
            for i, slot in enumerate(sample['net_input']['slots']):
                if self.check_adaptor_slot(slot):
                    image_tensor = slot.value
                    codes = self.tokenizer.encode(image_tensor.float()) + self.code_index_start
                    batch_size = codes.size()[0]
                    codes = torch.cat([codes.new_ones((batch_size, 1)) * 0, codes], dim=-1)
                    codes = torch.cat([codes, codes.new_ones((batch_size, 1)) * 2], dim=-1)
                    sample['net_input']['slots'][i].value = codes[:, :-1].contiguous()
                    sample['target'] = codes[:, 1:].contiguous()
                    sample['ntokens'] = sample['target'].ne(1).long().sum().item()

        return sample

    def forward(self, slot: Slot, **kwargs) -> AdaptorOutput:
        """
        Args:
            slot (Slot): ModalityType.IMAGE

        Returns:
            AdaptorOutput:
                - **embed** (Tensor): the processed embedding for OFA of
                  shape ``(src_len, batch, embed_dim)``
                - **padding_masks** (ByteTensor): the positions of
                  padding elements of shape ``(batch, src_len)``
                - **pos_embedding** (Tensor): the position embeddings
                  of shape ``(batch, src_len, embed_dim)``
        """
        src_tokens = slot.value
        bsz, tgt_len = src_tokens.shape
        padding_mask = src_tokens.eq(self.dictionary.pad())

        code_position_idx = self.code_position_idx[: src_tokens.size(1)]
        code_position_idx = code_position_idx.unsqueeze(0).expand(bsz, tgt_len)
        pos_embed = self.embed_code_positions(code_position_idx)

        token_embedding = self.embed_tokens(src_tokens)

        return AdaptorOutput(token_embedding, padding_mask, pos_embed, [])

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of ofa."""

        prefix = name + "." if name != "" else ""
        code_params = ["code_position_idx"]
        for code_param in code_params:
            state_dict[prefix + code_param] = self.state_dict()[code_param]
        # extend positions using rand_init if necessary
        if len(state_dict[prefix + "embed_code_positions.weight"]) < len(
            self.state_dict()["embed_code_positions.weight"]
        ):
            num_posids_to_add = len(self.state_dict()["embed_code_positions.weight"]) - len(
                state_dict[prefix + "embed_code_positions.weight"]
            )
            embed_dim = state_dict[prefix + "embed_code_positions.weight"].size(1)
            new_pos_embed_to_add = torch.zeros(num_posids_to_add, embed_dim)
            nn.init.normal_(new_pos_embed_to_add, mean=0, std=embed_dim**-0.5)
            new_pos_embed_to_add = new_pos_embed_to_add.to(
                dtype=state_dict[prefix + "embed_code_positions.weight"].dtype,
            )
            state_dict[prefix + "embed_code_positions.weight"] = torch.cat(
                [
                    state_dict[prefix + "embed_code_positions.weight"],
                    new_pos_embed_to_add,
                ]
            )

        return state_dict

    def forward_output(self, x: Tensor, extra: Dict[str, Any], slot: Slot, **kwargs):
        """
        Args:
            x (Tensor): hidden states from model in the shape of
             ``(batch_size, seq_length, embed_dim)``
            extra (Dict[str, Any]): extra model output information.
            slot (Slot):  input preprocessed data.

        Returns:
            tuple:
                - x (Tensor): Tensor of shape ``(batch_size, seq_length, vocab_size)``.
                - extra (Dict[str, Any]): model output with any modality-specific information.
        """
        return self.embed_tokens_T(x), extra
