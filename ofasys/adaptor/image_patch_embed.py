from ofasys.adaptor.base import BaseAdaptor, BaseAdaptorConfig
from ofasys.module.layer import Embedding
from ofasys.preprocessor.dictionary import Dictionary
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .base import AdaptorOutput, BaseAdaptor, Slot, BaseAdaptorConfig
from ofasys import ModalityType
from ofasys.configure import register_config
from ofasys.preprocessor import Dictionary


@dataclass
class ImagePatchEmbedAdaptorConfig(BaseAdaptorConfig):
    image_size_width: int = field(
        default=224, metadata={"help": "Image width"},
    )
    image_size_height: int = field(
        default=224, metadata={"help": "Image height"},
    )
    patch_size_width: int = field(
        default=14, metadata={"help": "Patch width"},
    )
    patch_size_height: int = field(
        default=14, metadata={"help": "Patch height"},
    )
    embed_dim: int = field(
        default=768, metadata={"help": "Embed dim"},
    )
    add_cls_token: bool = field(
        default=True, metadata={"help": "Add [CLS] token"},
    )


@register_config("ofasys.adaptor", "image_patch_embed", ImagePatchEmbedAdaptorConfig)
class ImagePatchEmbedAdaptor(BaseAdaptor):
    def __init__(
        self,
        embed_tokens: Embedding,
        dictionary: Dictionary,
        is_src: bool,
        general_adaptor,
        cfg: ImagePatchEmbedAdaptorConfig,
    ):
        super().__init__(embed_tokens, dictionary, is_src, general_adaptor, cfg)
        image_size = (cfg.image_size_height, cfg.image_size_width)
        patch_size = (cfg.patch_size_height, cfg.patch_size_width)
        num_patches = (image_size[1] // patch_size[1]) * \
            (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_image_positions = Embedding(
            num_patches + 1 if cfg.add_cls_token else num_patches, cfg.embed_dim)
        if cfg.add_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.proj = nn.Conv2d(
            3, cfg.embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, slot: Slot, **kwargs) -> AdaptorOutput:
        assert slot.modality == ModalityType.IMAGE
        image: torch.Tensor = slot.value
        batch_size, _, height, width = image.shape
        assert height == self.image_size[0] and width == self.image_size[1], \
            f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
        image_patch_embed: torch.Tensor = self.proj(
            image).flatten(2).transpose(1, 2)
        if self.cfg.add_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            image_patch_embed = torch.cat(
                (cls_tokens, image_patch_embed), dim=1)

        image_padding_mask: torch.Tensor = torch.zeros(
            (batch_size, image_patch_embed.size(1)), dtype=torch.bool, device=image.device)

        image_pos_embed: torch.Tensor = self.embed_image_positions(
            torch.arange(image_patch_embed.size(1), dtype=torch.long, device=image.device).unsqueeze(0).expand(batch_size, -1))
        return AdaptorOutput(image_patch_embed, image_padding_mask, image_pos_embed, None)
