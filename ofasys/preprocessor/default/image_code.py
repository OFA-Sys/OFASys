# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from ofasys import ModalityType
from ofasys.configure import register_config
from ofasys.utils.file_utils import cached_path

from ..dictionary import Dictionary
from ..instruction import Slot
from ..utils import collate_tokens
from .base import CollateOutput, PreprocessConfig, SafeBasePreprocess
from .image import load_image


@dataclass
class VQGANCodePreprocessConfig(PreprocessConfig):
    code_image_size: int = field(default=256, metadata={"help": "code image size"})
    vqgan_factor: int = field(default=8, metadata={"help": "vqgan factor"})
    code_dict_size: int = field(default=8192, metadata={"help": "code dict size"})
    code_entry_prefix: str = field(default='code', metadata={"help": "prefix of code entry in the global_dict"})
    use_encode: bool = field(default=True, metadata={"help": "where to use tokenizer.encode in map"})
    clip_model: str = field(
        default='oss://ofasys/tasks/image_gen/clip/ViT-B-16.pt', metadata={"help": "model path for a clip reranker"}
    )


def preprocess_vqgan(x):
    x = 2.0 * x - 1.0
    return x


@register_config("ofasys.preprocess", "image_vqgan", VQGANCodePreprocessConfig)
class VQGANCodePreprocess(SafeBasePreprocess):
    def __init__(self, global_dict: Dictionary, cfg: VQGANCodePreprocessConfig):
        super().__init__(global_dict, cfg, modality_type=ModalityType.IMAGE)
        self.num_codes = cfg.code_dict_size
        for i in range(self.num_codes):
            # global_dict.add_symbol("<{}_{}>".format(cfg.code_entry_prefix, i))
            global_dict.add_symbol(f'<code>_{i}')
        # get the start position of code entry in global dict
        self.code_index_start = self.global_dict.index('<code>_0')
        assert self.code_index_start >= 0
        self.code_image_size = cfg.code_image_size
        self.code_resize_transform_pil_image = transforms.Compose(
            [
                lambda image: image.convert("RGB"),
                transforms.Resize((cfg.code_image_size, cfg.code_image_size), interpolation=Image.LANCZOS),
                transforms.ToTensor(),
                preprocess_vqgan,
            ]
        )

        self.code_resize_transform_pytorch_tensor = transforms.Compose(
            [
                transforms.ToPILImage(),
                lambda image: image.convert("RGB"),
                transforms.Resize((cfg.code_image_size, cfg.code_image_size), interpolation=Image.LANCZOS),
                transforms.ToTensor(),
                preprocess_vqgan,
            ]
        )
        if cfg.clip_model:
            import clip

            self.clip = clip
            local_path = cached_path(cfg.clip_model)
            clip_model, clip_preprocess = clip.load(local_path, 'cpu')
            self.clip_model = clip_model
            self.clip_preprocess = clip_preprocess
            self.clip_model.eval()

    def dummy_slot(self, slot):
        slot.value = torch.empty(0, dtype=torch.long)
        return slot

    def map(self, slot: Slot) -> Slot:
        """
        Inputs:
            code: (`str` or `List` or `Tensor`) could be:
                A string separated by single-whitespaces like `6674 4336 4532 5334...` ;
                Tokens of a numpy or torch Tensor after user-defined preprocess

        Returns:
            `Torch.LongTensor`: 1-d int64 torch.Tensor
        """
        super().map(slot)
        if not slot.is_src and slot.value is None:
            return self.dummy_slot(slot)

        if self.cfg.use_encode:
            image = self.preprocess_image(slot.value)
            slot.value = image
            return slot
        else:
            code = slot.value
        if isinstance(code, np.ndarray) and np.issubdtype(code.dtype, np.integer) and code.ndim == 1:
            tokens = torch.LongTensor(code)
        elif (isinstance(code, torch.IntTensor) or isinstance(code, torch.LongTensor)) and code.ndim == 1:
            tokens = code.long()
        elif isinstance(code, str):
            tokens = self.split_str(code)
        else:
            raise ValueError("Incorrect input for code, only support string or 1-d int Tensor, " f"got {type(code)}")

        # TODO: add a parameter to control whether use these preprocess.

        if slot.get_attr('length') is not None:
            length = int(slot.get_attr('length'))
            tokens = tokens[:length]

        # add vocab size
        tokens = tokens + self.code_index_start
        if slot.is_src is False:
            tokens = torch.cat(
                [torch.LongTensor([self.global_dict.bos()]), tokens, torch.LongTensor([self.global_dict.eos()])]
            )
        slot.value = tokens
        return slot

    def split_str(self, tokens_str):
        tokens = [int(num) for num in tokens_str.strip().split()]
        return torch.LongTensor(tokens)

    def preprocess_image(self, image, **kwargs):

        if isinstance(image, np.ndarray) or isinstance(image, torch.Tensor):
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image)
            image_tensor = self.code_resize_transform_pytorch_tensor(image)
            return image_tensor
        else:
            image = load_image(image)
            image_tensor = self.code_resize_transform_pil_image(image)
        return image_tensor

    def collate(self, slots: List[Slot]) -> CollateOutput:
        super().collate(slots)
        if self.cfg.use_encode:
            slots[0].value = torch.stack([slot.value for slot in slots], dim=0)
            slot = slots[0]
            return CollateOutput(slot)
        else:
            if slots[0].is_src:
                slots[0].value = collate_tokens(
                    [slot.value for slot in slots],
                    pad_idx=self.global_dict.pad(),
                    eos_idx=self.global_dict.eos(),
                    pad_to_multiple=self.cfg.pad_to_multiple,
                )
                return CollateOutput(slots[0])
            else:
                input_value = collate_tokens(
                    [slot.value[:-1] for slot in slots],  # skip <EOS>
                    pad_idx=self.global_dict.pad(),
                    eos_idx=self.global_dict.eos(),
                    pad_to_multiple=self.cfg.pad_to_multiple,
                )
                target_value = collate_tokens(
                    [slot.value[1:] for slot in slots],  # skip <BOS>
                    pad_idx=self.global_dict.pad(),
                    eos_idx=self.global_dict.eos(),
                    pad_to_multiple=self.cfg.pad_to_multiple,
                )
                input_slot = Slot(
                    slots[0].modality,
                    slots[0].is_src,
                    input_value,
                    slots[0].global_position,
                    slots[0].column_name,
                    slots[0].attributes,
                )
                target_slot = Slot(
                    slots[0].modality,
                    slots[0].is_src,
                    target_value,
                    slots[0].global_position,
                    slots[0].column_name,
                    slots[0].attributes,
                )

                # for lagecy compatible
                ntokens = target_slot.value.ne(self.global_dict.pad()).long().sum().item()
                extra_dict = {
                    "target": target_slot.value,
                    "ntokens": ntokens,
                }
                return CollateOutput(input_slot, target_slot, extra_dict)

    def decode(self, tokens: torch.LongTensor, **kwargs):
        tokens -= self.code_index_start
        return self.tokenizer.decode(tokens, **kwargs)

    def rerank_with_clip(self, images, text):

        clip_images_input = torch.stack([self.clip_preprocess(hyp_image) for hyp_image in images], dim=0).cpu()
        clip_text_input = self.clip.tokenize([text]).cpu()
        with torch.no_grad():
            hyp_image_features = self.clip_model.encode_image(clip_images_input)
            hyp_image_features /= hyp_image_features.norm(dim=-1, keepdim=True)
            text_features = self.clip_model.encode_text(clip_text_input)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        ti_similarity = hyp_image_features @ text_features.T
        scores, indices = torch.sort(ti_similarity.view(-1), descending=True)
        return indices.tolist()
