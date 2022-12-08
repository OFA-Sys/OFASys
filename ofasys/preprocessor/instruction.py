# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import copy
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ofasys import ModalityType
from ofasys.module.utils import apply_to_sample

_instruction_help_doc = """
The instruction's template should format as "... [MODE] ... -> ... [MODE] ...",
where MODE should be one of {} and "..." could contains more [MODE].

For example, the instruction of image caption could be:
    Instruction("[TEXT] [IMAGE] -> [TEXT]").format(
        "What does the image describe?", patch_image, caption_label)
or
    Instruction("What does the image [IMAGE] describe? -> [TEXT]").format(
        patch_image, caption_label)
""".format(
    ', '.join([v.name for v in ModalityType])
)


@dataclass
class Slot:
    """
    Slot is the core concept of the multi-modal abstraction in OFASys.
    Each slot contains only one modality data that spans consecutive positions.
    A Slot is described by modality type, reference Name as well as several arguments for training or inference, marked as attr.
    Given different positions appeared in the instruction, we denote the slot appears in the encoder and decoder sentence by E-slot and D-slot, respectively.
    """

    modality: ModalityType
    is_src: bool
    value: Optional[Any]

    global_position: Optional[int] = None
    column_name: Optional[str] = None
    attributes: Optional[List[str]] = None

    preprocess: Optional[str] = None
    is_plaintext: bool = False
    split: str = 'train'
    decoder_plain_with_loss: bool = False

    # custom memory pinning method on custom type
    def pin_memory(self):
        def _pin_memory(x):
            return x.pin_memory()

        self.value = apply_to_sample(_pin_memory, self.value)
        return self

    def __post_init__(self):
        if self.column_name is None:
            self.column_name = str(self.global_position)
        if self.attributes is not None and isinstance(self.attributes, str):
            self.attributes = self.attributes.split(',')

    def has_attr(self, attr_key: str) -> bool:
        if self.attributes is None:
            return False
        for attribute in self.attributes:
            if attr_key == attribute or attribute.startswith(attr_key + '='):
                return True
        return False

    def get_attr(self, attr_key: str, class_factory: type = None) -> Optional[Any]:
        if self.attributes is None:
            return None
        for attr in self.attributes:
            if attr.startswith(attr_key + '='):
                val = attr[len(attr_key) + 1 :]
                if class_factory is not None:
                    return class_factory(val)
                else:
                    return val
        return None

    def attr2kwargs(self):
        if self.attributes is None:
            return {}
        kwargs = {}
        for attr in self.attributes:
            try:
                k, v = attr.split('=')
            except ValueError:
                k, v = attr, True
            kwargs[k] = v
        return kwargs

    @staticmethod
    def get_target_slot_from_slots(slots: List):
        target_slots = [slot for slot in slots if not slot.is_src]
        return target_slots[-1]

    @staticmethod
    def get_target_slot_from_sample(sample: Dict):
        slots = sample['net_input']['slots']
        target_slots = [slot for slot in slots if not slot.is_src]
        return target_slots[-1]


mod_regex = (
    r'\[(' + '|'.join([v.name for v in ModalityType]) + ')' '(?::([_A-Za-z0-9]+))?' '(?:,([_A-Za-z0-9,.=]+))?' + r'\]'
)
mod_regex = re.compile(mod_regex)


class Instruction:
    """
    The instruction's template should format as "... [MODE] ... -> ... [MODE] ...",
    where MODE should be one of ModalityType and "..." could contains more [MODE].

    For example, the instruction of image caption could be:

        - Illustration 1. Image Captioning::

            [IMAGE:img] what does the image describe? -> [TEXT:cap]

        - Illustration 2. MNLI Task in Glue Benchmark::

            can text1 [TEXT:sent1] imply text2 [TEXT:sent2]? -> [TEXT:label,closed_set]

            # Or we can use the prompt tuning which prepends some text prompts to decoder.
            can text1 [TEXT:sent1] imply text2 [TEXT:sent2]? ->  can text1 [TEXT:sent1,no_loss] imply text2 [TEXT:sent2,no_loss]? [TEXT:label,closed_set]

        - Illustration 3. Object Detection Task with variable-length output ::

            [IMAGE:img] detect the objects in the image. -> [[BOUNDING_BOX] [TEXT]]*

        - Illustration 4. Interleaved Image Text context with variable-length pairs::

            -> ([IMAGE] [TEXT])*


    """

    def __init__(
        self,
        template: str,
        split: str = 'train',
        decoder_plain_with_loss: bool = False,
    ):
        """
        Args:
            template: instruction template string.
            split: data split: train, valid, or test.
            decoder_plain_with_loss: whether compute loss (for decoder)
        """
        # template check
        template = template.strip()
        if template.count('->') != 1:
            raise ValueError(_instruction_help_doc)
        source, target = tuple(map(lambda x: x.strip(), template.split('->')))
        # if len(source) == 0 or len(target) == 0:
        #     raise ValueError("The source or target of instruction can not be empty.")

        # parse slots
        self.template = template
        self.split = split
        self.decoder_plain_with_loss = decoder_plain_with_loss
        self.slots: List[Slot] = []
        self._parse_slot(source, True)
        self._parse_slot(target, False)
        self.others = {}

    def __str__(self):
        s = ""
        last_is_source = True
        for slot in self.slots:
            if last_is_source and not slot.is_src:
                s = s + "-> "
                last_is_source = False
            s = s + str(slot.value) + " "
        return s.strip()

    def get_slot_names(self) -> List[str]:
        return [slot.column_name for slot in self.slots if slot.value is None]

    def format(self, *args, **kwargs):
        """
        Fill template with input data. The formatted instruction can be used for model inference.

        Usage:
            >>> model = OFASys.from_pretrain('OFASys.ckpt')
            >>> sample = Instruction(
            ...     "[IMAGE] what does the region describe in the image? region: [BOUNDING_BOX] -> [TEXT]"
            ... ).format(
            ...     image_data, box_data
            ... )
            >>> text = model.inference(sample)
        """
        ist = copy.deepcopy(self)

        available_slots = sum([not x.is_plaintext for x in ist.slots])
        counter = Counter([x.column_name for x in ist.slots if not x.is_plaintext])
        args = list(args)

        for slot in ist.slots:
            if slot.value is None:
                if len(args) > 0:
                    slot.value = args.pop(0)
                    counter[slot.column_name] -= 1
                    if counter[slot.column_name] != 0:
                        kwargs[slot.column_name] = slot.value
                    # else:
                    #    kwargs.pop(slot.column_name, None)
                else:
                    slot.value = kwargs.get(slot.column_name, None)
                    if slot.value is None and slot.is_src:
                        raise ValueError("Expect filling slot ({}) but missing".format(slot.column_name))
                    counter[slot.column_name] -= 1
                    # if counter[slot.column_name] == 0:
                    #    kwargs.pop(slot.column_name, None)

        if len(args) > 0:
            raise ValueError("Unexpect args ({})".format(args))

        ist.others = kwargs
        return ist

    def _parse_slot(self, template, is_src):
        lst_end = 0
        # TODO: use re.split
        for mat in mod_regex.finditer(template):
            # match regex of modality's slot
            mod, col_name, attr = mat.groups()
            span_start, span_end = mat.span()

            # add the text before current slot
            prefix = template[lst_end:span_start].strip()
            if prefix:
                self.slots.append(
                    Slot(
                        modality=ModalityType.TEXT,
                        is_src=is_src,
                        value=prefix,
                        global_position=len(self.slots),
                        is_plaintext=True,
                        split=self.split,
                        decoder_plain_with_loss=self.decoder_plain_with_loss,
                    )
                )

            # add current modality's slot
            self.slots.append(
                Slot(
                    modality=ModalityType.parse(mod),
                    is_src=is_src,
                    value=None,
                    global_position=len(self.slots),
                    column_name=col_name,
                    attributes=attr,
                    is_plaintext=False,
                    split=self.split,
                    decoder_plain_with_loss=self.decoder_plain_with_loss,
                )
            )
            lst_end = span_end

        suffix = template[lst_end:].strip()
        if suffix:
            self.slots.append(
                Slot(
                    modality=ModalityType.TEXT,
                    is_src=is_src,
                    value=suffix,
                    global_position=len(self.slots),
                    is_plaintext=True,
                    split=self.split,
                    decoder_plain_with_loss=self.decoder_plain_with_loss,
                )
            )


_adaptor_requirements_doc = """
Basic Usage:
                    Adaptor.transform(.)
    modality_input ---------------------> unified_embedding

Modality Input and Output Requirements:
    Notes:
        `Tensor[T](c1, c2)` denotes `np.array(dtype=T, shape=(c1, c2)`
                                 or `torch.Tensor(dtype=T, shape=(c1, c2))`

    1. Text
    Available inputs:
        (after preprocess:)
        `Tensor[int](seq_length)`: A 1-d tensor of tokens after preprocess and tokenizer
        (before preprocess:)
        `str`: A original text before tokenizer

    Default preprocess: ofasys.preprocessor.DefaultTextPreprocess
    Default adaptor: ofasys.adaptor.DefaultTextAdaptor

    Unified outpus:
        `Tensor[float](seq_length, hidden_size)`: Embeddings of Text

    2. Image
    Available inputs:
        (after preprocess:)
        `Tensor[float](C, W, H)`: A 3-d tensor of image after augumentation
        (before preprocess:)
        `str`: A local or HTTP url of image
        `base64 str`: A base64 string of image
        `PIL.Image.Image`: A PIL image object

    Unified outputs:
        `Tensor[float](seq_length, hidden_size)`: Embeddings of Image

    ...
"""
