# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import base64
import functools
import re
from typing import List, Optional

import numpy as np
import torch


def base64encode(data: bytes) -> str:
    """
    Encode `bytes` data to base64 `str`.

    Args:
        data(bytes): The data's type should be `bytes` for the compatibility
            with multi-modality data.

    Returns:
        str: The output is a base64 `str` that can be used in a text file.
    """
    return bytes.decode(base64.b64encode(data))


# https://regexland.com/base64/
_base64_regex = re.compile(r'^(?:[A-Za-z\d+/]{4})*(?:[A-Za-z\d+/]{3}=|[A-Za-z\d+/]{2}==)?$')


def base64decode(s: str) -> Optional[bytes]:
    """
    Decode base64 `str` to original `bytes`.
    If the input is not a valid base64 string, return None.

    Args:
        s(str): A base64 `str` that can be used in text file.

    Returns:
        Optional[bytes]: The original decoded data with type `bytes`.
            If the input is not a valid base64 string, return None.
    """
    # return base64.b64decode(s)
    s = s.translate(base64._urlsafe_decode_translation)
    if not _base64_regex.fullmatch(s):
        return None
    try:
        return base64.urlsafe_b64decode(s)
    except base64.binascii.Error:
        return None


def group_by_predicator(iterable, predicator):
    """
    group a list by predicator

    example:
        a = [1, 2, 2, 3, 4, 4, 4]
        res = group_by_predicator(a, lambda x, y: x == y)
        print(res)
        # [[1], [2, 2], [3], [4, 4, 4]]
    """

    def group_by(cum, x):
        if len(cum) == 0 or not predicator(cum[-1][-1], x):
            cum.append([x])
        else:
            cum[-1].append(x)
        return cum

    return functools.reduce(group_by, iterable, [])


def collate_tokens(
    values,
    pad_idx,
    eos_idx=None,
    left_pad=False,
    move_eos_to_beginning=False,
    pad_to_length=None,
    pad_to_multiple=1,
    pad_to_bsz=None,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            if eos_idx is None:
                # if no eos_idx is specified, then use the last token in src
                dst[0] = src[-1]
            else:
                dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    if values[0].dim() == 1:
        res = values[0].new(len(values), size).fill_(pad_idx)
    elif values[0].dim() == 2:
        assert move_eos_to_beginning is False
        res = values[0].new(len(values), size, values[0].size(1)).fill_(pad_idx)
    else:
        raise NotImplementedError

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
    return res


def collater_audio(audios, pad_audio, max_sample_size, random_crop=False):
    audio_sizes = [len(s) for s in audios]
    if pad_audio:
        audio_size = min(max(audio_sizes), max_sample_size)
    else:
        audio_size = min(min(audio_sizes), max_sample_size)
    collated_audios = audios[0].new_zeros(len(audios), audio_size)
    padding_mask = (
        torch.BoolTensor(collated_audios.shape).fill_(False)
        # if self.pad_audio else None
    )
    audio_starts = [0 for _ in audios]
    for i, audio in enumerate(audios):
        diff = len(audio) - audio_size
        if diff == 0:
            collated_audios[i] = audio
        elif diff < 0:
            collated_audios[i] = torch.cat([audio, audio.new_full((-diff,), 0.0)])
            padding_mask[i, diff:] = True
        else:
            collated_audios[i], audio_starts[i] = crop_to_max_size(audio, audio_size, random_crop)
            audio_sizes[i] = max_sample_size
    return collated_audios, padding_mask, audio_starts, torch.tensor(audio_sizes)


def crop_to_max_size(wav, target_size, random_crop):
    size = len(wav)
    diff = size - target_size
    if diff <= 0:
        return wav, 0

    start, end = 0, target_size
    if random_crop:
        start = np.random.randint(0, diff + 1)
        end = size - diff + start
    return wav[start:end], start


def collate_others(data: List):
    if len(data) == 0:
        return []

    data_type = type(data[0])
    for d in data:
        if not isinstance(d, data_type):
            data_type = None
            break
    if data_type is None:
        return data

    if isinstance(data[0], torch.Tensor):
        try:
            return torch.stack(data)
        except:  # noqa
            return data
    elif isinstance(data[0], np.ndarray):
        try:
            return np.concatenate(data)
        except:  # noqa
            return data
    elif isinstance(data[0], (int, float, bool, complex)):
        return np.array(data)
    else:
        return data
