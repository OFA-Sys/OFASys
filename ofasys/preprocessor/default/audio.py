# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import copy
import json
import logging
import random
import struct
import sys
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, List, Optional, Tuple, Union
from urllib.request import urlopen

import numpy as np
import torch
import torch.nn.functional as F

from ofasys.utils.logging_utils import master_logging

_no_soundfile_help = (
    "No soundfile found, please install it by `pip install soundfile` if you need to support audio tasks"
)
_no_libsndfile_help = "No libsndfile found, please install it by `sudo apt-get install libsndfile1` in Ubuntu-like system if you need to support audio tasks"

logger = logging.getLogger(__name__)
try:
    import soundfile as sf

    _is_soundfile_missing = 0
except ImportError as _:
    with master_logging():
        logger.info(_no_soundfile_help)
    _is_soundfile_missing = 1
except OSError as _:
    with master_logging():
        logger.info(_no_libsndfile_help)
    _is_soundfile_missing = 2

from ofasys.configure import register_config
from ofasys.module.vocoder import GriffinLimVocoder, HiFiGANVocoder
from ofasys.utils.audio_feature_transforms import *
from ofasys.utils.audio_feature_transforms.data_cfg import S2TDataConfig
from ofasys.utils.audio_utils import TTSMelScale, TTSSpectrogram
from ofasys.utils.file_utils import cached_path
from ofasys.utils.oss import oss_get

from ..instruction import ModalityType, Slot
from ..utils import base64decode, collater_audio
from .base import CollateOutput, PreprocessConfig, SafeBasePreprocess


@dataclass
class AudioEmbedPreprocessConfig(PreprocessConfig):
    audio_feature_dim: int = field(default=439, metadata={"help": "audio feature dim"})
    audio_feature_length: int = field(default=384, metadata={"help": "audio feature length"})


@register_config("ofasys.preprocess", "audio_embed", AudioEmbedPreprocessConfig)
class DefaultAudioEmbedPreprocess(SafeBasePreprocess):
    def __init__(self, global_dict, cfg: AudioEmbedPreprocessConfig):
        super().__init__(global_dict, cfg, ModalityType.AUDIO)
        self.audio_feature_dim = cfg.audio_feature_dim
        self.audio_feature_length = cfg.audio_feature_length

    def map(self, slot: Slot) -> Slot:
        super().map(slot)
        audio = slot.value['data']
        start_index = slot.value['start_index']
        m_len = len(base64decode(audio)) // self.audio_feature_dim // 4
        audio = struct.unpack('>%sf' % (m_len * self.audio_feature_dim), base64decode(audio))
        audio = torch.from_numpy(np.array(audio).reshape(m_len, self.audio_feature_dim))
        audio = audio[start_index : start_index + self.audio_feature_length, :]
        slot.value = audio
        return slot

    def collate(self, slots: List[Slot]) -> CollateOutput:
        super().collate(slots)
        slots[0].value = torch.stack([slot.value for slot in slots], dim=0)
        slot = slots[0]
        return CollateOutput(slot)


def load_waveform(
    wav: Union[str, np.ndarray],
    output_speed: float = 1.0,
    normalization: bool = True,
    mono: bool = True,
    frames: int = -1,
    start: int = 0,
    always_2d: bool = True,
    input_sample_rate: Optional[int] = None,
    output_sample_rate: Optional[int] = None,
    normalize_volume: bool = False,
) -> Tuple[np.ndarray, int]:
    """
    Get the waveform and sample rate of a 16-bit WAV/FLAC/OGG Vorbis audio.

    Args:
        wav (str or BinaryIO): the path or file-like object
        output_speed (int): speed rate
        normalization (bool): normalize values to [-1, 1] (Default: True)
        mono (bool): convert multi-channel audio to mono-channel one
        frames (int): the number of frames to read. (-1 for reading all)
        start (int): Where to start reading. A negative value counts from the end.
        always_2d (bool): always return 2D array even for mono-channel audios
        input_sample_rate (Optional[int]): input sample rate
        output_sample_rate (Optional[int]): output sample rate
        normalize_volume (bool): normalize volume
    Returns:
        waveform (numpy.ndarray): 1D or 2D waveform (channels x length)
        sample_rate (float): sample rate
    """
    if isinstance(wav, str):
        if wav.startswith("http://") or wav.startswith("https://"):
            path_or_fp = BytesIO(urlopen(wav).read())
        elif wav.startswith("oss://"):
            fin = oss_get(wav)
            path_or_fp = BytesIO(fin.read())
            del fin
        else:
            wav_bytes = base64decode(wav)
            if wav_bytes is not None:
                path_or_fp = BytesIO(wav_bytes)
            elif os.path.isfile(wav):
                path_or_fp = wav
            else:
                raise ValueError(f"Incorrect format used for audio.{load_waveform.__doc__}Got {wav}")
        waveform, sample_rate = sf.read(path_or_fp, dtype="float32", always_2d=True, frames=frames, start=start)

    elif isinstance(wav, np.ndarray):
        waveform, sample_rate = wav, input_sample_rate
    else:
        raise ValueError(f"Incorrect format used for image.{load_waveform.__doc__}Got {wav}")

    waveform = waveform.T  # T x C -> C x T
    waveform, sample_rate = convert_waveform(
        waveform,
        sample_rate,
        normalize_volume=normalize_volume,
        to_mono=mono,
        to_speed=output_speed,
        to_sample_rate=output_sample_rate,
    )

    if not normalization:
        waveform *= 2**15  # denormalized to 16-bit signed integers
    if not always_2d:
        waveform = waveform.squeeze(axis=0)

    return waveform, sample_rate


@dataclass
class AudioPreprocessConfig(PreprocessConfig):
    original_sample_rate: Optional[int] = field(default=None, metadata={"help": "waveform original sample rate"})
    target_sample_rate: int = field(default=16000, metadata={"help": "waveform target sample rate"})
    max_seconds: int = field(default=120, metadata={"help": "max duration(seconds) of audio"})
    input_type: str = field(default="wave", metadata={"help": "output type of audio", "choices": ["fbank", "wave"]})
    output_type: str = field(default="fbank", metadata={"help": "output type of audio", "choices": ["fbank", "wave"]})
    vocoder: str = field(default="hifigan", metadata={"help": "vocoder type", "choices": {"griffin_lim", "hifigan"}})
    spec_bwd_max_iter: int = field(default=8, metadata={"help": "spec_bwd_max_iter"})
    speed_augmentation: str = field(default="[1.0]", metadata={"help": "data augmentation that change audio speed"})
    config_yaml: Optional[str] = field(
        default='oss://ofasys/tasks/asr/config.yaml', metadata={"help": "data augmentation for fbank"}
    )
    output_frame_dim: int = field(default=80, metadata={"help": "output_frame_dim"})
    n_frames_per_step: int = field(default=1, metadata={"help": "pack fbank n_frames_per_step"})
    normalize: bool = field(default=True, metadata={"help": "waveform normalization"})
    random_crop: Optional[bool] = field(
        default=True,
        metadata={"help": "always crop from the beginning if false"},
    )
    pad_audio: Optional[bool] = field(
        default=True,
        metadata={"help": "pad audio to the longest one in the batch if true"},
    )
    normalize_volume: Optional[bool] = field(
        default=False,
        metadata={"help": "normalize volume"},
    )
    win_length: Optional[int] = field(
        default=1024,
        metadata={"help": "win length"},
    )
    hop_length: Optional[int] = field(
        default=256,
        metadata={"help": "hop length"},
    )
    n_fft: Optional[int] = field(
        default=1024,
        metadata={"help": "n_fft"},
    )
    f_min: Optional[int] = field(
        default=0,
        metadata={"help": "f_min"},
    )
    f_max: Optional[int] = field(
        default=8000,
        metadata={"help": "f_max"},
    )


@register_config("ofasys.preprocess", "audio", AudioPreprocessConfig)
class DefaultAudioPreprocess(SafeBasePreprocess):
    def __init__(self, global_dict, cfg: AudioPreprocessConfig):
        super().__init__(global_dict, cfg, ModalityType.AUDIO)

        if _is_soundfile_missing:
            logger.error(_no_soundfile_help if _is_soundfile_missing == 1 else _no_libsndfile_help)
            sys.exit(1)

        self.cfg = cfg
        self.global_dict = global_dict
        self.original_sr = cfg.original_sample_rate
        self.target_sr = cfg.target_sample_rate

        self.max_tokens = cfg.max_seconds * cfg.target_sample_rate
        self.input_type = cfg.input_type
        self.output_type = cfg.output_type
        self.speed_augmentation = eval(cfg.speed_augmentation)

        self.data_cfg = None
        self._vocoder = None
        self.train_feature_transforms = None
        self.test_feature_transforms = None
        if self.input_type == "fbank" or self.output_type == "fbank":
            self.output_type = "fbank"
            if cfg.config_yaml:
                local_config_yaml = cached_path(cfg.config_yaml)
                self.data_cfg = S2TDataConfig(Path(local_config_yaml))
                self.train_feature_transforms = CompositeAudioFeatureTransform.from_config_dict(
                    self.data_cfg.get_feature_transforms("_train", True)
                )
                self.test_feature_transforms = CompositeAudioFeatureTransform.from_config_dict(
                    self.data_cfg.get_feature_transforms("_eval", False)
                )

        self.n_frames_per_step = cfg.n_frames_per_step
        self.output_frame_dim = cfg.output_frame_dim

        self.normalize = cfg.normalize
        self.normalize_volume = cfg.normalize_volume

        self.pad_audio = cfg.pad_audio
        self.random_crop = cfg.random_crop

    @property
    def vocoder(self):
        if self._vocoder is None and self.cfg.config_yaml is not None:
            data_cfg = S2TDataConfig(Path(cached_path(self.cfg.config_yaml)))
            self._vocoder = build_vocoder(self.cfg, data_cfg)
        return self._vocoder

    @vocoder.setter
    def vocoder(self, _vocoder):
        self._vocoder = _vocoder

    def dummy_slot(self, slot):
        slot.value = {
            "fbank": torch.empty(0, 0, dtype=torch.float32),
            "fbank_lengths": torch.tensor([0], dtype=torch.long),
        }
        return slot

    def map(self, slot: Slot) -> Slot:

        super().map(slot)
        if not slot.is_src and slot.value is None:
            return self.dummy_slot(slot)

        if self.input_type == "wave":

            wav = slot.value
            split = slot.split

            if slot.split == "train":
                speed = random.choice(self.speed_augmentation)
            else:
                speed = 1.0
            if self.output_type == "fbank":
                wav, sr = load_waveform(
                    wav=wav,
                    output_speed=speed,
                    normalization=False,
                    mono=True,
                    frames=-1,
                    start=0,
                    always_2d=True,
                    input_sample_rate=self.original_sr,
                    output_sample_rate=self.target_sr,
                    normalize_volume=self.normalize_volume,
                )

                if slot.is_src:
                    fbank = _get_kaldi_fbank(wav, sr, self.output_frame_dim)
                    if fbank is None:
                        fbank = _get_torchaudio_fbank(wav, sr, self.output_frame_dim)
                    if fbank is None:
                        raise ImportError("Please install pyKaldi or torchaudio to enable fbank feature extraction")

                    n_frames_per_step = slot.get_attr("n_frames_per_step")
                    fbank = self.prepare_fbank(fbank, split, n_frames_per_step=n_frames_per_step)

                    slot.value = {'fbank': fbank, 'fbank_lengths': fbank.shape[0]}
                else:
                    wav /= 2**15
                    wav = torch.tensor(wav)
                    fbank = extract_logmel_spectrogram(
                        wav,
                        sr,
                        None,
                        win_length=self.cfg.win_length,
                        hop_length=self.cfg.hop_length,
                        n_fft=self.cfg.n_fft,
                        n_mels=self.cfg.output_frame_dim,
                        f_min=self.cfg.f_min,
                        f_max=self.cfg.f_max,
                        target_length=None,
                    )

                    n_frames_per_step = slot.get_attr("n_frames_per_step")
                    fbank = self.prepare_fbank(
                        fbank.cpu().detach().numpy(), split, n_frames_per_step=n_frames_per_step
                    )
                    slot.value = {'fbank': fbank, 'fbank_lengths': fbank.shape[0]}

            else:
                wav, sr = load_waveform(
                    wav=wav,
                    output_speed=speed,
                    normalization=True,
                    mono=True,
                    frames=-1,
                    start=0,
                    always_2d=True,
                    input_sample_rate=self.original_sr,
                    output_sample_rate=self.target_sr,
                    normalize_volume=self.normalize_volume,
                )
                wav = wav.mean(-1)
                assert wav.ndim == 1, wav.ndim

                wav = torch.from_numpy(wav)
                wav = self.maybe_normalize_waveform(wav)

                slot.value = {'wav': wav, 'wav_lengths': wav.shape[0]}
        else:
            assert self.input_type == "fbank"
            self.output_type = "fbank"

            fbank = slot.value
            split = slot.split

            if fbank.startswith("oss://"):
                fin = oss_get(fbank)
                fbank = BytesIO(fin.read()).read()
                del fin

            fbank = np.frombuffer(base64decode(fbank), np.float32).reshape([-1, self.data_cfg.input_feat_per_channel])

            n_frames_per_step = slot.get_attr("n_frames_per_step")
            fbank = self.prepare_fbank(fbank, split, n_frames_per_step=n_frames_per_step)

            slot.value = {'fbank': fbank, 'fbank_lengths': fbank.shape[0]}

        return slot

    def prepare_fbank(self, fbank, split="eval", n_frames_per_step=None):
        if split == "train":
            if self.train_feature_transforms is not None:
                fbank = self.train_feature_transforms(fbank)
        else:
            if self.test_feature_transforms is not None:
                fbank = self.test_feature_transforms(fbank)
        fbank = torch.from_numpy(fbank).float()
        fbank = self.pack_frames(fbank, n_frames_per_step=n_frames_per_step)
        return fbank

    def decode(self, feature: torch.Tensor):
        """
        Convert frequency domain features to time domain features,
        i.e., convert fbank features to waveform.
        This function aims to single input.
        """
        waveform = self.vocoder(feature).squeeze(0)
        return waveform

    def postprocess(self, outputs, **sample):
        for single_output in outputs:
            single_output.waveform = self.decode(single_output.feature)
            if single_output.targ_feature is not None:
                single_output.targ_waveform = self.decode(single_output.targ_feature)
        return outputs

    def collate(self, slots: List[Slot]) -> CollateOutput:
        """
        Inputs:
            samples: List of Tensors after preprocess

        Returns:
            dict:
                src_tokens (Tensor): batched tokens with shape `[batch, seq_len]`
        """
        super().collate(slots)

        def _collate_frames(frames: List[torch.Tensor]):
            """
            Convert a list of 2D frames into a padded 3D tensor

            Args:
                frames (list): list of 2D frames of size L[i]*f_dim. Where L[i] is
                    length of i-th frame and f_dim is static dimension of features
            Returns:
                3D tensor of size len(frames)*len_max*f_dim where len_max is max of L[i]
            """
            max_len = max(frame.size(0) for frame in frames)
            out = frames[0].new_zeros((len(frames), max_len, frames[0].size(1)))
            for i, v in enumerate(frames):
                out[i, : v.size(0)] = v
            return out

        if slots[0].is_src:
            if self.output_type == "fbank":
                slots[0].value['fbank'] = _collate_frames([slot.value['fbank'] for slot in slots])
                slots[0].value['fbank_lengths'] = torch.tensor(
                    [slot.value['fbank_lengths'] for slot in slots], dtype=torch.long
                )
            else:
                wav_data, wav_padding_mask, wav_starts, wav_lengths = collater_audio(
                    [slot.value['wav'] for slot in slots], self.pad_audio, self.max_tokens, self.random_crop
                )
                slots[0].value['wav'] = wav_data
                slots[0].value['wav_padding_mask'] = wav_padding_mask
                slots[0].value['audio_starts'] = wav_starts
                slots[0].value['wav_lengths'] = wav_lengths

            return CollateOutput(slots[0])

        else:
            if self.output_type == "fbank":
                input_slot, target_slot = copy.deepcopy(slots[0]), copy.deepcopy(slots[0])
                feat = _collate_frames([slot.value['fbank'] for slot in slots])
                bsz, _, d = feat.size()
                prev_outputs = torch.cat((feat.new_zeros((bsz, 1, d)), feat[:, :-1, :]), dim=1)
                target_length = torch.tensor([slot.value['fbank_lengths'] for slot in slots], dtype=torch.long)

                input_slot.value["fbank"] = prev_outputs
                input_slot.value["fbank_lengths"] = target_length
                target_slot.value["fbank"] = feat
                target_slot.value["fbank_lengths"] = target_length
                ntokens = target_length.sum().item()
                extra_dict = {
                    "target": target_slot.value["fbank"],
                    "target_lengths": target_slot.value["fbank_lengths"],
                    "ntokens": ntokens,
                    "type": self.output_type,
                    "dict_start": self.global_dict.index("<phone>_dict_begin") + 1,
                    "dict_end": self.global_dict.index("<phone>_dict_end"),
                    "blank_id": self.global_dict.index("<phone>_dict_begin"),
                }

                return CollateOutput(input_slot, target_slot, extra_dict)

    def pack_frames(self, feature: torch.Tensor, n_frames_per_step=None):
        if n_frames_per_step is None:
            n_frames_per_step = self.n_frames_per_step
        else:
            n_frames_per_step = int(n_frames_per_step)
        if n_frames_per_step == 1:
            return feature
        n_packed_frames = feature.shape[0] // n_frames_per_step
        feature = feature[: n_frames_per_step * n_packed_frames]
        return feature.reshape(n_packed_frames, -1)

    def maybe_normalize_waveform(self, wav):
        if self.normalize:
            with torch.no_grad():
                wav = F.layer_norm(wav, wav.shape)
        return wav


def _get_kaldi_fbank(waveform: np.ndarray, sample_rate: int, n_bins=80) -> Optional[np.ndarray]:
    """Get mel-filter bank features via PyKaldi."""
    try:
        from kaldi.feat.fbank import Fbank, FbankOptions
        from kaldi.feat.mel import MelBanksOptions
        from kaldi.feat.window import FrameExtractionOptions
        from kaldi.matrix import Vector

        mel_opts = MelBanksOptions()
        mel_opts.num_bins = n_bins
        frame_opts = FrameExtractionOptions()
        frame_opts.samp_freq = sample_rate
        opts = FbankOptions()
        opts.mel_opts = mel_opts
        opts.frame_opts = frame_opts
        fbank = Fbank(opts=opts)
        features = fbank.compute(Vector(waveform.squeeze()), 1.0).numpy()
        return features
    except ImportError:
        return None


def _get_torchaudio_fbank(waveform: np.ndarray, sample_rate, n_bins=80) -> Optional[np.ndarray]:
    """Get mel-filter bank features via TorchAudio."""
    try:
        import torchaudio.compliance.kaldi as ta_kaldi

        waveform = torch.from_numpy(waveform)
        features = ta_kaldi.fbank(waveform, num_mel_bins=n_bins, sample_frequency=sample_rate)
        return features.numpy()
    except ImportError:
        return None


def trim_or_pad_to_target_length(data_1d_or_2d: np.ndarray, target_length: int) -> np.ndarray:
    assert len(data_1d_or_2d.shape) in {1, 2}
    delta = data_1d_or_2d.shape[0] - target_length
    if delta >= 0:  # trim if being longer
        data_1d_or_2d = data_1d_or_2d[:target_length]
    else:  # pad if being shorter
        if len(data_1d_or_2d.shape) == 1:
            data_1d_or_2d = np.concatenate([data_1d_or_2d, np.zeros(-delta)], axis=0)
        else:
            data_1d_or_2d = np.concatenate([data_1d_or_2d, np.zeros((-delta, data_1d_or_2d.shape[1]))], axis=0)
    return data_1d_or_2d


def extract_logmel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int,
    output_path: Optional[Path] = None,
    win_length: int = 1024,
    hop_length: int = 256,
    n_fft: int = 1024,
    win_fn: callable = torch.hann_window,
    n_mels: int = 80,
    f_min: float = 0.0,
    f_max: float = 8000,
    eps: float = 1e-5,
    overwrite: bool = False,
    target_length: Optional[int] = None,
):
    if output_path is not None and output_path.is_file() and not overwrite:
        return

    spectrogram_transform = TTSSpectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length, window_fn=win_fn)
    mel_scale_transform = TTSMelScale(
        n_mels=n_mels, sample_rate=sample_rate, f_min=f_min, f_max=f_max, n_stft=n_fft // 2 + 1
    )
    spectrogram = spectrogram_transform(waveform)
    mel_spec = mel_scale_transform(spectrogram)
    logmel_spec = torch.clamp(mel_spec, min=eps).log()
    assert len(logmel_spec.shape) == 3 and logmel_spec.shape[0] == 1
    logmel_spec = logmel_spec.squeeze().t()  # D x T -> T x D
    if target_length is not None:
        logmel_spec = trim_or_pad_to_target_length(logmel_spec, target_length)

    if output_path is not None:
        np.save(output_path.as_posix(), logmel_spec)
    else:
        return logmel_spec


def convert_waveform(
    waveform: Union[np.ndarray, torch.Tensor],
    sample_rate: int,
    normalize_volume: bool = False,
    to_mono: bool = False,
    to_speed: Optional[float] = None,
    to_sample_rate: Optional[int] = None,
) -> Tuple[Union[np.ndarray, torch.Tensor], int]:
    """convert a waveform:
    - to a target sample rate
    - from multi-channel to mono channel
    - volume normalization

    Args:
        waveform (numpy.ndarray or torch.Tensor): 2D original waveform
            (channels x length)
        sample_rate (int): original sample rate
        normalize_volume (bool): perform volume normalization
        to_mono (bool): convert to mono channel if having multiple channels
        to_sample_rate (Optional[int]): target sample rate
    Returns:
        waveform (numpy.ndarray): converted 2D waveform (channels x length)
        sample_rate (float): target sample rate
    """
    try:
        import torchaudio.sox_effects as ta_sox
    except ImportError:
        raise ImportError("Please install torchaudio: pip install torchaudio")

    effects = []
    if normalize_volume:
        effects.append(["gain", "-n"])
    if to_mono and waveform.shape[0] > 1:
        effects.append(["channels", "1"])
    if to_speed is not None:
        effects.append(["speed", f"{to_speed}"])
    if to_sample_rate is not None and to_sample_rate != sample_rate:
        effects.append(["rate", f"{to_sample_rate}"])
    else:
        effects.append(['rate', f"{sample_rate}"])

    if len(effects) > 0:
        is_np_input = isinstance(waveform, np.ndarray)
        _waveform = torch.from_numpy(waveform) if is_np_input else waveform
        converted, converted_sample_rate = ta_sox.apply_effects_tensor(_waveform, sample_rate, effects)
        if is_np_input:
            converted = converted.numpy()
        return converted, converted_sample_rate
    return waveform, sample_rate


def build_vocoder(args, data_cfg: S2TDataConfig):
    if args.vocoder == "griffin_lim":
        return GriffinLimVocoder.from_data_cfg(args, data_cfg)

    elif args.vocoder == "hifigan":
        vocoder_cfg = data_cfg.vocoder
        assert vocoder_cfg.get("type", "griffin_lim") == "hifigan"
        local_config = cached_path(vocoder_cfg["config"])
        with open(Path(local_config)) as f:
            model_cfg = json.load(f)
        local_checkpoint = cached_path(vocoder_cfg["checkpoint"])
        return HiFiGANVocoder(Path(local_checkpoint), model_cfg, fp16=False)
    # elif args.vocoder == "code_hifigan":
    #     vocoder_cfg = data_cfg.vocoder
    #     assert vocoder_cfg is not None, "vocoder not specified in the data config"
    #     local_config = cached_path(vocoder_cfg["config"])
    #     with open(Path(local_config)) as f:
    #         model_cfg = json.load(f)
    #     local_checkpoint = cached_path(vocoder_cfg["checkpoint"])
    #     return CodeHiFiGANVocoder(Path(local_checkpoint), model_cfg, fp16=args.fp16)
    else:
        raise ValueError("Unknown vocoder")
