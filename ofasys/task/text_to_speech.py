# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import soundfile as sf
except:
    pass

import torch
import torch.nn.functional as F

from ofasys.configure import register_config
from ofasys.generator import AutoRegressiveSpeechGenerator, SpeechGeneratorOutput
from ofasys.task.base import OFATask, TaskConfig
from ofasys.utils.audio_feature_transforms.data_cfg import S2TDataConfig
from ofasys.utils.file_utils import cached_path

logger = logging.getLogger(__name__)


@dataclass
class Text2SpeechTaskConfig(TaskConfig):
    config_yaml: str = field(default=None, metadata={"help": "data augmentation for fbank"})
    n_frames_per_step: int = field(default=1, metadata={"help": "pack fbank n_frames_per_step"})
    eos_prob_threshold: float = field(default=0.5, metadata={"help": "threshold of eos probability"})
    eval_tts: bool = field(default=False, metadata={"help": "whether to eval inference waveform"})
    fp16: bool = field(default=True, metadata={"help": "use fp16"})
    cpu: bool = field(default=False, metadata={"help": "use cpu"})


@register_config("ofasys.task", "text_to_speech", dataclass=Text2SpeechTaskConfig)
class Text2SpeechTask(OFATask):
    def __init__(self, cfg: Text2SpeechTaskConfig, **kwargs):
        super().__init__(cfg, **kwargs)
        local_config_yaml = cached_path(cfg.config_yaml)
        self.data_cfg = S2TDataConfig(Path(local_config_yaml))

    def initialize(self, global_dict, **kwargs):
        super().initialize(global_dict, **kwargs)
        if self.cfg.eval_tts:
            audio_preprocessor = self.general_preprocess.name2pre["audio"]
            if torch.cuda.is_available() and not self.cfg.cpu:
                audio_preprocessor.vocoder = audio_preprocessor.vocoder.cuda()
            if self.cfg.fp16:
                audio_preprocessor.vocoder = audio_preprocessor.vocoder.half()

    def build_speech_generator(self, **gen_kwargs):
        stats_npz_path = self.data_cfg.config.get("ofa_global_cmvn", {}).get("stats_npz_path", None)
        return AutoRegressiveSpeechGenerator(
            self.source_dictionary,
            stats_npz_path=stats_npz_path,
            max_iter=self.cfg.max_target_positions,
            eos_prob_threshold=self.cfg.eos_prob_threshold,
        )

    def inference(self, model, sample, **kwargs):
        outputs = super().inference(model, sample, **kwargs)

        if self.cfg.evaluation.output_dir:
            os.makedirs(self.cfg.evaluation.output_dir, exist_ok=True)
            wav_dir = os.path.join(self.cfg.evaluation.output_dir, f"{self.data_cfg.sample_rate}hz")
            os.makedirs(wav_dir, exist_ok=True)
            wav_tgt_dir = os.path.join(self.cfg.evaluation.output_dir, f"{self.data_cfg.sample_rate}hz_tgt")
            os.makedirs(wav_tgt_dir, exist_ok=True)
            feat_dir = os.path.join(self.cfg.evaluation.output_dir, f"{self.data_cfg.sample_rate}hz_fbank")
            os.makedirs(feat_dir, exist_ok=True)
            feat_tgt_dir = os.path.join(self.cfg.evaluation.output_dir, f"{self.data_cfg.sample_rate}hz_fbank_tgt")
            os.makedirs(feat_tgt_dir, exist_ok=True)

            single_output: SpeechGeneratorOutput
            for sample_id, single_output in zip(sample['id'], outputs):
                single_output.save_audio(
                    os.path.join(wav_dir, f"{sample_id}.wav"), sample_rate=self.data_cfg.sample_rate
                )
                single_output.save_audio(
                    os.path.join(wav_tgt_dir, f"{sample_id}.wav"), sample_rate=self.data_cfg.sample_rate, target=True
                )
                single_output.save_fbank(os.path.join(feat_dir, f"{sample_id}.npy"))
                single_output.save_fbank(os.path.join(feat_tgt_dir, f"{sample_id}.npy"), target=True)

        return outputs

    def valid_step(self, sample, model):
        loss, sample_size, logging_output = super().valid_step(sample, model)
        if self.cfg.eval_tts:
            hypos, inference_losses = self.valid_step_with_inference(sample, model)
            for k, v in inference_losses.items():
                assert k not in logging_output
                logging_output[k] = v

        return loss, sample_size, logging_output

    def valid_step_with_inference(self, sample, model):
        hypos = self.inference(model, sample, has_targ=True)

        losses = {
            "mcd_loss": 0.0,
            "targ_frames": 0.0,
            "pred_frames": 0.0,
            "nins": 0.0,
            "ndel": 0.0,
        }
        rets = batch_mel_cepstral_distortion(
            [hypo.targ_waveform for hypo in hypos],
            [hypo.waveform for hypo in hypos],
            self.data_cfg.sample_rate,
            normalize_type=None,
        )
        for d, extra in rets:
            pathmap = extra[-1]
            losses["mcd_loss"] += d.item()
            losses["targ_frames"] += pathmap.size(0)
            losses["pred_frames"] += pathmap.size(1)
            losses["nins"] += (pathmap.sum(dim=1) - 1).sum().item()
            losses["ndel"] += (pathmap.sum(dim=0) - 1).sum().item()

        return hypos, losses


def antidiag_indices(offset, min_i=0, max_i=None, min_j=0, max_j=None):
    """
    for a (3, 4) matrix with min_i=1, max_i=3, min_j=1, max_j=4, outputs

    offset=2 (1, 1),
    offset=3 (2, 1), (1, 2)
    offset=4 (2, 2), (1, 3)
    offset=5 (2, 3)

    constraints:
        i + j = offset
        min_j <= j < max_j
        min_i <= offset - j < max_i
    """
    if max_i is None:
        max_i = offset + 1
    if max_j is None:
        max_j = offset + 1
    min_j = max(min_j, offset - max_i + 1, 0)
    max_j = min(max_j, offset - min_i + 1, offset + 1)
    j = torch.arange(min_j, max_j)
    i = offset - j
    return torch.stack([i, j])


def batch_dynamic_time_warping(distance, shapes=None):
    """full batched DTW without any constraints

    distance:  (batchsize, max_M, max_N) matrix
    shapes: (batchsize,) vector specifying (M, N) for each entry
    """
    # ptr: 0=left, 1=up-left, 2=up
    ptr2dij = {0: (0, -1), 1: (-1, -1), 2: (-1, 0)}

    bsz, m, n = distance.size()
    cumdist = torch.zeros_like(distance)
    backptr = torch.zeros_like(distance).type(torch.int32) - 1

    # initialize
    cumdist[:, 0, :] = distance[:, 0, :].cumsum(dim=-1)
    cumdist[:, :, 0] = distance[:, :, 0].cumsum(dim=-1)
    backptr[:, 0, :] = 0
    backptr[:, :, 0] = 2

    # DP with optimized anti-diagonal parallelization, O(M+N) steps
    for offset in range(2, m + n - 1):
        ind = antidiag_indices(offset, 1, m, 1, n)
        c = torch.stack(
            [
                cumdist[:, ind[0], ind[1] - 1],
                cumdist[:, ind[0] - 1, ind[1] - 1],
                cumdist[:, ind[0] - 1, ind[1]],
            ],
            dim=2,
        )
        v, b = c.min(axis=-1)
        backptr[:, ind[0], ind[1]] = b.int()
        cumdist[:, ind[0], ind[1]] = v + distance[:, ind[0], ind[1]]

    # backtrace
    pathmap = torch.zeros_like(backptr)
    for b in range(bsz):
        i = m - 1 if shapes is None else (shapes[b][0] - 1).item()
        j = n - 1 if shapes is None else (shapes[b][1] - 1).item()
        dtwpath = [(i, j)]
        while (i != 0 or j != 0) and len(dtwpath) < 10000:
            assert i >= 0 and j >= 0
            di, dj = ptr2dij[backptr[b, i, j].item()]
            i, j = i + di, j + dj
            dtwpath.append((i, j))
        dtwpath = dtwpath[::-1]
        indices = torch.from_numpy(np.array(dtwpath))
        pathmap[b, indices[:, 0], indices[:, 1]] = 1

    return cumdist, backptr, pathmap


def compute_l2_dist(x1, x2):
    """compute an (m, n) L2 distance matrix from (m, d) and (n, d) matrices"""
    return torch.cdist(x1.unsqueeze(0), x2.unsqueeze(0), p=2).squeeze(0).pow(2)


def compute_rms_dist(x1, x2):
    l2_dist = compute_l2_dist(x1, x2)
    return (l2_dist / x1.size(1)).pow(0.5)


def get_divisor(pathmap, normalize_type):
    if normalize_type is None:
        return 1
    elif normalize_type == "len1":
        return pathmap.size(0)
    elif normalize_type == "len2":
        return pathmap.size(1)
    elif normalize_type == "path":
        return pathmap.sum().item()
    else:
        raise ValueError(f"normalize_type {normalize_type} not supported")


def batch_compute_distortion(y1, y2, sr, feat_fn, dist_fn, normalize_type):
    d, s, x1, x2 = [], [], [], []
    for cur_y1, cur_y2 in zip(y1, y2):
        assert cur_y1.ndim == 1 and cur_y2.ndim == 1
        cur_x1 = feat_fn(cur_y1)
        cur_x2 = feat_fn(cur_y2)
        x1.append(cur_x1)
        x2.append(cur_x2)

        cur_d = dist_fn(cur_x1, cur_x2)
        d.append(cur_d)
        s.append(d[-1].size())
    max_m = max(ss[0] for ss in s)
    max_n = max(ss[1] for ss in s)
    d = torch.stack([F.pad(dd, (0, max_n - dd.size(1), 0, max_m - dd.size(0))) for dd in d])
    s = torch.LongTensor(s).to(d.device)
    cumdists, backptrs, pathmaps = batch_dynamic_time_warping(d, s)

    rets = []
    itr = zip(s, x1, x2, d, cumdists, backptrs, pathmaps)
    for (m, n), cur_x1, cur_x2, dist, cumdist, backptr, pathmap in itr:
        cumdist = cumdist[:m, :n]
        backptr = backptr[:m, :n]
        pathmap = pathmap[:m, :n]
        divisor = get_divisor(pathmap, normalize_type)

        distortion = cumdist[-1, -1] / divisor
        ret = distortion, (cur_x1, cur_x2, dist, cumdist, backptr, pathmap)
        rets.append(ret)
    return rets


def batch_mel_cepstral_distortion(y1, y2, sr, normalize_type="path", mfcc_fn=None):
    """
    https://arxiv.org/pdf/2011.03568.pdf

    The root mean squared error computed on 13-dimensional MFCC using DTW for
    alignment. MFCC features are computed from an 80-channel log-mel
    spectrogram using a 50ms Hann window and hop of 12.5ms.

    y1: list of waveforms
    y2: list of waveforms
    sr: sampling rate
    """

    try:
        import torchaudio
    except ImportError:
        raise ImportError("Please install torchaudio: pip install torchaudio")

    if mfcc_fn is None or mfcc_fn.sample_rate != sr:
        melkwargs = {
            "n_fft": int(0.05 * sr),
            "win_length": int(0.05 * sr),
            "hop_length": int(0.0125 * sr),
            "f_min": 20,
            "n_mels": 80,
            "window_fn": torch.hann_window,
        }
        mfcc_fn = torchaudio.transforms.MFCC(sr, n_mfcc=13, log_mels=True, melkwargs=melkwargs).to(y1[0].device)
    return batch_compute_distortion(
        y1,
        y2,
        sr,
        lambda y: mfcc_fn(y).transpose(-1, -2),
        compute_rms_dist,
        normalize_type,
    )
