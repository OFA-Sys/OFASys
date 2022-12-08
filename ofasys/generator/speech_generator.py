# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

try:
    import soundfile as sf
except:
    pass

import torch

from ofasys import ModalityType
from ofasys.preprocessor import Slot
from ofasys.utils.file_utils import cached_path

from .base import BatchGeneratorOutput, Generator, GeneratorOutput, to_numpy


@dataclass
class SpeechGeneratorOutput(GeneratorOutput):
    """
    Output of SpeechGeneratorOutput.
    Output with origin data format (e.g. string, audio wav) of different modalities are available.
    Original output in tensor format and extra information are also provided.
    """

    feature: Union[torch.FloatTensor, np.ndarray]
    eos_prob: torch.FloatTensor
    attn: torch.FloatTensor
    alignment: torch.Tensor
    text: Optional[str] = None
    waveform: Optional[Union[torch.FloatTensor, np.ndarray]] = None
    targ_feature: Optional[Union[torch.FloatTensor, np.ndarray]] = None
    targ_waveform: Optional[Union[torch.FloatTensor, np.ndarray]] = None

    def save_audio(self, audio_name: str, sample_rate: int = 22050, target: bool = False):
        waveform = to_numpy(self.targ_waveform if target else self.waveform)
        assert waveform is not None
        if not audio_name.endswith(".wav"):
            audio_name = audio_name + ".wav"
        sf.write(audio_name, waveform, sample_rate)

    def save_fbank(self, fbank_name: str, target: bool = False):
        feature = to_numpy(self.targ_feature if target else self.feature)
        assert feature is not None
        if not fbank_name.endswith(".npy"):
            fbank_name = fbank_name + ".npy"
        np.save(fbank_name, feature)


class SpeechGenerator(Generator):
    def __init__(self, src_dict, stats_npz_path: Optional[str] = None):
        """
        Base Generator class for Audio modality.
        """
        super().__init__()
        self.pad = src_dict.pad()
        self.unk = src_dict.unk()
        self.bos = src_dict.bos()
        self.eos = src_dict.eos()

        self.gcmvn_stats = None
        if stats_npz_path is not None:
            local_stats_npz_path = cached_path(stats_npz_path)
            self.gcmvn_stats = np.load(Path(local_stats_npz_path))

    def gcmvn_denormalize(self, x):
        # x: B x T x C
        if self.gcmvn_stats is None:
            return x
        mean = torch.from_numpy(self.gcmvn_stats["mean"]).to(x)
        std = torch.from_numpy(self.gcmvn_stats["std"]).to(x)
        assert len(x.shape) == 3 and mean.shape[0] == std.shape[0] == x.shape[2]
        x = x * std.view(1, 1, -1).expand_as(x)
        return x + mean.view(1, 1, -1).expand_as(x)


class AutoRegressiveSpeechGenerator(SpeechGenerator):
    def __init__(
        self,
        src_dict,
        stats_npz_path: Optional[str] = None,
        max_iter: int = 6000,
        eos_prob_threshold: float = 0.5,
        **unused_kwargs,
    ):
        """A autoregressive generator for contiguous audio feature sequences .
        Modified from `fairseq <https://github.com/facebookresearch/fairseq>`_.

        Args:
            src_dict: source dictionary.
            stats_npz_path: gcmvn_stats path.
            max_iter: max iteration steps.
            eos_prob_threshold: threshold for generating end of sequence.
        """
        super().__init__(src_dict, stats_npz_path)
        self.max_iter = max_iter
        self.eos_prob_threshold = eos_prob_threshold

    @torch.no_grad()
    def generate(self, model, sample, **kwargs):
        """
        Generate function.
        """
        model.eval()

        has_targ: bool = kwargs.pop("has_targ", False)

        net_input = sample["net_input"]
        source_slots = list(filter(lambda x: x.is_src, net_input['slots']))
        target_slot = Slot.get_target_slot_from_slots(net_input["slots"])
        assert target_slot.modality == ModalityType.AUDIO, (
            f"the target slot does not match the generator,"
            f" target_slot: {target_slot.modality}, generator: AutoRegressiveSpeechGenerator"
        )

        if source_slots[0].modality == ModalityType.AUDIO:
            src_tokens = source_slots[0].value["fbank"]
        else:
            src_tokens = source_slots[0].value
        bsz = src_tokens.shape[0]

        audio_adaptor = model.decoder.adaptor.get_adaptor(target_slot)
        n_frames_per_step = audio_adaptor.n_frames_per_step
        out_dim = audio_adaptor.out_dim
        raw_dim = out_dim // n_frames_per_step

        # initialize
        encoder_out = model.encoder.forward(slots=source_slots)

        incremental_state = {}
        feat, attn, eos_prob = [], [], []
        finished = src_tokens.new_zeros((bsz,)).bool()
        out_lens = src_tokens.new_zeros((bsz,)).long().fill_(self.max_iter)

        prev_feat_out = encoder_out["encoder_out"][0].new_zeros(bsz, 1, out_dim)
        for step in range(self.max_iter):
            cur_out_lens = out_lens.clone()
            cur_out_lens.masked_fill_(cur_out_lens.eq(self.max_iter), step + 1)
            target_slot.value = {
                "fbank": prev_feat_out,
                "fbank_lengths": cur_out_lens,
            }
            _, cur_extra = model.decoder.forward(
                [target_slot],
                encoder_out=encoder_out,
                incremental_state=incremental_state,
            )
            cur_eos_out = cur_extra['eos_out']
            cur_eos_prob = torch.sigmoid(cur_eos_out).squeeze(2)
            feat.append(cur_extra['feature_out'])
            attn.append(cur_extra['attn'])
            eos_prob.append(cur_eos_prob)

            cur_finished = cur_eos_prob.squeeze(1) > self.eos_prob_threshold
            out_lens.masked_fill_((~finished) & cur_finished, step + 1)
            finished = finished | cur_finished
            if finished.sum().item() == bsz:
                break
            prev_feat_out = torch.cat([prev_feat_out, cur_extra['feature_out']], dim=1)

        feat = torch.cat(feat, dim=1)
        feat = audio_adaptor.postnet(feat) + feat
        eos_prob = torch.cat(eos_prob, dim=1)
        attn = torch.cat(attn[0], dim=2)
        alignment = attn.max(dim=1)[1]

        feat = feat.reshape(bsz, -1, raw_dim)
        feat = self.gcmvn_denormalize(feat)

        eos_prob = eos_prob.repeat_interleave(n_frames_per_step, dim=1)
        attn = attn.repeat_interleave(n_frames_per_step, dim=2)
        alignment = alignment.repeat_interleave(n_frames_per_step, dim=1)
        out_lens = out_lens * n_frames_per_step

        finalized: BatchGeneratorOutput = [
            SpeechGeneratorOutput(
                feature=feat[b, :out_len],
                eos_prob=eos_prob[b, :out_len],
                attn=attn[b, :, :out_len],
                alignment=alignment[b, :out_len],
            )
            for b, out_len in zip(range(bsz), out_lens)
        ]

        if has_targ:
            assert sample["target"].size(-1) == out_dim
            tgt_feats = sample["target"].view(bsz, -1, raw_dim)
            tgt_feats = self.gcmvn_denormalize(tgt_feats)
            tgt_lens = sample["target_lengths"] * n_frames_per_step
            for b, (f, l) in enumerate(zip(tgt_feats, tgt_lens)):
                finalized[b].targ_feature = f[:l]

        return finalized
