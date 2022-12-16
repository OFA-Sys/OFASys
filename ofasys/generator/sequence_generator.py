# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor

from ofasys import ModalityType
from ofasys.model.incremental_decoder import IncrementalDecoder
from ofasys.preprocessor import Slot
from ofasys.utils import search
from ofasys.utils.ngram_repeat_block import NGramRepeatBlock
from ofasys.utils.trie import Trie

from .base import BatchGeneratorOutput, Generator, GeneratorOutput, MultiGeneratorOutput


@dataclass
class SequenceGeneratorOutput(GeneratorOutput):
    """
    Output of SequenceGenerator.
    Output with origin data format (e.g. string, png image file) of different modalities are available.
    Original output in tensor format and extra information are also provided.
    """

    tokens: torch.LongTensor
    score: torch.FloatTensor
    attention: torch.FloatTensor
    positional_scores: torch.FloatTensor
    text: Optional[str] = None
    image: Optional[Image.Image] = None
    box: Optional[torch.Tensor] = None

    def save_image(self, image_name: str):
        """
        Save the output image to a file.
        Parameters
        ----------
        image_name: image save path

        """
        assert self.image is not None
        if not image_name.endswith((".png", ".bmp", ".jpg", ".jpeg", ".jfif")):
            image_name = image_name + ".png"
        self.image.save(image_name)

    def save_box(self, image_name: str):
        import cv2

        assert self.image is not None and self.box is not None
        if not image_name.endswith((".png", ".bmp", ".jpg", ".jpeg", ".jfif")):
            image_name = image_name + ".png"
        image = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)
        box = self.box.to(torch.int32).tolist()
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        cv2.imwrite(image_name, image)


class SequenceGenerator(Generator):
    def __init__(
        self,
        tgt_dict,
        beam_size: int = 1,
        return_n_best: int = -1,
        max_len_a: int = 0,
        max_len_b: int = 200,
        max_len: int = 256,
        min_len: int = 1,
        normalize_scores: bool = True,
        len_penalty: float = 1.0,
        unk_penalty: float = 0.0,
        temperature: float = 1.0,
        match_source_len: bool = False,
        no_repeat_ngram_size: int = 0,
        search_strategy: Optional[search.Search] = None,
        lm_model=None,
        lm_weight: float = 1.0,
        constraint_trie: Optional[Trie] = None,
        constraint_range: Optional[str] = None,
        **unused_kwargs,
    ):
        """A autoregressive generator for discrete token sequences .
            Modified from `fairseq <https://github.com/facebookresearch/fairseq>`_.

        Args:
            tgt_dict (Dictionary): target dictionary
            beam_size (int, optional): beam width (default: 1)
            return_n_best (int, optional) return best n results (default: -1, which indicates beam_size)
            max_len_a/b (int, optional): generate sequences of maximum length
                ax + b, where x is the source length
            max_len (int, optional): the maximum length of the generated output
                (not including end-of-sentence)
            min_len (int, optional): the minimum length of the generated output
                (not including end-of-sentence)
            normalize_scores (bool, optional): normalize scores by the length
                of the output (default: True)
            len_penalty (float, optional): length penalty, where <1.0 favors
                shorter, >1.0 favors longer sentences (default: 1.0)
            unk_penalty (float, optional): unknown word penalty, where <0
                produces more unks, >0 produces fewer (default: 0.0)
            temperature (float, optional): temperature, where values
                >1.0 produce more uniform samples and values <1.0 produce
                sharper samples (default: 1.0)
            match_source_len (bool, optional): outputs should match the source
                length (default: False)
        """
        super().__init__()

        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.bos = tgt_dict.bos()
        self.eos = tgt_dict.eos()
        self.vocab_size = len(tgt_dict)

        self.beam_size = beam_size
        # the max beam size is the dictionary size - 1, since we never select pad
        self.beam_size = min(beam_size, self.vocab_size - 1)
        if return_n_best == -1:
            return_n_best = self.beam_size
        self.return_n_best = return_n_best
        self.return_n_best = min(self.beam_size, return_n_best)
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.min_len = min_len
        self.max_len = max_len

        self.normalize_scores = normalize_scores
        self.len_penalty = len_penalty
        self.unk_penalty = unk_penalty
        self.temperature = temperature
        self.match_source_len = match_source_len

        if no_repeat_ngram_size > 0:
            self.repeat_ngram_blocker = NGramRepeatBlock(no_repeat_ngram_size)
        else:
            self.repeat_ngram_blocker = None

        assert temperature > 0, "--temperature must be greater than 0"

        self.search = search.BeamSearch(tgt_dict) if search_strategy is None else search_strategy
        # We only need to set src_lengths in LengthConstrainedBeamSearch.
        # As a module attribute, setting it would break in multithread
        # settings when the model is shared.
        self.should_set_src_lengths = hasattr(self.search, "needs_src_lengths") and self.search.needs_src_lengths

        self.lm_model = lm_model
        self.lm_weight = lm_weight
        if self.lm_model is not None:
            self.lm_model.eval()

        self.constraint_trie = constraint_trie
        self.constraint_start, self.constraint_end = None, None
        if constraint_range is not None:
            self.constraint_start, self.constraint_end = eval(constraint_range)

    @torch.no_grad()
    def generate(self, model, sample, **kwargs):
        """Generate function.

        Args:
            models (ofasys.model.OFAModel): OFAModel
            sample (dict): batch
        """
        model = WrapperModel(model)
        model.eval()

        constraints: Optional[Tensor] = kwargs.pop("constraints", None)

        incremental_state = torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})

        net_input = sample["net_input"]
        source_slots = list(filter(lambda x: x.is_src, net_input["slots"]))
        text_slots = list(
            filter(lambda x: x.is_src and x.modality == ModalityType and not x.is_plaintext, net_input["slots"])
        )
        target_slot = Slot.get_target_slot_from_slots(net_input["slots"])
        prefix_tokens: Optional[Tensor] = sample.get("prefix_tokens", None)

        if len(text_slots) == 1:
            src_tokens = text_slots[0].value
            src_lengths = (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
            bsz, src_len = src_tokens.size()[:2]
        else:
            if source_slots[0].modality == ModalityType.AUDIO:
                src_tokens = source_slots[0].value["fbank"]
            else:
                src_tokens = source_slots[0].value
            src_len, src_lengths = None, None
            bsz = src_tokens.shape[0]

        beam_size = self.beam_size

        if constraints is not None and not self.search.supports_constraints:
            raise NotImplementedError(
                "Target-side constraints were provided, " "but search method doesn't support them"
            )

        # Initialize constraints, when active
        self.search.init_constraints(constraints, beam_size)

        max_len: int = self.max_len
        if target_slot.modality == ModalityType.TEXT:
            if self.match_source_len and src_lengths is not None:
                max_len = src_lengths.max().item()
            elif src_len is not None:
                max_len = min(max_len, int(self.max_len_a * src_len + self.max_len_b))
        assert self.min_len <= max_len, "min_len cannot be larger than max_len, please adjust these!"
        # compute the encoder output for each beam
        with torch.autograd.profiler.record_function("Model: forward_encoder"):
            encoder_out = model.forward_encoder(slots=source_slots)

        # placeholder of indices for bsz * beam_size to hold tokens
        # and accumulative scores
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_out = model.reorder_encoder_out(encoder_out, new_order)

        # initialize buffers
        scores = (
            torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens).float()
        )  # +1 for eos; pad is never chosen for scoring
        tokens = torch.zeros(bsz * beam_size, max_len + 2).to(src_tokens).long().fill_(self.pad)  # +2 for eos and pad
        # tokens[:, 0] = self.eos if bos_token is None else bos_token
        tokens[:, 0] = self.bos
        attn: Optional[Tensor] = None

        # A list that indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then cands_to_ignore would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        cands_to_ignore = (
            torch.zeros(bsz, beam_size).to(src_tokens).eq(-1)
        )  # forward and backward-compatible False mask

        # list of completed sentences
        finalized: BatchGeneratorOutput = [[] for _ in range(bsz)]
        # contains lists of dictionaries of infomation about the hypothesis being
        # finalized at each step

        # a boolean array indicating if the sentence at the index is finished or not
        finished = [False for i in range(bsz)]
        num_remaining_sent = bsz  # number of sentences remaining

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(tokens).to(src_tokens.device)
        cand_offsets = torch.arange(0, cand_size).type_as(tokens).to(src_tokens.device)

        reorder_state: Optional[Tensor] = None
        batch_idxs: Optional[Tensor] = None

        original_batch_idxs: Optional[Tensor] = None
        if "id" in sample and isinstance(sample["id"], Tensor):
            original_batch_idxs = sample["id"]
        else:
            original_batch_idxs = torch.arange(0, bsz).type_as(tokens)
        for step in range(max_len + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(batch_idxs)
                    reorder_state.view(-1, beam_size).add_(corr.unsqueeze(-1) * beam_size)
                    original_batch_idxs = original_batch_idxs[batch_idxs]
                model.reorder_incremental_state(incremental_state, reorder_state)
                encoder_out = model.reorder_encoder_out(encoder_out, reorder_state)

            with torch.autograd.profiler.record_function("Model: forward_decoder"):
                target_slot.value = tokens[:, : step + 1]
                lprobs, attn_scores = model.forward_decoder(
                    [target_slot],
                    tokens[:, : step + 1],
                    encoder_out,
                    incremental_state,
                    self.temperature,
                    constraint_trie=self.constraint_trie,
                    constraint_start=self.constraint_start,
                    constraint_end=self.constraint_end,
                    prefix_tokens=prefix_tokens,
                )

            if self.lm_model is not None:
                lm_out = self.lm_model(tokens[:, : step + 1])
                probs = self.lm_model.get_normalized_probs(lm_out, log_probs=True, sample=None)
                probs = probs[:, -1, :] * self.lm_weight
                lprobs += probs
            # handle prefix tokens (possibly with different lengths)
            if prefix_tokens is not None and step < prefix_tokens.size(1) and step < max_len:
                lprobs, tokens, scores = self._prefix_tokens(step, lprobs, scores, tokens, prefix_tokens, beam_size)
            elif step < self.min_len:
                # minimum length constraint (does not apply if using prefix_tokens)
                lprobs[:, self.eos] = -math.inf

            lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)

            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            # handle max length constraint
            if step >= max_len:
                lprobs[:, : self.eos] = -math.inf
                lprobs[:, self.eos + 1 :] = -math.inf

            # Record attention scores, only support attn_scores is a Tensor
            if attn_scores is not None:
                if attn is None:
                    attn = torch.empty(bsz * beam_size, attn_scores.size(1), max_len + 2).to(scores)
                attn[:, :, step + 1].copy_(attn_scores)

            scores = scores.type_as(lprobs)
            eos_bbsz_idx = torch.empty(0).to(tokens)  # indices of hypothesis ending with eos (finished sentences)
            eos_scores = torch.empty(0).to(scores)  # scores of hypothesis ending with eos (finished sentences)

            if self.should_set_src_lengths and src_lengths is not None:
                self.search.set_src_lengths(src_lengths)
            if self.repeat_ngram_blocker is not None:
                # process prefix_tokens
                p_toks_len = prefix_tokens.ne(self.pad).sum(dim=1) if prefix_tokens is not None else None
                if p_toks_len is not None:
                    p_toks_len_beam = p_toks_len.unsqueeze(-1).repeat(1, beam_size).view(-1)
                    no_repeat_ngram_size = self.repeat_ngram_blocker.no_repeat_ngram_size
                    out_prefix = p_toks_len_beam < (step + no_repeat_ngram_size - 1)
                else:
                    out_prefix = torch.ones(bsz * beam_size).bool()
                ngram_blocker_tokens = tokens[out_prefix]
                ngram_blocker_lprobs = lprobs[out_prefix]
                ngram_blocker_bsz = out_prefix.sum() // beam_size
                lprobs[out_prefix] = self.repeat_ngram_blocker(
                    tokens=ngram_blocker_tokens,
                    lprobs=ngram_blocker_lprobs,
                    bsz=ngram_blocker_bsz,
                    beam_size=beam_size,
                    step=step,
                )

            # Shape: (batch, cand_size)
            cand_scores, cand_indices, cand_beams = self.search.step(
                step,
                lprobs.view(bsz, -1, self.vocab_size),
                scores.view(bsz, beam_size, -1)[:, :, :step],
                tokens[:, : step + 1],
                original_batch_idxs,
            )

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # finalize hypotheses that end in eos
            # Shape of eos_mask: (batch size, beam size)
            eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-math.inf)
            eos_mask[:, :beam_size][cands_to_ignore] = torch.tensor(0).to(eos_mask)

            # only consider eos when it's among the top beam_size indices
            # Now we know what beam item(s) to finish
            # Shape: 1d list of absolute-numbered
            eos_bbsz_idx = torch.masked_select(cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size])
            finalized_sents: List[int] = []
            # add `and` condition: prefix_tokens not equal eos
            if eos_bbsz_idx.numel() > 0:
                eos_scores = torch.masked_select(cand_scores[:, :beam_size], mask=eos_mask[:, :beam_size])

                finalized_sents = self.finalize_hypos(
                    step,
                    eos_bbsz_idx,
                    eos_scores,
                    tokens,
                    scores,
                    finalized,
                    finished,
                    beam_size,
                    attn,
                    src_lengths,
                    max_len,
                )
                num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            if self.search.stop_on_max_len and step >= max_len:
                break
            assert step <= max_len, f"{step} <= {max_len}"

            # Remove finalized sentences (ones for which {beam_size}
            # finished hypotheses have been generated) from the batch.
            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep
                # for the next pass
                batch_mask = torch.ones(bsz, dtype=torch.bool, device=cand_indices.device)
                batch_mask[finalized_sents] = False
                # TODO replace `nonzero(as_tuple=False)` after TorchScript supports it
                batch_idxs = torch.arange(bsz, device=cand_indices.device).masked_select(batch_mask)

                # Choose the subset of the hypothesized constraints that will continue
                self.search.prune_sentences(batch_idxs)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]

                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]
                if src_lengths is not None:
                    src_lengths = src_lengths[batch_idxs]
                cands_to_ignore = cands_to_ignore[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, attn.size(1), -1)
                bsz = new_bsz
            else:
                batch_idxs = None

            # Set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos

            # Rewrite the operator since the element wise or is not supported
            # in torchscript.

            eos_mask[:, :beam_size] = ~((~cands_to_ignore) & (~eos_mask[:, :beam_size]))
            active_mask = torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[: eos_mask.size(1)],
            )

            # get the top beam_size active hypotheses, which are just
            # the hypos with the smallest values in active_mask.
            # {active_hypos} indicates which {beam_size} hypotheses
            # from the list of {2 * beam_size} candidates were
            # selected. Shapes: (batch size, beam size)
            new_cands_to_ignore, active_hypos = torch.topk(active_mask, k=beam_size, dim=1, largest=False)

            # update cands_to_ignore to ignore any finalized hypos.
            cands_to_ignore = new_cands_to_ignore.ge(cand_size)[:, :beam_size]
            # Make sure there is at least one active item for each sentence in the batch.
            assert (~cands_to_ignore).any(dim=1).all()

            # update cands_to_ignore to ignore any finalized hypos

            # {active_bbsz_idx} denotes which beam number is continued for each new
            # hypothesis (a beam can be selected more than once).
            active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos)
            active_scores = torch.gather(cand_scores, dim=1, index=active_hypos)

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses

            # Set the tokens for each beam (can select the same row more than once)
            tokens[:, : step + 1] = torch.index_select(tokens[:, : step + 1], dim=0, index=active_bbsz_idx)
            # Select the next token for each of them
            tokens.view(bsz, beam_size, -1)[:, :, step + 1] = torch.gather(cand_indices, dim=1, index=active_hypos)
            if step > 0:
                scores[:, :step] = torch.index_select(scores[:, :step], dim=0, index=active_bbsz_idx)
            scores.view(bsz, beam_size, -1)[:, :, step] = torch.gather(cand_scores, dim=1, index=active_hypos)

            # Update constraints based on which candidates were selected
            # for the next beam
            self.search.update_constraints(active_hypos)

            # copy attention for active hypotheses
            if attn is not None:
                attn[:, :, : step + 2] = torch.index_select(attn[:, :, : step + 2], dim=0, index=active_bbsz_idx)

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(len(finalized)):
            scores = torch.tensor([float(elem.score.item()) for elem in finalized[sent]])
            _, sorted_scores_indices = torch.sort(scores, descending=True)
            if self.return_n_best == 1:
                finalized[sent] = finalized[sent][sorted_scores_indices[0]]
            else:
                finalized[sent] = [finalized[sent][ssi] for ssi in sorted_scores_indices][: self.return_n_best]
        return finalized

    def _prefix_tokens(self, step: int, lprobs, scores, tokens, prefix_tokens, beam_size: int):
        """Handle prefix tokens"""
        prefix_toks = prefix_tokens[:, step].unsqueeze(-1).repeat(1, beam_size).view(-1)
        prefix_lprobs = lprobs.gather(-1, prefix_toks.unsqueeze(-1))
        prefix_mask = prefix_toks.ne(self.pad)
        if self.constraint_trie is None:
            lprobs[prefix_mask] = torch.min(prefix_lprobs) - 1
        else:
            lprobs[prefix_mask] = -math.inf
        lprobs[prefix_mask] = lprobs[prefix_mask].scatter(
            -1, prefix_toks[prefix_mask].unsqueeze(-1), prefix_lprobs[prefix_mask]
        )
        # if prefix includes eos, then we should make sure tokens and
        # scores are the same across all beams
        eos_mask = prefix_toks.eq(self.eos)
        if eos_mask.any():
            # validate that the first beam matches the prefix
            first_beam = tokens[eos_mask].view(-1, beam_size, tokens.size(-1))[:, 0, 1 : step + 1]
            eos_mask_batch_dim = eos_mask.view(-1, beam_size)[:, 0]
            target_prefix = prefix_tokens[eos_mask_batch_dim][:, :step]
            assert (first_beam == target_prefix).all()

            # copy tokens, scores and lprobs from the first beam to all beams
            tokens = self.replicate_first_beam(tokens, eos_mask_batch_dim, beam_size)
            scores = self.replicate_first_beam(scores, eos_mask_batch_dim, beam_size)
            lprobs = self.replicate_first_beam(lprobs, eos_mask_batch_dim, beam_size)
        return lprobs, tokens, scores

    def replicate_first_beam(self, tensor, mask, beam_size: int):
        tensor = tensor.view(-1, beam_size, tensor.size(-1))
        tensor[mask] = tensor[mask][:, :1, :]
        return tensor.view(-1, tensor.size(-1))

    def finalize_hypos(
        self,
        step: int,
        bbsz_idx,
        eos_scores,
        tokens,
        scores,
        finalized: BatchGeneratorOutput,
        finished: List[bool],
        beam_size: int,
        attn: Optional[Tensor],
        src_lengths,
        max_len: int,
    ):
        """
        Finalize hypothesis, store finalized information in `finalized`, and change
        `finished` accordingly. A sentence is finalized when {beam_size} finished items
        have been collected for it.

        Returns number of sentences (not beam items) being finalized.
        These will be removed from the batch and not processed further.
        """
        assert bbsz_idx.numel() == eos_scores.numel()

        # clone relevant token and attention tensors.
        # tokens is (batch * beam, max_len). So the index_select
        # gets the newly EOS rows, then selects cols 1..{step + 2}
        tokens_clone = tokens.index_select(0, bbsz_idx)[:, 1 : step + 2]  # skip the first index, which is EOS

        tokens_clone[:, step] = self.eos
        attn_clone = attn.index_select(0, bbsz_idx)[:, :, 1 : step + 2] if attn is not None else None

        # compute scores per token position
        pos_scores = scores.index_select(0, bbsz_idx)[:, : step + 1]
        pos_scores[:, step] = eos_scores
        # convert from cumulative to per-position scores
        pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

        # normalize sentence-level scores
        if self.normalize_scores:
            eos_scores /= (step + 1) ** self.len_penalty

        # cum_unfin records which sentences in the batch are finished.
        # It helps match indexing between (a) the original sentences
        # in the batch and (b) the current, possibly-reduced set of
        # sentences.
        cum_unfin: List[int] = []
        prev = 0
        for f in finished:
            if f:
                prev += 1
            else:
                cum_unfin.append(prev)
        cum_fin_tensor = torch.tensor(cum_unfin, dtype=torch.int).to(bbsz_idx)

        unfin_idx = bbsz_idx // beam_size
        # unfin_idx = torch.div(bbsz_idx, beam_size, rounding_mode='trunc')

        sent = unfin_idx + torch.index_select(cum_fin_tensor, 0, unfin_idx)

        # Create a set of "{sent}{unfin_idx}", where
        # "unfin_idx" is the index in the current (possibly reduced)
        # list of sentences, and "sent" is the index in the original,
        # unreduced batch
        # For every finished beam item
        # sentence index in the current (possibly reduced) batch
        seen = (sent << 32) + unfin_idx
        unique_seen: List[int] = torch.unique(seen).tolist()

        if self.match_source_len and src_lengths is not None:
            condition = step > torch.index_select(src_lengths, 0, unfin_idx)
            eos_scores = torch.where(condition, torch.tensor(-math.inf), eos_scores)
        sent_list: List[int] = sent.tolist()
        for i in range(bbsz_idx.size()[0]):
            # An input sentence (among those in a batch) is finished when
            # beam_size hypotheses have been collected for it
            if len(finalized[sent_list[i]]) < beam_size:
                if attn_clone is not None:
                    # remove padding tokens from attn scores
                    hypo_attn = attn_clone[i]
                else:
                    hypo_attn = torch.empty(0)

                finalized[sent_list[i]].append(
                    SequenceGeneratorOutput(
                        tokens=tokens_clone[i],
                        score=eos_scores[i],
                        attention=hypo_attn,  # src_len x tgt_len
                        positional_scores=pos_scores[i],
                    )
                )

        newly_finished: List[int] = []
        for unique_s in unique_seen:
            # check termination conditions for this sentence
            unique_sent: int = unique_s >> 32
            unique_unfin_idx: int = unique_s - (unique_sent << 32)

            if not finished[unique_sent] and self.is_finished(
                step, unique_unfin_idx, max_len, len(finalized[unique_sent]), beam_size
            ):
                finished[unique_sent] = True
                newly_finished.append(unique_unfin_idx)

        return newly_finished

    def is_finished(
        self,
        step: int,
        unfin_idx: int,
        max_len: int,
        finalized_sent_len: int,
        beam_size: int,
    ):
        """
        Check whether decoding for a sentence is finished, which
        occurs when the list of finalized sentences has reached the
        beam size, or when we reach the maximum length.
        """
        assert finalized_sent_len <= beam_size
        if finalized_sent_len == beam_size or step == max_len:
            return True
        return False


class WrapperModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

        self.has_incremental: bool = False
        if hasattr(self.model, "decoder") and isinstance(self.model.decoder, IncrementalDecoder):
            self.has_incremental = True

    def forward(self):
        pass

    def has_encoder(self):
        return hasattr(self.model, "encoder")

    def has_incremental_states(self):
        return self.has_incremental

    def max_decoder_positions(self):
        max_positions = getattr(self.model, "max_decoder_positions", 100000000)
        return max_positions

    @torch.jit.export
    def forward_encoder(self, *args, **kwargs):
        if not self.has_encoder():
            return None
        return self.model.encoder(*args, **kwargs)

    @torch.jit.export
    def forward_decoder(
        self,
        slots,
        tokens,
        encoder_out: Dict[str, List[Tensor]],
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        temperature: float = 1.0,
        constraint_trie=None,
        constraint_start=None,
        constraint_end=None,
        prefix_tokens=None,
    ):
        # decode each model
        if self.has_incremental_states():
            decoder_out = self.model.decoder.forward(
                slots,
                encoder_out=encoder_out,
                incremental_state=incremental_state,
            )
        else:
            if hasattr(self.model, "decoder"):
                decoder_out = self.model.decoder.forward(slots, encoder_out=encoder_out)
            else:
                decoder_out = self.model.forward(slots)

        attn: Optional[Tensor] = None
        decoder_len = len(decoder_out)
        if decoder_len > 1 and decoder_out[1] is not None:
            if isinstance(decoder_out[1], Tensor):
                attn = decoder_out[1]
            else:
                attn_holder = decoder_out[1]["attn"]
                if isinstance(attn_holder, Tensor):
                    attn = attn_holder
                elif attn_holder is not None:
                    attn = attn_holder[0]
            if attn is not None:
                attn = attn[:, -1, :]

        decoder_out_tuple = (
            decoder_out[0][:, -1:, :].div_(temperature),
            None if decoder_len <= 1 else decoder_out[1],
        )

        beam_size = decoder_out_tuple[0].size(0) // prefix_tokens.size(0) if prefix_tokens is not None else 0
        if constraint_trie is not None:
            assert constraint_start is None and constraint_end is None
            constraint_masks = decoder_out_tuple[0].new_zeros(decoder_out_tuple[0].size()).bool()
            constraint_prefix_tokens = tokens.tolist()
            for idx, constraint_prefix_token in enumerate(constraint_prefix_tokens):
                prefix_len = prefix_tokens[idx // beam_size].ne(1).sum().item() if prefix_tokens is not None else 0
                if len(constraint_prefix_token) > prefix_len:
                    constraint_prefix_token = [0] + constraint_prefix_token[prefix_len + 1 :]
                    constraint_nodes = constraint_trie.get_next_layer(constraint_prefix_token)
                    constraint_masks[idx][:, constraint_nodes] = True
                else:
                    constraint_masks[idx] = True
            decoder_out_tuple[0].masked_fill_(~constraint_masks, -math.inf)
        if constraint_start is not None and constraint_end is not None:
            assert constraint_trie is None
            decoder_out_tuple[0][:, :, 4:constraint_start] = -math.inf
            decoder_out_tuple[0][:, :, constraint_end:] = -math.inf

        probs = self.model.get_normalized_probs(decoder_out_tuple, log_probs=True, sample=None)
        probs = probs[:, -1, :]

        return probs, attn

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Optional[Dict[str, List[Tensor]]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        new_out: Optional[Dict[str, List[Tensor]]] = None
        if not self.has_encoder():
            return new_out
        new_out = self.model.encoder.reorder_encoder_out(encoder_out, new_order)
        return new_out

    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order,
    ):
        if not self.has_incremental_states():
            return
        self.model.decoder.reorder_incremental_state_scripting(incremental_state, new_order)
