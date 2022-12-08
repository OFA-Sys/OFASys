# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import math

import torch

from ofasys import ModalityType
from ofasys.preprocessor.instruction import Slot
from ofasys.preprocessor.utils import collate_tokens
from ofasys.task.base import OFATask, TaskConfig


class TraverseTask(OFATask):
    def __init__(self, cfg: TaskConfig, **kwargs):
        super().__init__(cfg, **kwargs)

    def initialize(self, global_dict, **kwargs):
        super().initialize(global_dict, **kwargs)
        self.valid_batch_size = (
            self.cfg.dataset.micro_valid_batch_size
            if self.cfg.dataset.micro_valid_batch_size is not None
            else self.cfg.dataset.micro_batch_size
        )
        tgt_list = []
        prev_output_list = []
        self.index2ans = {}
        text_preprocessor = self.general_preprocess.name2pre['text']
        self.label2ans_d = {v: k for k, v in text_preprocessor.ans2label_dict.items()}
        self.constraint_trie = text_preprocessor.constraint_trie
        for i, answer in enumerate(text_preprocessor.ans2label_dict.keys()):
            # seems clumsy.. candidate answers are tokenized again..
            answer_item = self.target_dictionary.encode_line(
                line=text_preprocessor.bpe.encode(' ' + answer), add_if_not_exist=False, append_eos=False
            ).long()
            tgt_list += [torch.cat([answer_item, torch.LongTensor([self.target_dictionary.eos()])])]
            prev_output_list += [torch.cat([torch.LongTensor([self.target_dictionary.bos()]), answer_item])]
            self.index2ans[i] = answer

        constraint_mask_list = []
        for prev_output_item in prev_output_list:
            constraint_mask = torch.zeros((len(prev_output_item), len(self.target_dictionary))).bool()
            for i in range(len(prev_output_item)):
                cons_pre_tokens = prev_output_item[: i + 1].tolist()
                constraint_nodes = self.constraint_trie.get_next_layer(cons_pre_tokens)
                constraint_mask[i][constraint_nodes] = True
            constraint_mask_list.append(constraint_mask)

        eos = self.source_dictionary.eos()
        pad = self.source_dictionary.pad()
        self.val_tgt_l = []
        self.val_prev_output_l = []
        self.val_cons_masks_l = []
        for i in range(0, len(tgt_list), self.valid_batch_size):
            tgt_item = tgt_list[i : i + self.valid_batch_size]
            prev_output_item = prev_output_list[i : i + self.valid_batch_size]
            constrain_mask = constraint_mask_list[i : i + self.valid_batch_size]
            self.val_tgt_l.append(collate_tokens(tgt_item, pad_idx=pad, eos_idx=eos, left_pad=False))
            self.val_prev_output_l.append(collate_tokens(prev_output_item, pad_idx=pad, eos_idx=eos, left_pad=False))
            self.val_cons_masks_l.append(collate_tokens(constrain_mask, pad_idx=pad, left_pad=False))

    def inference(self, model, sample):
        model.eval()

        with torch.no_grad():

            slots = sample["net_input"]["slots"]
            for slot in slots:
                if slot.modality == ModalityType.TEXT and slot.is_src:
                    src_tokens = slot.value

            batch_size = src_tokens.size(0)
            encoder_out = model.encoder(list(filter(lambda x: x.is_src, slots)))
            device = src_tokens.device
            valid_result = []
            for val_tgt, val_prev_output, val_cons_masks in zip(
                self.val_tgt_l, self.val_prev_output_l, self.val_cons_masks_l
            ):
                valid_tgt_size = val_tgt.size(0)
                val_tgt = val_tgt.repeat(batch_size, 1).to(device)
                val_prev_output = val_prev_output.repeat(batch_size, 1).to(device)
                val_cons_masks = val_cons_masks.repeat(batch_size, 1, 1).to(device)
                new_encoder_out = {}
                new_encoder_out["encoder_out"] = [
                    encoder_out["encoder_out"][0].repeat_interleave(valid_tgt_size, dim=1)
                ]
                new_encoder_out["encoder_padding_mask"] = [
                    encoder_out["encoder_padding_mask"][0].repeat_interleave(valid_tgt_size, dim=0)
                ]
                new_encoder_out["position_embeddings"] = [
                    encoder_out["position_embeddings"][0].repeat_interleave(valid_tgt_size, dim=0)
                ]

                decoder_text_slot = Slot(
                    modality=ModalityType.TEXT, is_src=False, value=val_prev_output, split='valid'
                )

                decoder_out = model.decoder([decoder_text_slot], encoder_out=new_encoder_out)
                decoder_out[0].masked_fill_(~val_cons_masks, -math.inf)
                lprobs = model.get_normalized_probs(decoder_out, log_probs=True)
                scores = lprobs.gather(dim=-1, index=val_tgt.unsqueeze(-1)).squeeze(-1)
                scores = scores.masked_fill(val_tgt.eq(self.target_dictionary.pad()), 0)
                scores = scores.sum(1)
                scores = scores.view(-1, valid_tgt_size)
                valid_result.append(scores)
            valid_result = torch.cat(valid_result, dim=-1)
            predicts = valid_result.argmax(1).tolist()
            hyps = [self.index2ans[predict_index] for predict_index in predicts]
        return hyps
