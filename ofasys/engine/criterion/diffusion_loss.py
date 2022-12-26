# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from dataclasses import dataclass, field

import torch

from ofasys.configure import register_config
from ofasys.engine.criterion.base import BaseCriterion, CriterionConfig
from ofasys.logging import metrics
from ofasys.module.diffusion import DiffusionWrapper, build_denoise_fn


@dataclass
class DiffusionCriterionConfig(CriterionConfig):
    scale_main_loss: float = field(default=1.0, metadata={"help": ""})
    scale_aux_loss_1: float = field(default=0.0, metadata={"help": ""})
    scale_aux_loss_2: float = field(default=0.0, metadata={"help": ""})


@register_config("ofasys.criterion", "diffusion_criterion", DiffusionCriterionConfig)
class DiffusionCriterion(BaseCriterion):
    def __init__(self, task, cfg: DiffusionCriterionConfig):
        super().__init__(task, cfg)
        self.diffusion = DiffusionWrapper(**task.diffuser_args)
        self.general_preprocess = task.general_preprocess
        self.scale_main_loss = cfg.scale_main_loss
        self.scale_aux_loss_1 = cfg.scale_aux_loss_1
        self.scale_aux_loss_2 = cfg.scale_aux_loss_2

    def forward(self, model, sample, update_num=0, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        denoise_fn, x_start, slot = build_denoise_fn(net_input=sample["net_input"], model=model)
        loss, x_predict, sample_weights = self.diffusion.p_losses(denoise_fn=denoise_fn, x_start=x_start)  # [B,T,D]

        assert len(loss.shape) == 3, (
            'This criterion assumes that the model input and output are of shape'
            ' [batch_size, num_tokens, token_embedding_size].'
            ' When processing images, please reshape images from shape [B,C,H,W] to shape [B,H*W,C].'
        )

        if "masks" in slot.value:
            loss = loss.mean(dim=-1)  # [B,T]
            weights = torch.logical_not(slot.value["masks"]).type_as(loss)
            assert loss.shape == weights.shape, "This criterion assumes masks to be of shape [batch_size, num_tokens]."
            loss = torch.mean(torch.sum(weights * loss, dim=-1) / torch.sum(weights, dim=-1), dim=-1)  # [B,T]->scalar
        else:
            loss = loss.mean()
        loss = self.scale_main_loss * loss

        sample_size = 1
        logging_output = {
            "main_loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
        }

        # Example usage: To measure how physically plausible the prediction is and regularize it.
        # In this case, only the preprocessor knows the floats' physical meanings and knows how to do it.
        slot_preproc = self.general_preprocess.get_preprocess(slot)
        if (self.scale_aux_loss_1 > 0) and hasattr(slot_preproc, 'custom_reg_loss'):
            aux_loss_1 = self.scale_aux_loss_1 * slot_preproc.custom_reg_loss(slot, x_predict, x_start, sample_weights)
            logging_output["aux_loss_1"] = aux_loss_1.data
            loss = loss + aux_loss_1

        # Example usage: To regularize the slot adaptor's latent states based on the ground-truth.
        # Caution: Don't introduce new trainable parameters not used by model.forward. Or DDP may fail.
        slot_adaptor = model.decoder.adaptor.get_adaptor(slot)
        if (self.scale_aux_loss_2 > 0) and hasattr(slot_adaptor, 'custom_reg_loss'):
            aux_loss_2 = self.scale_aux_loss_2 * slot_adaptor.custom_reg_loss(slot, x_predict, x_start, sample_weights)
            logging_output["aux_loss_2"] = aux_loss_2.data
            loss = loss + aux_loss_2

        logging_output["loss"] = loss.data

        return loss * self.weight, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs, prefix_name=None) -> None:
        """Aggregate logging outputs from data parallel training."""
        task_name = prefix_name + '/' if prefix_name else ''
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        for k in ["loss", "main_loss", "aux_loss_1", "aux_loss_2"]:
            loss_sum = sum(log.get(k, 0) for log in logging_outputs)
            metrics.log_scalar(f"{task_name}{k}", loss_sum / sample_size, sample_size, round=5)
            if k == "loss":
                metrics.log_scalar(f"loss", loss_sum / sample_size, sample_size, priority=0, round=5)
        metrics.log_scalar(f"{task_name}ntokens", ntokens, 1, round=3)
        metrics.log_scalar(f"{task_name}bsz", nsentences, 1, round=3)
        metrics.log_scalar(f"{task_name}sample_size", sample_size, 1, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improve distributed training speed.
        """
        return True
