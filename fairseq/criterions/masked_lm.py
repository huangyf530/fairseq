# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from fairseq import metrics, modules, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass

@dataclass
class MaskedLmConfig(FairseqDataclass):
    knowledge_alpha: Optional[float] = field(
        default=1e-4,
        metadata={'help': "parameter for knowledge KL loss."}
    )

@register_criterion("masked_lm", dataclass=MaskedLmConfig)
class MaskedLmLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, task, knowledge_alpha, tpu=False):
        super().__init__(task)
        self.tpu = tpu
        self.knowledge_alpha = knowledge_alpha

    def forward(self, model, sample, reduce=True, **kwargs):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        4) knowledge layer to get knowledge moe loss
        """
        masked_tokens = sample["target"].ne(self.padding_idx)
        sample_size = masked_tokens.int().sum()

        # Rare: when all tokens are masked, project all tokens.
        # We use torch.where to avoid device-to-host transfers,
        # except on CPU where torch.where is not well supported
        # (see github.com/pytorch/pytorch/issues/26247).
        if self.tpu:
            masked_tokens = None  # always project all tokens on TPU
        elif masked_tokens.device == torch.device("cpu"):
            if not masked_tokens.any():
                masked_tokens = None
        else:
            masked_tokens = torch.where(
                masked_tokens.any(),
                masked_tokens,
                masked_tokens.new([True]),
            )

        logits = model(**sample["net_input"], masked_tokens=masked_tokens)[0]
        targets = model.get_targets(sample, [logits])
        if masked_tokens is not None:
            targets = targets[masked_tokens]

        loss = modules.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction="sum",
            ignore_index=self.padding_idx,
        )
        knowledge_layer = kwargs.get("knowledge_layer", [])
        knowledge_loss = 0
        for l in knowledge_layer:
            token_num, expert_num = l.knowledge_loss.shape
            print("know shape", l.knowledge_loss.shape)
            print("pos shape", sample['net_input']['pos'].shape)
            print("feature shape", sample['net_input']['src_tokens'].shape)
            kl = l.knowledge_loss.view(sample['net_input']['pos'].shape[:2] + (expert_num,))
            # mask_know_loss = kl[masked_tokens]
            non_mask_know_loss = kl[masked_tokens == False]
            knowledge_loss += non_mask_know_loss.sum()
            non_mask_size = (masked_tokens == False).int().sum()
        if not isinstance(knowledge_loss, int):
            syn_loss = (1 - self.knowledge_alpha) * loss + self.knowledge_alpha * knowledge_loss

        logging_output = {
            "loss": loss if self.tpu else loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
        }
        if not isinstance(knowledge_loss, int):
            logging_output['syn_loss'] = syn_loss if self.tpu else syn_loss.data
            logging_output['kl_loss'] = knowledge_loss if self.tpu else knowledge_loss.data
            logging_output['non_mask_size'] = non_mask_size
            return syn_loss, sample_size, logging_output
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        syn_loss_sum = sum(log.get("syn_loss", 0) for log in logging_outputs)
        kl_loss_sum = sum(log.get("kl_loss", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        )
        if syn_loss_sum > 0:
            non_mask_size = sum(log.get("non_mask_size", 0) for log in logging_outputs)
            metrics.log_scalar(
                "syn_loss", syn_loss_sum / sample_size / math.log(2), sample_size, round=3
            )
            metrics.log_scalar(
                "kl_loss", kl_loss_sum / non_mask_size / math.log(2), non_mask_size, round=3
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
