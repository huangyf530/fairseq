# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from typing import Optional

import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class CrossEntropyCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")
    knowledge_alpha: Optional[float] = field(
        default=1e-4,
        metadata={'help': "parameter for knowledge KL loss."}
    )


@register_criterion("cross_entropy", dataclass=CrossEntropyCriterionConfig)
class CrossEntropyCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg, knowledge_alpha):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.knowledge_alpha = knowledge_alpha

    def forward(self, model, sample, reduce=True, **kwargs):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        knowledge_layer = kwargs.get("knowledge_layer", [])
        net_output = model(**sample["net_input"])
        loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        knowledge_loss = 0
        for l in knowledge_layer:
            if reduce:
                knowledge_loss += l.knowledge_loss.sum()
            else:
                knowledge_loss += l.knowledge_loss
        if not isinstance(knowledge_loss, int):
            syn_loss = (1 - self.knowledge_alpha) * loss + self.knowledge_alpha * knowledge_loss
        # print("l2", loss, loss.dtype)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if not isinstance(knowledge_loss, int):
            logging_output['syn_loss'] = syn_loss.data
            logging_output['kl_loss'] = knowledge_loss.data
            return syn_loss, sample_size, logging_output
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
        )
        return loss, loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        syn_loss_sum = sum(log.get("syn_loss", 0) for log in logging_outputs)
        kl_loss_sum = sum(log.get("kl_loss", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        # log language model loss and kl divergence loss
        if syn_loss_sum > 0:
            metrics.log_scalar(
                "syn_loss", syn_loss_sum / sample_size / math.log(2), sample_size, round=3
            )
            metrics.log_scalar(
                "kl_loss", kl_loss_sum / sample_size / math.log(2), sample_size, round=3
            )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
