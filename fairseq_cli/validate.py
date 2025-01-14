#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
from argparse import Namespace
from itertools import chain

import torch
from fairseq import checkpoint_utils, distributed_utils, options, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import metrics, progress_bar
from omegaconf import DictConfig
from fairseq.data.encoders.gpt2_bpe import get_encoder


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.validate")


def main(cfg: DictConfig, override_args=None):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)

    assert (
        cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"

    use_fp16 = cfg.common.fp16
    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    if use_cuda:
        torch.cuda.set_device(cfg.distributed_training.device_id)

    if cfg.distributed_training.distributed_world_size > 1:
        data_parallel_world_size = distributed_utils.get_data_parallel_world_size()
        data_parallel_rank = distributed_utils.get_data_parallel_rank()
    else:
        data_parallel_world_size = 1
        data_parallel_rank = 0

    if override_args is not None:
        overrides = vars(override_args)
        overrides.update(eval(getattr(override_args, "model_overrides", "{}")))
    else:
        overrides = None

    # Load ensemble
    if cfg.common_eval.load_checkpoint_on_all_dp_ranks:
        cfg.checkpoint.checkpoint_suffix = f"-rank-{cfg.distributed_training.distributed_rank}"
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [cfg.common_eval.path],
        arg_overrides=overrides,
        suffix=cfg.checkpoint.checkpoint_suffix,
    )
    model = models[0]

    # Move models to GPU
    for model in models:
        model.eval()
        if use_fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Print args
    logger.info(saved_cfg)

    # Build criterion
    criterion = task.build_criterion(saved_cfg.criterion)
    criterion.eval()

    for subset in cfg.dataset.valid_subset.split(","):
        try:
            task.load_dataset(subset, combine=False, epoch=1, task_cfg=saved_cfg.task)
            dataset = task.dataset(subset)
        except KeyError:
            raise Exception("Cannot find dataset: " + subset)

        # Initialize data iterator
        batch_itr = task.get_batch_iterator(
            dataset=dataset,
            max_tokens=cfg.dataset.max_tokens,
            max_sentences=cfg.dataset.batch_size,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                *[m.max_positions() for m in models],
            ),
            ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
            seed=cfg.common.seed,
            num_shards=data_parallel_world_size,
            shard_id=data_parallel_rank,
            num_workers=cfg.dataset.num_workers,
            data_buffer_size=cfg.dataset.data_buffer_size,
        )
        itr = batch_itr.next_epoch_itr(shuffle=False)
        progress = progress_bar.progress_bar(
            itr,
            log_format=cfg.common.log_format,
            log_interval=cfg.common.log_interval,
            prefix=f"valid on '{subset}' subset",
            default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
        )
        _dummy_batch = batch_itr.first_batch

        log_outputs = []
        # encoder = get_encoder("gpt2_bpe/encoder.json", "gpt2_bpe/vocab.bpe")
        for i, sample in enumerate(progress):
            if len(sample) == 0:
                sample = _dummy_batch
            # for t in [1, 2]:
            #     tokens = task.dictionary.string(sample['net_input']['src_tokens'][t]).split(' ')
            #     length = sample['net_input']['src_lengths'][t].item()
            #     # print(tokens[:length])
            #     print(encoder.decode(map(int, tokens[:length])))
            # quit()
            if 'pos' in sample:
                sample['token']['net_input']['pos'] = sample['pos']['net_input']['src_tokens']
                sample = sample['token']
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            _loss, _sample_size, log_output = task.valid_step(sample, model, criterion)
            progress.log(log_output, step=i)
            log_outputs.append(log_output)

        if data_parallel_world_size > 1:
            logger.info("gather log output among different workers.")
            log_outputs = distributed_utils.all_gather_list(
                log_outputs,
                max_size=cfg.common.all_gather_list_size,
                group=distributed_utils.get_data_parallel_group(),
            )
            log_outputs = list(chain.from_iterable(log_outputs))
            logger.info("gather over.")

        with metrics.aggregate() as agg:
            task.reduce_metrics(log_outputs, criterion)
            log_output = agg.get_smoothed_values()

        progress.print(log_output, tag=subset, step=i)
    if saved_cfg.model.base_layers >= 0:
        if distributed_utils.is_master(cfg.distributed_training):
            cnt = 0
            # print(model.decoder.layers)
            for layer in model.decoder.layers:
                if hasattr(layer, "expert_network"):
                    cnt += 1
                    each_expert_count = layer.each_expert_count.tolist()
                    total = sum(each_expert_count)
                    print(layer.each_expert_count)
                    logger.info("Base layer {}:".format(cnt))
                    for i, count in enumerate(each_expert_count):
                        logger.info("\texpert {}: {}".format(i, count / total))
        if saved_cfg.task.add_pos:
            cnt = 0
            header = ""
            for i, pos_name in enumerate(task.pos_dictionary.indices):
                header += f",{pos_name}"
            for layer in model.decoder.layers:
                if hasattr(layer, "expert_network"):
                    cnt += 1
                    pos_tensor = distributed_utils.all_gather(layer.pos_count, distributed_utils.get_data_parallel_group(), return_tensor=True)
                    # print(pos_tensor.shape)
                    pos_sum = torch.sum(pos_tensor, 0, True)
                    pos_rate_list = (pos_tensor / pos_sum).tolist()
                    logger.info("Base layer {}:".format(cnt))
                    logger.info(header)
                    for expert_id, t in enumerate(pos_rate_list):
                        log_info = "expert {}".format(expert_id)
                        for i, count in enumerate(t):
                            log_info += ",{:.3f}".format(count)
                        logger.info(log_info)


def cli_main():
    parser = options.get_validation_parser()
    args = options.parse_args_and_arch(parser)
    # only override args that are explicitly given on the command line
    override_parser = options.get_validation_parser()
    override_args = options.parse_args_and_arch(
        override_parser, suppress_defaults=True
    )

    distributed_utils.call_main(
        convert_namespace_to_omegaconf(args), main, override_args=override_args
    )


if __name__ == "__main__":
    cli_main()
