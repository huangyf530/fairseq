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
import copy

import torch
from fairseq import checkpoint_utils, distributed_utils, options, utils, tasks
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import metrics, progress_bar
from omegaconf import DictConfig
from fairseq.data.encoders.gpt2_bpe import GPT2BPE, get_encoder
from fairseq.data import (
    AppendTokenDataset,
    Dictionary,
    IdDataset,
    LMContextWindowDataset,
    MonolingualDataset,
    NestedDictionaryDataset,
    NumelDataset,
    PadDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TokenBlockDataset,
    TruncatedDictionary,
    data_utils,
)
from fairseq.data.indexed_dataset import get_available_dataset_impl


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("test-pos")

encoder = get_encoder("gpt2_bpe/encoder.json", "gpt2_bpe/vocab.bpe")

def show_sent_pos(src_tokens, poses, task, target=None):
    tokens = task.dictionary.string(src_tokens).split(' ')
    if target is not None:
        targets = task.dictionary.string(target).split(' ')
        print(len(tokens))
        print(len(targets))
        ids = []
        for index, t in enumerate(tokens):
            try:
                ids.append(int(t))
            except ValueError as e:
                print(t)
                print(targets[index])
                ids.append(int(targets[index]))
        tokens = ids
    print(encoder.decode(map(int, tokens)))
    print(task.pos_dictionary.string(poses))


def main(cfg: DictConfig, override_args=None):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)
    utils.import_user_module(cfg.common)
    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(cfg.task)
    for subset in cfg.dataset.valid_subset.split(","):
        try:
            task.load_dataset(subset, combine=False, epoch=1, task_cfg=cfg.task)
            dataset = task.dataset(subset)
            print(dataset[0].keys())
            # show_sent_pos(dataset[0]['token.net_input.src_tokens'], dataset[0]['pos']['source'], task, target=dataset[0]['token.target'])
            for index, t in enumerate(dataset[100]['token.net_input.src_tokens']):
                token = task.dictionary[t]
                pos = task.pos_dictionary[dataset[100]['pos']['source'][index]]
                target = task.dictionary[dataset[100]['token.target'][index]]
                try:
                    print(encoder.decode([int(token)]), '|', pos, '|', target)
                except ValueError as e:
                    print(token, '|', pos, '|', target)
            quit()
            # show_sent_pos(dataset[0]['tokens'][])
        except KeyError:
            raise Exception("Cannot find dataset: " + subset)
        
        # Initialize data iterator
        batch_itr = task.get_batch_iterator(
            dataset=dataset,
            max_tokens=cfg.dataset.max_tokens,
            max_sentences=cfg.dataset.batch_size,
            max_positions=512,
            ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
            seed=cfg.common.seed,
            num_shards=1,
            shard_id=0,
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
        for i, sample in enumerate(progress):
            if len(sample) == 0:
                sample = _dummy_batch
            for t in [1, 2]:
                sample['token']['net_input']['pos'] = sample['pos']['net_input']['src_tokens']
                tokens = task.dictionary.string(sample['token']['net_input']['src_tokens'][t]).split(' ')
                # length = sample['net_input']['src_lengths'][t].item()
                # print(tokens[:length])
                print(encoder.decode(map(int, tokens)))
                print(task.pos_dictionary.string(sample['token']['net_input']['pos'][t]))
            quit()
    

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
