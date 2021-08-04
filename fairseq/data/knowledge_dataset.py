# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import json
import random

from . import FairseqDataset, data_utils


def collate(samples, pad_idx, eos_idx, fixed_pad_length=None, pad_to_bsz=None, default_expert=[0, 0]):
    if len(samples) == 0:
        return {}

    def merge(key, is_list=False, current_pad_idx=pad_idx, current_eos_idx=eos_idx):
        if is_list:
            res = []
            for i in range(len(samples[0][key])):
                res.append(
                    data_utils.collate_tokens(
                        [s[key][i] for s in samples],
                        current_pad_idx,
                        current_eos_idx,
                        left_pad=False,
                        pad_to_length=fixed_pad_length,
                        pad_to_bsz=pad_to_bsz,
                    )
                )
            return res
        else:
            return data_utils.collate_tokens(
                [s[key] for s in samples],
                current_pad_idx,
                current_eos_idx,
                left_pad=False,
                pad_to_length=fixed_pad_length,
                pad_to_bsz=pad_to_bsz,
            )

    src_tokens = merge("source")
    expert = merge("expert", current_pad_idx=default_expert[0], \
        current_eos_idx=default_expert[0])

    return {
        "id": torch.LongTensor([s["id"] for s in samples]),
        "nsentences": len(samples),
        "ntokens": sum(len(s["source"]) for s in samples),
        "net_input": {
            "src_tokens": src_tokens,
            "expert": expert,
            "src_lengths": torch.LongTensor([s["source"].numel() for s in samples]),
        },
    }


class KnowledgeDataset(FairseqDataset):
    """
    A wrapper around torch.utils.data.Dataset for monolingual data.

    Args:
        dataset (torch.utils.data.Dataset): dataset to wrap
        sizes (List[int]): sentence lengths
        vocab (~fairseq.data.Dictionary): vocabulary
        shuffle (bool, optional): shuffle the elements before batching
            (default: True).
    """

    def __init__(
        self,
        dataset,
        sizes,
        src_vocab,
        pos_expert_map,
        tgt_vocab=None,
        add_eos_for_other_targets=False,
        shuffle=False,
        targets=None,
        add_bos_token=False,
        fixed_pad_length=None,
        pad_to_bsz=None,
        src_lang_idx=None,
        tgt_lang_idx=None,
    ):
        self.dataset = dataset
        self.sizes = np.array(sizes)
        self.vocab = src_vocab
        self.tgt_vocab = tgt_vocab or src_vocab
        self.add_eos_for_other_targets = add_eos_for_other_targets
        self.shuffle = shuffle
        self.add_bos_token = add_bos_token
        self.fixed_pad_length = fixed_pad_length
        self.pad_to_bsz = pad_to_bsz
        self.src_lang_idx = src_lang_idx
        self.tgt_lang_idx = tgt_lang_idx

        assert targets is None or all(
            t in {"self", "future", "past"} for t in targets
        ), "targets must be none or one of 'self', 'future', 'past'"
        if targets is not None and len(targets) == 0:
            targets = None
        self.targets = targets
        self.default_expert = [0, 0]
        self.pos_expert_map = self.load_pos_expert_map(pos_expert_map, self.default_expert)
        # print(self.pos_expert_map)
    
    def load_pos_expert_map(self, pos_expert_map_file, default_expert=[0, 0]):
        with open(pos_expert_map_file, 'r') as f:
            pos_expert_map = json.load(f)
        for key in self.vocab.indices:
            if key not in pos_expert_map:
                pos_expert_map[key] = default_expert
        pos_id_to_expert_map = {}
        for key in pos_expert_map:
            key_id = self.vocab.indices[key]
            pos_id_to_expert_map[key_id] = pos_expert_map[key]
        return pos_id_to_expert_map

    def __getitem__(self, index):
        if self.targets is not None:
            # *future_target* is the original sentence
            # *source* is shifted right by 1 (maybe left-padded with eos)
            # *past_target* is shifted right by 2 (left-padded as needed)
            #
            # Left-to-right language models should condition on *source* and
            # predict *future_target*.
            # Right-to-left language models should condition on *source* and
            # predict *past_target*.
            source, future_target, past_target = self.dataset[index]
            source, target = self._make_source_target(
                source, future_target, past_target
            )
        else:
            source = self.dataset[index]
            target = None
        source = self._maybe_add_bos(source)
        expert_id = list(map(self.get_expert_id, source.tolist()))
        expert_id = torch.tensor(expert_id, dtype=source.dtype, device=source.device)
        return {"id": index, "source": source, "expert": expert_id}

    def get_expert_id(self, pos_id):
        return random.randint(*self.pos_expert_map[pos_id])
    
    def __len__(self):
        return len(self.dataset)
    
    def _make_source_target(self, source, future_target, past_target):
        if self.targets is not None:
            target = []

            if (
                self.add_eos_for_other_targets
                and (("self" in self.targets) or ("past" in self.targets))
                and source[-1] != self.vocab.eos()
            ):
                # append eos at the end of source
                source = torch.cat([source, source.new([self.vocab.eos()])])

                if "future" in self.targets:
                    future_target = torch.cat(
                        [future_target, future_target.new([self.vocab.pad()])]
                    )
                if "past" in self.targets:
                    # first token is before the start of sentence which is only used in "none" break mode when
                    # add_eos_for_other_targets is False
                    past_target = torch.cat(
                        [
                            past_target.new([self.vocab.pad()]),
                            past_target[1:],
                            source[-2, None],
                        ]
                    )

            for t in self.targets:
                if t == "self":
                    target.append(source)
                elif t == "future":
                    target.append(future_target)
                elif t == "past":
                    target.append(past_target)
                else:
                    raise Exception("invalid target " + t)

            if len(target) == 1:
                target = target[0]
        else:
            target = future_target

        return source, self._filter_vocab(target)

    def _maybe_add_bos(self, source):
        if self.add_bos_token:
            source = torch.cat([source.new([self.vocab.bos()]), source])
        return source

    def num_tokens_vec(self, indices):
        """Return the number of tokens for a set of positions defined by indices.
        This value is used to enforce ``--max-tokens`` during batching."""
        return self.sizes[indices]
    
    def _filter_vocab(self, target):
        if len(self.tgt_vocab) != len(self.vocab):

            def _filter(target):
                mask = target.ge(len(self.tgt_vocab))
                if mask.any():
                    target[mask] = self.tgt_vocab.unk()
                return target

            if isinstance(target, list):
                return [_filter(t) for t in target]
            return _filter(target)
        return target

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the right.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the right.
        """
        return collate(
            samples,
            self.vocab.pad(),
            self.vocab.eos(),
            self.fixed_pad_length,
            self.pad_to_bsz,
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.sizes[index]

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.sizes[index]

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]
        order.append(self.sizes)
        return np.lexsort(order)

    @property
    def supports_prefetch(self):
        return getattr(self.dataset, "supports_prefetch", False)

    def prefetch(self, indices):
        self.dataset.prefetch(indices)
