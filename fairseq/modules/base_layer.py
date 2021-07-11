# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch
import sys
from fairseq import utils
from fairseq.distributed import utils as distributed_utils
from fairseq.modules.layer_norm import LayerNorm

def soft_split(size, num_to_split):
    """
    caculate the number of features each bock contains
    """
    min_size = size // num_to_split
    left = size % num_to_split
    splits = []
    for i in range(left):
        splits.append(min_size + 1)
    for i in range(left, num_to_split):
        splits.append(min_size)
    return splits

def pend_for_assign(features, num_expert):
    """
    balanced assignment require the tokens can be divided by experts
    """
    size = features.shape[0]
    if size % num_expert != 0:
        left = num_expert - (size % num_expert)
        to_append = torch.rand((left,) + features.shape[1:], dtype=features.dtype, device=features.device)
        features = torch.cat([features, to_append])
    return features, size


class BaseLayer(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.num_workers = distributed_utils.get_data_parallel_world_size()
        self.base_sublayers = args.base_sublayers  # how many experts in each worker
        embed_dim = args.decoder_embed_dim if hasattr(args, "decoder_embed_dim") else args.encoder_embed_dim
        expert_centroids = torch.empty(self.num_workers * self.base_sublayers, embed_dim)
        torch.nn.init.orthogonal_(expert_centroids, gain=0.1)
        self.register_parameter("expert_centroids", torch.nn.Parameter(expert_centroids))
        self.expert_network = nn.ModuleList(([BaseSublayer(args) for _ in range(args.base_sublayers)]))
        self.expert_id = distributed_utils.get_data_parallel_rank() * self.base_sublayers
        self.shuffle = args.base_shuffle
        self.cpp = self.load_assignment()
        # self.cnt = 0

        # Add a special attribute to the expert parameters, so we know not to sync their gradients
        for param in self.expert_network.parameters():
            param.expert = True

    def forward(self, input_features, *args, **kwargs):
        features = input_features.reshape(-1, input_features.size(-1))
        is_training = input_features.requires_grad

        if self.shuffle and is_training:
            # get input and output splits (when sentence length is not always same)
            input_splits_list = soft_split(features.size(0), self.num_workers)
            input_splits_tensor = torch.tensor(input_splits_list, dtype=torch.int64, device=features.device)
            output_splits_list = All2All.apply(input_splits_tensor).tolist()

            # Send each token to a random worker, to break correlations within the batch
            shuffle_sort = torch.randperm(features.size(0), device=features.device)
            features = All2All.apply(features[shuffle_sort], input_splits_list, output_splits_list)
        if is_training:
            # pend for balanced assignment
            features, init_size = pend_for_assign(features, self.num_workers * self.base_sublayers)

        with torch.no_grad():
            # Compute similarity of each token to each expert, for routing
            token_expert_affinities = features.matmul(self.expert_centroids.transpose(0, 1))

        # Compute which token goes to which expert
        # print("features shape", features.shape)
        sort_by_expert, input_splits, output_splits, expert_counts = self.balanced_assignment(token_expert_affinities) \
            if is_training else self.greedy_assignment(token_expert_affinities)
        # print("expert counts", expert_counts)

        # Swap these tokens for the right ones for our expert
        routed_features = All2All.apply(features[sort_by_expert], output_splits, input_splits)

        if routed_features.size(0) > 0:
            # Mix in the expert network based on how appropriate it is for these tokens
            start = 0
            expert_output = []
            for expert_id in range(self.expert_id, self.expert_id + self.base_sublayers):
                expert_centroids = self.expert_centroids[expert_id]  # bug here
                # print(start + expert_counts[expert_id])
                expert_features = routed_features[start: start + expert_counts[expert_id]]
                alpha = torch.sigmoid(expert_features.mv(expert_centroids)).unsqueeze(1)
                expert_features = alpha * self.expert_network[expert_id - self.expert_id](expert_features) \
                     + (1 - alpha) * expert_features
                expert_output.append(expert_features)
                try:
                    start = start + expert_counts[expert_id].item()
                except RuntimeError as e:
                    print(start)
                    print(expert_counts[expert_id])
                    quit()
            try:
                routed_features = torch.cat(expert_output, 0)
            except RuntimeError as e:
                for e in expert_output:
                    print(e.shape)
                quit()
        # Return to original worker and ordering
        result = All2All.apply(routed_features, input_splits, output_splits)[self.inverse_sort(sort_by_expert)]
        # print(result.shape)
        # quit()
        if is_training:
            result = result[:init_size]

        if self.shuffle and is_training:
            # Undo shuffling
            result = All2All.apply(result, output_splits_list, input_splits_list)[self.inverse_sort(shuffle_sort)]

        # Return additional Nones for compatibility with TransformerDecoderLayer
        return result.view(input_features.size()), None, None

    def inverse_sort(self, order):
        # Creates an index that undoes a sort: xs==xs[order][inverse_sort(order)]
        return torch.empty_like(order).scatter_(0, order, torch.arange(0, order.size(0), device=order.device))

    def balanced_assignment(self, scores):
        ok = scores.isfinite()
        if not ok.all():
            # NaNs here can break the assignment algorithm
            scores[~ok] = scores[ok].min()
        # get input and output splits (when sentence length is not always same)
        input_splits_list = soft_split(scores.size(0), self.num_workers)
        input_splits_tensor = torch.tensor(input_splits_list, dtype=torch.int64, device=scores.device)
        output_splits_list = All2All.apply(input_splits_tensor).tolist()
        # expert_counts = torch.zeros((self.num_workers * self.base_sublayers,), dtype=torch.long, device=scores.device)
        expert_counts = soft_split(scores.size(0), self.num_workers * self.base_sublayers)
        expert_counts = torch.tensor(expert_counts, dtype=torch.int64, device=scores.device)
        torch.distributed.all_reduce(expert_counts)
        return self.cpp.balanced_assignment(scores), output_splits_list, input_splits_list, expert_counts

    # Assigns each token to the top k experts
    def greedy_assignment(self, scores, k=1):
        token_to_workers = torch.topk(scores, dim=1, k=k, largest=True).indices.view(-1)
        token_to_workers, sort_ordering = torch.sort(token_to_workers)
        worker2token = sort_ordering // k

        # Find how many tokens in each expert
        expert_counts = torch.zeros((self.num_workers * self.base_sublayers,), dtype=torch.long, device=scores.device)
        experts, counts = torch.unique_consecutive(token_to_workers, return_counts=True)
        expert_counts[experts] = counts
        torch.distributed.all_reduce(expert_counts)

        # Find how many tokens we're sending to each other worker (being careful for sending 0 tokens to some workers)
        token_to_workers = token_to_workers // self.base_sublayers
        output_splits = torch.zeros((self.num_workers,), dtype=torch.long, device=scores.device)
        workers, counts = torch.unique_consecutive(token_to_workers, return_counts=True)
        output_splits[workers] = counts
        # Tell other workers how many tokens to expect from us
        input_splits = All2All.apply(output_splits)
        return worker2token, input_splits.tolist(), output_splits.tolist(), expert_counts

    def load_assignment(self):
        try:
            from fairseq import libbase

            return libbase

        except ImportError as e:
            sys.stderr.write(
                "ERROR: missing libbase. run `python setup.py build_ext --inplace`\n"
            )
            raise e


class BaseSublayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu') or "relu"
        )
        embed_dim = args.decoder_embed_dim if hasattr(args, "decoder_embed_dim") else args.encoder_embed_dim
        ffn_embed_dim = args.decoder_ffn_embed_dim if hasattr(args, "decoder_ffn_embed_dim") else args.encoder_ffn_embed_dim
        self.norm = LayerNorm(embed_dim, export=False)
        self.ff1 = torch.nn.Linear(embed_dim, ffn_embed_dim)
        self.ff2 = torch.nn.Linear(ffn_embed_dim, embed_dim)
        self.ff2.weight.data.zero_()

    def forward(self, xs):
        return xs + self.ff2(self.activation_fn(self.ff1(self.norm(xs))))


# Wraps torch.distributed.all_to_all_single as a function that supports autograd
class All2All(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xs, input_splits=None, output_splits=None):
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits

        ys = torch.empty_like(xs) if output_splits is None else \
            xs.new_empty(size=[sum(output_splits)] + list(xs.size()[1:]))
        torch.distributed.all_to_all_single(ys, xs, output_split_sizes=output_splits, input_split_sizes=input_splits)
        return ys

    @staticmethod
    def backward(ctx, grad_output):
        result = torch.empty_like(grad_output) if ctx.input_splits is None else \
            grad_output.new_empty(size=[sum(ctx.input_splits)] + list(grad_output.size()[1:]))
        torch.distributed.all_to_all_single(result, grad_output,
                                            output_split_sizes=ctx.input_splits, input_split_sizes=ctx.output_splits)
        return result, None, None
