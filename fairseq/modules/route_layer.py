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
import fairseq.utils as utils


class RouteLayer(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.num_workers = distributed_utils.get_data_parallel_world_size()
        self.isdecoder = hasattr(args, "decoder_embed_dim")
        embed_dim = args.decoder_embed_dim if hasattr(args, "decoder_embed_dim") else args.encoder_embed_dim
        expert_centroids = torch.empty(self.num_workers, embed_dim)
        torch.nn.init.orthogonal_(expert_centroids, gain=0.1)
        self.register_parameter("expert_centroids", torch.nn.Parameter(expert_centroids))
        self.expert_network = nn.Sequential(*([BaseSublayer(args) for _ in range(args.base_sublayers)]))
        self.expert_id = distributed_utils.get_data_parallel_rank()

        # Add a special attribute to the expert parameters, so we know not to sync their gradients
        for param in self.expert_network.parameters():
            param.expert = True

    def forward(self, input_features, *args, **kwargs):
        features = input_features.reshape(-1, input_features.size(-1))
        is_training = input_features.requires_grad
        pos = kwargs.get('pos', None)
        if pos is not None:
            pos = pos.reshape(-1, 1)

        with torch.no_grad():
            # Compute which token goes to which expert, no need to have grad
            if is_training:
                sort_by_expert, input_splits, output_splits = self.knowledge_assignment(pos)
            else:
                token_expert_affinities = features.matmul(self.expert_centroids.transpose(0, 1))
                sort_by_expert, input_splits, output_splits = self.greedy_assignment(token_expert_affinities)
        # Swap these tokens for the right ones for our expert
        routed_features = All2All.apply(features[sort_by_expert], output_splits, input_splits)

        if routed_features.size(0) > 0:
            # Mix in the expert network based on how appropriate it is for these tokens
            alpha = torch.sigmoid(routed_features.mv(self.expert_centroids[self.expert_id])).unsqueeze(1)
            routed_features = alpha * self.expert_network(routed_features) + (1 - alpha) * routed_features
            # routed_features = self.expert_network(routed_features)
        # Return to original worker and ordering
        result = All2All.apply(routed_features, input_splits, output_splits)[self.inverse_sort(sort_by_expert)]

        if self.isdecoder:
            # Return additional Nones for compatibility with TransformerDecoderLayer
            return result.view(input_features.size()), None, None
        else:
            return result.view(input_features.size())

    def inverse_sort(self, order):
        # Creates an index that undoes a sort: xs==xs[order][inverse_sort(order)]
        return torch.empty_like(order).scatter_(0, order, torch.arange(0, order.size(0), device=order.device))

    def knowledge_assignment(self, knowledges):
        sorted_knowledges, indices = torch.sort(knowledges, dim=0)
        indices = indices.view(-1)
        expert_index, expert_count = torch.unique_consecutive(sorted_knowledges, return_counts=True)
        input_splits_tensor = torch.zeros((self.num_workers,), dtype=torch.long, device=knowledges.device)
        input_splits_tensor[expert_index] = expert_count
        output_splits_list = All2All.apply(input_splits_tensor).tolist()
        input_splits_list = input_splits_tensor.tolist()
        return indices, output_splits_list, input_splits_list

    # Assigns each token to the top k experts
    def greedy_assignment(self, scores, k=1):
        token_to_workers = torch.topk(scores, dim=1, k=k, largest=True).indices.view(-1)
        token_to_workers, sort_ordering = torch.sort(token_to_workers)
        worker2token = sort_ordering // k

        # Find how many tokens we're sending to each other worker (being careful for sending 0 tokens to some workers)
        output_splits = torch.zeros((self.num_workers,), dtype=torch.long, device=scores.device)
        workers, counts = torch.unique_consecutive(token_to_workers, return_counts=True)
        output_splits[workers] = counts
        # Tell other workers how many tokens to expect from us
        input_splits = All2All.apply(output_splits)
        return worker2token, input_splits.tolist(), output_splits.tolist()


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
