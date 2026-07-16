"""
gnn_model.py
------------
The "local GNN" referenced throughout the proposal: a K-layer
GraphSAGE-Mean encoder, implemented in PyTorch.

    h_v^(l+1) = ReLU( W^(l) . concat( h_v^(l), mean_{u in N(v)} h_u^(l) ) )

This is deliberately the same operator DynaDeltaGNN (Section 13 of the
proposal) defines exactness against: "the local update equals full
GraphSAGE-Mean recomputation over all defined neighbors."
"""

import torch
import torch.nn as nn


class GraphSAGEMeanLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim)

    def forward(self, node_ids, get_h, adjacency):
        """
        node_ids : iterable of node ids to produce an output embedding for
        get_h    : callable(node_id) -> this layer's input embedding tensor
        adjacency: dict node_id -> set of neighbor ids (current topology)
        returns  : dict node_id -> output embedding tensor
        """
        out = {}
        for v in node_ids:
            h_v = get_h(v)
            nbrs = adjacency[v]
            if nbrs:
                agg = torch.stack([get_h(u) for u in nbrs]).mean(dim=0)
            else:
                agg = torch.zeros_like(h_v)
            combined = torch.cat([h_v, agg], dim=-1)
            out[v] = torch.relu(self.linear(combined))
        return out


class GraphSAGEMean(nn.Module):
    """A fixed-depth GraphSAGE-Mean encoder (default: 2 layers, matching
    the proposal's fixed GNN depth in Step 7 of the running example)."""

    def __init__(self, in_dim, hidden_dim, num_layers=2):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList([
            GraphSAGEMeanLayer(dims[i], dims[i + 1]) for i in range(num_layers)
        ])

    @property
    def num_layers(self):
        return len(self.layers)


def full_forward(model, node_ids, features, adjacency):
    """Full forward pass through every layer, WITH autograd enabled --
    used for training. (Contrast with DynaDeltaGNN, which wraps every
    call in torch.no_grad() because it's an inference-time cache.)

    features : dict node_id -> layer-0 input tensor
    returns  : dict node_id -> final-layer embedding tensor
    """
    h = dict(features)
    for layer in model.layers:
        get_h = lambda n, cur=h: cur[n]
        h = layer.forward(node_ids, get_h, adjacency)
    return h
