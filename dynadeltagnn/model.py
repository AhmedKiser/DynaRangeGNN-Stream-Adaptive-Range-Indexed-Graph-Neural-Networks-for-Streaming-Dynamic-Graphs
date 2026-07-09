from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor, nn

from .graph_store import GraphStore


torch.set_default_dtype(torch.float64)


def degree_tensor(graph: GraphStore, dtype: torch.dtype, device: torch.device) -> Tensor:
    return torch.tensor(graph.degrees_list(), dtype=dtype, device=device)


def neighbor_sum(h: Tensor, graph: GraphStore) -> Tensor:
    """
    Computes M[v] = sum_{u in N(v)} h[u].
    h shape: [num_nodes, feature_dim]
    """
    result = torch.zeros_like(h)
    for v in range(graph.num_nodes):
        if graph.adj[v]:
            idx = torch.tensor(sorted(graph.adj[v]), dtype=torch.long, device=h.device)
            result[v] = h[idx].sum(dim=0)
    return result


class MeanSAGELayer(nn.Module):
    """
    h_v^l = ReLU(W_self h_v^{l-1} + W_neighbor mean_{u in N(v)} h_u^{l-1} + b)
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.self_linear = nn.Linear(in_dim, out_dim, bias=True)
        self.neighbor_linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, h_prev: Tensor, msg_sum: Tensor, degree: Tensor) -> Tensor:
        degree_safe = degree.clamp_min(1.0).unsqueeze(1)
        mean = msg_sum / degree_safe
        return torch.relu(self.self_linear(h_prev) + self.neighbor_linear(mean))


@dataclass
class GraphSAGECache:
    H: List[Tensor]                 # H[0], H[1], H[2]
    M: List[Optional[Tensor]]       # M[0]=None, M[1], M[2]
    degree: Tensor


class TwoLayerGraphSAGEMean(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layer1 = MeanSAGELayer(input_dim, hidden_dim)
        self.layer2 = MeanSAGELayer(hidden_dim, output_dim)

    def forward_with_cache(self, x: Tensor, graph: GraphStore) -> GraphSAGECache:
        graph.check_consistency()
        degree = degree_tensor(graph, dtype=x.dtype, device=x.device)

        h0 = x.clone()
        m1 = neighbor_sum(h0, graph)
        h1 = self.layer1(h0, m1, degree)

        m2 = neighbor_sum(h1, graph)
        h2 = self.layer2(h1, m2, degree)

        return GraphSAGECache(H=[h0, h1, h2], M=[None, m1, m2], degree=degree)

    def recompute_nodes_layer1(self, cache: GraphSAGECache, graph: GraphStore, nodes: set[int]) -> dict[int, Tensor]:
        if not nodes:
            return {}

        cache.degree = degree_tensor(graph, cache.H[0].dtype, cache.H[0].device)

        delta: dict[int, Tensor] = {}
        for v in sorted(nodes):
            old = cache.H[1][v].clone()
            msg = torch.zeros_like(cache.M[1][v])
            if graph.adj[v]:
                idx = torch.tensor(sorted(graph.adj[v]), dtype=torch.long, device=cache.H[0].device)
                msg = cache.H[0][idx].sum(dim=0)

            cache.M[1][v] = msg
            degree_v = cache.degree[v].view(1)
            new = self.layer1(cache.H[0][v:v+1], msg.view(1, -1), degree_v).squeeze(0)
            cache.H[1][v] = new
            delta[v] = new - old

        return delta

    def recompute_nodes_layer2(self, cache: GraphSAGECache, graph: GraphStore, nodes: set[int]) -> dict[int, Tensor]:
        if not nodes:
            return {}

        cache.degree = degree_tensor(graph, cache.H[0].dtype, cache.H[0].device)

        delta: dict[int, Tensor] = {}
        for v in sorted(nodes):
            old = cache.H[2][v].clone()
            msg = torch.zeros_like(cache.M[2][v])
            if graph.adj[v]:
                idx = torch.tensor(sorted(graph.adj[v]), dtype=torch.long, device=cache.H[1].device)
                msg = cache.H[1][idx].sum(dim=0)

            cache.M[2][v] = msg
            degree_v = cache.degree[v].view(1)
            new = self.layer2(cache.H[1][v:v+1], msg.view(1, -1), degree_v).squeeze(0)
            cache.H[2][v] = new
            delta[v] = new - old

        return delta
