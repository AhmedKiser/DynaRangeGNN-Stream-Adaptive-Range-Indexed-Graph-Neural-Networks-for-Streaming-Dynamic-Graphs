from __future__ import annotations

import torch
from torch import Tensor

from .graph_store import GraphStore


def classroom_graph() -> tuple[GraphStore, Tensor]:
    """
    Graph:
          B ----- D
          |
    A ----- C ----- E ----- F

    G ----- H ----- I
    """
    edges = [
        (0, 1), (0, 2), (1, 3),
        (2, 4), (4, 5),
        (6, 7), (7, 8)
    ]
    graph = GraphStore.from_edges(num_nodes=9, edges=edges)

    x = torch.tensor([
        [0.70, 0.10, 0.20],  # A = 0
        [0.82, 0.20, 0.10],  # B = 1
        [0.65, 0.30, 0.40],  # C = 2
        [0.90, 0.10, 0.30],  # D = 3
        [0.74, 0.25, 0.15],  # E = 4
        [0.55, 0.20, 0.35],  # F = 5
        [0.30, 0.10, 0.20],  # G = 6
        [0.45, 0.30, 0.10],  # H = 7
        [0.60, 0.15, 0.25],  # I = 8
    ], dtype=torch.float64)

    return graph, x


def star_graph(num_leaves: int = 1000, feature_dim: int = 3) -> tuple[GraphStore, Tensor]:
    edges = [(0, i) for i in range(1, num_leaves + 1)]
    graph = GraphStore.from_edges(num_nodes=num_leaves + 1, edges=edges)

    torch.manual_seed(123)
    x = torch.randn(num_leaves + 1, feature_dim, dtype=torch.float64)

    return graph, x
