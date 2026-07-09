from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from .events import EdgeAdd, EdgeDelete, FeatureUpdate
from .graph_store import GraphStore
from .model import GraphSAGECache, TwoLayerGraphSAGEMean, neighbor_sum, degree_tensor


@dataclass
class UpdateStats:
    feature_nodes: int
    added_edges: int
    removed_edges: int
    layer1_dirty: int
    layer2_dirty: int
    max_abs_error_vs_full: float | None = None


class DynaDeltaEngine:
    """
    Correctness-first exact local update engine for a 2-layer GraphSAGE-Mean model.

    The engine rebuilds M[layer][v] from the final graph for dirty nodes only.
    This is less optimized than pure delta arithmetic, but much safer for the first prototype.
    """

    def __init__(self, model: TwoLayerGraphSAGEMean, graph: GraphStore, x: torch.Tensor) -> None:
        self.model = model
        self.graph = graph
        self.x = x.clone()
        self.cache = model.forward_with_cache(self.x, self.graph)

    def full_recompute(self) -> GraphSAGECache:
        return self.model.forward_with_cache(self.x, self.graph)

    def process_event_batch(self, events: Iterable[object]) -> UpdateStats:
        old_x = self.x.clone()
        old_edges = set(self.graph.edge_hash)

        # 1. Apply events sequentially to obtain final graph and final features.
        for event in events:
            if isinstance(event, FeatureUpdate):
                self.graph._validate_node(event.node)
                if event.new_value.shape != self.x[event.node].shape:
                    raise ValueError(
                        f"FeatureUpdate shape mismatch for node {event.node}: "
                        f"{event.new_value.shape} vs {self.x[event.node].shape}"
                    )
                self.x[event.node] = event.new_value.to(dtype=self.x.dtype, device=self.x.device)

            elif isinstance(event, EdgeAdd):
                self.graph.add_edge(event.u, event.v)

            elif isinstance(event, EdgeDelete):
                self.graph.delete_edge(event.u, event.v)

            else:
                raise TypeError(f"Unknown event type: {type(event)}")

        self.graph.check_consistency()
        new_edges = set(self.graph.edge_hash)

        added_edges = new_edges - old_edges
        removed_edges = old_edges - new_edges
        topology_endpoints = {u for e in added_edges | removed_edges for u in e}

        # 2. Update H[0] and detect changed feature nodes.
        feature_nodes: set[int] = set()
        for v in range(self.graph.num_nodes):
            if torch.max(torch.abs(self.x[v] - old_x[v])).item() > 0:
                feature_nodes.add(v)
                self.cache.H[0][v] = self.x[v]

        # 3. Build layer-1 dirty set.
        layer1_dirty: set[int] = set(feature_nodes) | set(topology_endpoints)
        for src in feature_nodes:
            layer1_dirty.update(self.graph.adj[src])

        # 4. Recompute layer 1 for dirty nodes.
        delta_h1 = self.model.recompute_nodes_layer1(self.cache, self.graph, layer1_dirty)

        changed_h1_nodes = {
            v for v, delta in delta_h1.items()
            if torch.max(torch.abs(delta)).item() > 0
        }

        # 5. Build layer-2 dirty set.
        layer2_dirty: set[int] = set(changed_h1_nodes) | set(topology_endpoints)
        for src in changed_h1_nodes:
            layer2_dirty.update(self.graph.adj[src])

        # 6. Recompute layer 2 for dirty nodes.
        self.model.recompute_nodes_layer2(self.cache, self.graph, layer2_dirty)

        # 7. Refresh degree cache.
        self.cache.degree = degree_tensor(self.graph, self.x.dtype, self.x.device)

        return UpdateStats(
            feature_nodes=len(feature_nodes),
            added_edges=len(added_edges),
            removed_edges=len(removed_edges),
            layer1_dirty=len(layer1_dirty),
            layer2_dirty=len(layer2_dirty),
        )

    def assert_cache_invariants(self, atol: float = 1e-10) -> None:
        expected_m1 = neighbor_sum(self.cache.H[0], self.graph)
        expected_m2 = neighbor_sum(self.cache.H[1], self.graph)

        torch.testing.assert_close(self.cache.M[1], expected_m1, rtol=0.0, atol=atol)
        torch.testing.assert_close(self.cache.M[2], expected_m2, rtol=0.0, atol=atol)

        expected_degree = degree_tensor(self.graph, self.x.dtype, self.x.device)
        torch.testing.assert_close(self.cache.degree, expected_degree, rtol=0.0, atol=0.0)

    def compare_to_full_recompute(self, atol: float = 1e-6) -> float:
        full = self.full_recompute()
        max_error = torch.max(torch.abs(self.cache.H[2] - full.H[2])).item()

        torch.testing.assert_close(self.cache.H[1], full.H[1], rtol=0.0, atol=atol)
        torch.testing.assert_close(self.cache.H[2], full.H[2], rtol=0.0, atol=atol)
        torch.testing.assert_close(self.cache.M[1], full.M[1], rtol=0.0, atol=atol)
        torch.testing.assert_close(self.cache.M[2], full.M[2], rtol=0.0, atol=atol)

        return max_error
