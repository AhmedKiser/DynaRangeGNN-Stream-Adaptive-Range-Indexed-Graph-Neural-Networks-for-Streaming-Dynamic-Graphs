from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Set, Tuple


def edge_key(u: int, v: int) -> Tuple[int, int]:
    """Canonical undirected edge key."""
    if u == v:
        raise ValueError("Self-loops are not used in this first prototype.")
    return (u, v) if u < v else (v, u)


@dataclass
class GraphStore:
    """Simple undirected graph store: adjacency sets + edge hash."""

    num_nodes: int
    adj: List[Set[int]]
    edge_hash: Set[Tuple[int, int]]

    @classmethod
    def from_edges(cls, num_nodes: int, edges: Iterable[Tuple[int, int]]) -> "GraphStore":
        adj = [set() for _ in range(num_nodes)]
        edge_hash: set[Tuple[int, int]] = set()
        graph = cls(num_nodes=num_nodes, adj=adj, edge_hash=edge_hash)
        for u, v in edges:
            graph.add_edge(u, v)
        return graph

    def copy(self) -> "GraphStore":
        return GraphStore(
            num_nodes=self.num_nodes,
            adj=[set(nei) for nei in self.adj],
            edge_hash=set(self.edge_hash),
        )

    def _validate_node(self, u: int) -> None:
        if not (0 <= u < self.num_nodes):
            raise ValueError(f"Invalid node {u}; graph has {self.num_nodes} nodes.")

    def has_edge(self, u: int, v: int) -> bool:
        self._validate_node(u)
        self._validate_node(v)
        return edge_key(u, v) in self.edge_hash

    def add_edge(self, u: int, v: int) -> bool:
        self._validate_node(u)
        self._validate_node(v)
        key = edge_key(u, v)
        if key in self.edge_hash:
            return False
        self.edge_hash.add(key)
        self.adj[u].add(v)
        self.adj[v].add(u)
        return True

    def delete_edge(self, u: int, v: int) -> bool:
        self._validate_node(u)
        self._validate_node(v)
        key = edge_key(u, v)
        if key not in self.edge_hash:
            return False
        self.edge_hash.remove(key)
        self.adj[u].remove(v)
        self.adj[v].remove(u)
        return True

    def degree(self, u: int) -> int:
        self._validate_node(u)
        return len(self.adj[u])

    def degrees_list(self) -> list[int]:
        return [len(nei) for nei in self.adj]

    def neighbors(self, u: int) -> Set[int]:
        self._validate_node(u)
        return self.adj[u]

    def check_consistency(self) -> None:
        for u in range(self.num_nodes):
            for v in self.adj[u]:
                assert u in self.adj[v], f"Asymmetric adjacency: {u}-{v}"
                assert edge_key(u, v) in self.edge_hash, f"Missing edge hash for {u}-{v}"
        for u, v in self.edge_hash:
            assert v in self.adj[u], f"Missing adjacency {u}->{v}"
            assert u in self.adj[v], f"Missing adjacency {v}->{u}"
