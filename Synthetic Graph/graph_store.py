"""
graph_store.py
--------------
Phase 1 building block: the dynamic graph store.

Holds the graph's topology (adjacency lists + an edge hash table for O(1)
existence checks) and every node's current feature vector. Exposes the
three streaming event types defined in the proposal:

    1. feature update   -> set_feature(node, new_vector)
    2. edge insertion    -> add_edge(u, v)
    3. edge deletion      -> remove_edge(u, v)

Node set is fixed at construction time (matches Phase 1-2 scope: dynamic
*attributes and edges*, not dynamic node insertion/removal).
"""

import torch


class DynamicGraphStore:
    def __init__(self, node_ids, feature_dim):
        self.node_ids = list(node_ids)
        self.index_of = {nid: i for i, nid in enumerate(self.node_ids)}
        self.feature_dim = feature_dim

        # Adjacency list: node_id -> set of neighbor node_ids
        self.adjacency = {nid: set() for nid in self.node_ids}

        # Edge hash table: O(1) "does this edge exist?" checks
        self.edge_hash = set()  # set of frozenset({u, v})

        # Feature matrix, one row per node
        self.features = torch.zeros(len(self.node_ids), feature_dim, dtype=torch.float32)

    # ---------------- topology events ----------------
    def add_edge(self, u, v):
        """Returns True if a new edge was actually created."""
        key = frozenset((u, v))
        if key in self.edge_hash:
            return False
        self.edge_hash.add(key)
        self.adjacency[u].add(v)
        self.adjacency[v].add(u)
        return True

    def remove_edge(self, u, v):
        """Returns True if an existing edge was actually removed."""
        key = frozenset((u, v))
        if key not in self.edge_hash:
            return False
        self.edge_hash.discard(key)
        self.adjacency[u].discard(v)
        self.adjacency[v].discard(u)
        return True

    def has_edge(self, u, v):
        return frozenset((u, v)) in self.edge_hash

    def degree(self, node):
        return len(self.adjacency[node])

    def neighbors(self, node):
        return self.adjacency[node]

    # ---------------- attribute events ----------------
    def set_feature(self, node, vector):
        idx = self.index_of[node]
        self.features[idx] = torch.as_tensor(vector, dtype=torch.float32)

    def get_feature(self, node):
        return self.features[self.index_of[node]]

    def feature_matrix(self):
        return self.features

    # ---------------- misc ----------------
    def __repr__(self):
        n_edges = len(self.edge_hash)
        return f"DynamicGraphStore(nodes={len(self.node_ids)}, edges={n_edges}, feature_dim={self.feature_dim})"
