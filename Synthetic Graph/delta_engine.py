"""
delta_engine.py
----------------
Phase 1 of the roadmap: "Implement DynaDeltaGNN exact local updates for
GraphSAGE-Mean." This is the core efficiency claim of the whole project
(RQ1): after a streaming event, update only the embeddings that could
possibly have changed, instead of recomputing the entire graph -- while
still being provably identical to a full recomputation (Section 13,
"Exactness Definition").

Dirty-frontier propagation rule
--------------------------------
If a node's embedding at layer L changes, exactly two kinds of nodes can
have a different embedding at layer L+1:
  (a) the node itself           (its own embedding feeds its own next layer)
  (b) every neighbor of that node (their aggregate step reads its embedding)

So: dirty_(L+1) = dirty_L  UNION  { neighbors of n : n in dirty_L }

Two event types seed this propagation differently:
  - a FEATURE update changes a node's layer-0 input directly, so the
    node itself AND its neighbors are already dirty at layer 1.
  - an EDGE event does not change any layer-0 feature; it changes the
    *aggregation structure* for u and v specifically, so only {u, v}
    are dirty at layer 1 (their neighbors only become dirty starting
    at layer 2, once u/v's own embeddings have actually changed).

Both cases are handled below; both are checked against a from-scratch
recomputation in test_correctness.py.
"""

import time
import torch


class DynaDeltaGNN:
    def __init__(self, model, graph_store, retrieval_index=None, retrieval_key_fn=None):
        self.model = model
        self.graph = graph_store
        self.retrieval_index = retrieval_index     # GlobalOrderStatisticTree or None
        self.retrieval_key_fn = retrieval_key_fn    # feature vector -> float key
        self.num_layers = model.num_layers

        # cache[0]            = raw features
        # cache[1..num_layers] = embeddings after each GraphSAGE-Mean layer
        self.cache = [dict() for _ in range(self.num_layers + 1)]
        self.last_stats = {}

    # ---------------- internal helpers ----------------
    def _recompute_layer(self, layer_idx, dirty_nodes):
        """Recompute cache[layer_idx + 1] for dirty_nodes using cache[layer_idx]."""
        if not dirty_nodes:
            return
        layer = self.model.layers[layer_idx]
        get_h = lambda n: self.cache[layer_idx][n]
        with torch.no_grad():
            out = layer.forward(dirty_nodes, get_h, self.graph.adjacency)
        self.cache[layer_idx + 1].update(out)

    def _expand(self, dirty_nodes):
        """dirty_(L+1) = dirty_L UNION neighbors(dirty_L)."""
        nxt = set(dirty_nodes)
        for n in dirty_nodes:
            nxt |= self.graph.neighbors(n)
        return nxt

    def _propagate(self, seed_dirty_at_layer1):
        """Run the recompute+expand loop for every layer, starting from a
        dirty set that is already correct for layer 1's output."""
        dirty = set(seed_dirty_at_layer1)
        recomputed_total = set()
        per_layer_counts = []
        for layer_idx in range(self.num_layers):
            self._recompute_layer(layer_idx, dirty)
            recomputed_total |= dirty
            per_layer_counts.append(len(dirty))
            dirty = self._expand(dirty)
        return recomputed_total, per_layer_counts

    # ---------------- public API ----------------
    def initial_run(self):
        """One-time full forward pass. Paid once before streaming starts."""
        t0 = time.perf_counter()
        all_nodes = self.graph.node_ids
        self.cache[0] = {n: self.graph.get_feature(n) for n in all_nodes}
        for layer_idx in range(self.num_layers):
            self._recompute_layer(layer_idx, all_nodes)
        elapsed = time.perf_counter() - t0

        if self.retrieval_index is not None:
            for n in all_nodes:
                key = self.retrieval_key_fn(self.graph.get_feature(n))
                self.retrieval_index.insert(n, key)

        self.last_stats = {"mode": "full", "recomputed_nodes": len(all_nodes),
                            "total_nodes": len(all_nodes), "time_sec": elapsed}
        return self.embeddings()

    def embeddings(self):
        """Current cached final-layer embedding for every node."""
        return dict(self.cache[self.num_layers])

    def on_feature_update(self, node_id, new_feature):
        t0 = time.perf_counter()
        self.graph.set_feature(node_id, new_feature)
        self.cache[0][node_id] = self.graph.get_feature(node_id)

        if self.retrieval_index is not None:
            new_key = self.retrieval_key_fn(self.graph.get_feature(node_id))
            self.retrieval_index.update_key(node_id, new_key)

        # Layer-0 input changed for node_id -> node_id itself AND its
        # neighbors are already dirty for layer 1's output.
        seed = self._expand({node_id})
        recomputed, per_layer = self._propagate(seed)

        elapsed = time.perf_counter() - t0
        self.last_stats = {"mode": "delta-feature", "event": node_id,
                            "recomputed_nodes": len(recomputed),
                            "total_nodes": len(self.graph.node_ids),
                            "per_layer_dirty_counts": per_layer,
                            "time_sec": elapsed}
        return recomputed

    def _on_edge_event(self, u, v, apply_fn, label):
        t0 = time.perf_counter()
        changed = apply_fn(u, v)
        if not changed:
            self.last_stats = {"mode": label, "event": (u, v), "recomputed_nodes": 0,
                                "total_nodes": len(self.graph.node_ids),
                                "time_sec": 0.0, "note": "no-op: edge state unchanged"}
            return set()

        # No layer-0 feature changed; only u and v's aggregation structure
        # changed, so only {u, v} are dirty for layer 1's output.
        recomputed, per_layer = self._propagate({u, v})

        elapsed = time.perf_counter() - t0
        self.last_stats = {"mode": label, "event": (u, v),
                            "recomputed_nodes": len(recomputed),
                            "total_nodes": len(self.graph.node_ids),
                            "per_layer_dirty_counts": per_layer,
                            "time_sec": elapsed}
        return recomputed

    def on_edge_insert(self, u, v):
        return self._on_edge_event(u, v, self.graph.add_edge, "delta-edge-insert")

    def on_edge_delete(self, u, v):
        return self._on_edge_event(u, v, self.graph.remove_edge, "delta-edge-delete")

    # ---------------- correctness check ----------------
    def full_recompute(self):
        """Ground-truth recomputation from scratch. Used only to VERIFY
        exactness -- never called on the streaming hot path."""
        all_nodes = self.graph.node_ids
        cache = [dict() for _ in range(self.num_layers + 1)]
        cache[0] = {n: self.graph.get_feature(n) for n in all_nodes}
        with torch.no_grad():
            for layer_idx in range(self.num_layers):
                layer = self.model.layers[layer_idx]
                get_h = lambda n, L=layer_idx: cache[L][n]
                out = layer.forward(all_nodes, get_h, self.graph.adjacency)
                cache[layer_idx + 1].update(out)
        return cache[self.num_layers]

    def verify_exactness(self, atol=1e-6):
        """Compares the delta-maintained cache against a full recompute.
        Returns (is_exact: bool, max_abs_diff: float)."""
        full = self.full_recompute()
        cached = self.embeddings()
        max_diff = 0.0
        for n in self.graph.node_ids:
            diff = (full[n] - cached[n]).abs().max().item()
            max_diff = max(max_diff, diff)
        return max_diff <= atol, max_diff
