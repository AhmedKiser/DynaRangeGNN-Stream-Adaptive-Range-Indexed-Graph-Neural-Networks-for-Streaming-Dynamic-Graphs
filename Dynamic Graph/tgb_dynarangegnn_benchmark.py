
"""
TGB + DynaRangeGNN-Stream benchmark and visualization.

Purpose
-------
1. Load a Temporal Graph Benchmark link-prediction dataset.
2. Visualize the temporal graph.
3. Train a simple range-conditioned GraphSAGE link predictor.
4. Compare:
   - full range-conditioned GraphSAGE recomputation
   - DynaRangeGNN-Stream local update
5. Report prediction, time, dirty-node, and space results.

Install
-------
pip install torch torch-geometric py-tgb scikit-learn pandas numpy matplotlib networkx tqdm

Quick run
---------
python tgb_dynarangegnn_benchmark.py --dataset tgbl-wiki --max_events 5000 --num_batches 50 --batch_size 100 --epochs 5
"""

from __future__ import annotations

import argparse
import bisect
import random
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Set, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm


# ============================================================
# Basic utilities
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def edge_key(u: int, v: int) -> Tuple[int, int]:
    if u == v:
        raise ValueError("Self-loops are not used in this prototype.")
    return (u, v) if u < v else (v, u)


@dataclass(frozen=True)
class TemporalEvent:
    src: int
    dst: int
    t: float


@dataclass(frozen=True)
class FeatureKeyUpdate:
    node: int
    new_key: float


class DynamicGraph:
    """Undirected dynamic graph using adjacency sets and an edge hash."""

    def __init__(self, num_nodes: int, edges: Iterable[Tuple[int, int]] = ()) -> None:
        self.num_nodes = int(num_nodes)
        self.adj: List[Set[int]] = [set() for _ in range(self.num_nodes)]
        self.edge_hash: Set[Tuple[int, int]] = set()
        for u, v in edges:
            self.add_edge(u, v)

    def has_edge(self, u: int, v: int) -> bool:
        return u != v and edge_key(u, v) in self.edge_hash

    def add_edge(self, u: int, v: int) -> bool:
        if u == v:
            return False
        k = edge_key(u, v)
        if k in self.edge_hash:
            return False
        self.edge_hash.add(k)
        self.adj[u].add(v)
        self.adj[v].add(u)
        return True

    def degree(self, u: int) -> int:
        return len(self.adj[u])

    def check(self) -> None:
        for u in range(self.num_nodes):
            for v in self.adj[u]:
                assert u in self.adj[v], f"Asymmetric adjacency {u}-{v}"
                assert edge_key(u, v) in self.edge_hash, f"Missing edge hash {u}-{v}"


# ============================================================
# TGB loading
# ============================================================

def to_numpy(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_tgb_events(dataset_name: str, root: str, max_events: Optional[int]) -> Tuple[List[TemporalEvent], int, Dict[str, str]]:
    """
    Loads TGB temporal link prediction data.

    Expected common API:
        from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
        dataset = PyGLinkPropPredDataset(name="tgbl-wiki", root="./tgb_data")
        data = dataset.get_TemporalData()
        data.src, data.dst, data.t
    """
    try:
        from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
    except ImportError as exc:
        raise ImportError(
            "Install TGB first:\n"
            "pip install py-tgb torch-geometric"
        ) from exc

    dataset = PyGLinkPropPredDataset(name=dataset_name, root=root)
    data = dataset.get_TemporalData() if hasattr(dataset, "get_TemporalData") else dataset[0]

    src = to_numpy(getattr(data, "src", None))
    dst = to_numpy(getattr(data, "dst", None))
    t = to_numpy(getattr(data, "t", None))

    if src is None or dst is None or t is None:
        edge_index = getattr(data, "edge_index", None)
        timestamp = getattr(data, "timestamp", None)
        if edge_index is None or timestamp is None:
            raise RuntimeError("Could not find src/dst/t or edge_index/timestamp in the TGB data object.")
        edge_index = to_numpy(edge_index)
        src = edge_index[0]
        dst = edge_index[1]
        t = to_numpy(timestamp)

    src = src.astype(np.int64)
    dst = dst.astype(np.int64)
    t = t.astype(np.float64)

    order = np.argsort(t)
    src, dst, t = src[order], dst[order], t[order]

    if max_events is not None:
        src, dst, t = src[:max_events], dst[:max_events], t[:max_events]

    num_nodes = int(max(src.max(), dst.max()) + 1)
    events = [
        TemporalEvent(int(src[i]), int(dst[i]), float(t[i]))
        for i in range(len(src))
        if int(src[i]) != int(dst[i])
    ]

    meta = {
        "dataset": dataset_name,
        "events_loaded": str(len(events)),
        "num_nodes": str(num_nodes),
        "min_time": str(float(t.min())),
        "max_time": str(float(t.max())),
    }
    return events, num_nodes, meta


# ============================================================
# Visualization
# ============================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def visualize_temporal_graph(events: List[TemporalEvent], num_nodes: int, out_dir: Path, sample_edges: int = 2000) -> None:
    ensure_dir(out_dir)

    src = np.array([e.src for e in events], dtype=int)
    dst = np.array([e.dst for e in events], dtype=int)
    ts = np.array([e.t for e in events], dtype=float)

    unique_edges = {edge_key(int(u), int(v)) for u, v in zip(src, dst) if u != v}
    active_nodes = set(src.tolist()) | set(dst.tolist())

    degrees = np.zeros(num_nodes, dtype=int)
    for u, v in unique_edges:
        degrees[u] += 1
        degrees[v] += 1

    pd.DataFrame([{
        "num_nodes_total": num_nodes,
        "num_active_nodes": len(active_nodes),
        "num_events": len(events),
        "num_unique_undirected_edges": len(unique_edges),
        "mean_degree": float(degrees.mean()),
        "median_degree": float(np.median(degrees)),
        "max_degree": int(degrees.max()),
        "min_time": float(ts.min()),
        "max_time": float(ts.max()),
    }]).to_csv(out_dir / "01_dataset_summary.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.hist(ts, bins=50)
    plt.xlabel("Timestamp")
    plt.ylabel("Number of events")
    plt.title("Temporal event distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "02_event_count_over_time.png", dpi=300)
    plt.close()

    seen = set()
    xs, ys = [], []
    step = max(1, len(events) // 200)
    for i, e in enumerate(events):
        seen.add(e.src)
        seen.add(e.dst)
        if i % step == 0 or i == len(events) - 1:
            xs.append(i + 1)
            ys.append(len(seen))

    plt.figure(figsize=(9, 5))
    plt.plot(xs, ys)
    plt.xlabel("Processed events")
    plt.ylabel("Active nodes")
    plt.title("Active node growth")
    plt.tight_layout()
    plt.savefig(out_dir / "03_active_nodes_over_events.png", dpi=300)
    plt.close()

    nz_deg = degrees[degrees > 0]
    plt.figure(figsize=(9, 5))
    plt.hist(nz_deg, bins=50)
    plt.xlabel("Degree after loaded events")
    plt.ylabel("Number of nodes")
    plt.title("Degree distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "04_degree_distribution.png", dpi=300)
    plt.close()

    # Graph snapshot.
    g = nx.Graph()
    for u, v in list(unique_edges)[:sample_edges]:
        g.add_edge(u, v)

    if g.number_of_edges() > 0:
        if g.number_of_nodes() > 300:
            deg = dict(g.degree())
            keep = sorted(deg, key=deg.get, reverse=True)[:300]
            g = g.subgraph(keep).copy()

        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(g, seed=42, iterations=50)
        nx.draw_networkx_edges(g, pos, width=0.3, alpha=0.4)
        nx.draw_networkx_nodes(g, pos, node_size=25)
        plt.title("Sample graph snapshot")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / "05_sample_graph_snapshot.png", dpi=300)
        plt.close()


# ============================================================
# Activity key and range index
# ============================================================

class ActivityKeyManager:
    """
    Time-decayed activity score:
        key increases when the node participates in an event.
    """

    def __init__(self, num_nodes: int, decay: float = 0.995, boost: float = 1.0) -> None:
        self.num_nodes = num_nodes
        self.decay = float(decay)
        self.boost = float(boost)
        self.key = np.zeros(num_nodes, dtype=np.float64)
        self.last_t = np.zeros(num_nodes, dtype=np.float64)

    def value(self, v: int, t: float) -> float:
        dt = max(0.0, float(t) - self.last_t[v])
        factor = self.decay ** np.log1p(dt)
        return float(self.key[v] * factor)

    def touch(self, v: int, t: float) -> float:
        val = self.value(v, t) + self.boost
        self.key[v] = val
        self.last_t[v] = float(t)
        return val

    def snapshot(self, t: float) -> np.ndarray:
        return np.array([self.value(v, t) for v in range(self.num_nodes)], dtype=np.float64)


class RangeIndex:
    """Per-node sorted neighbor lists: [(key[neighbor], neighbor_id), ...]."""

    def __init__(self, graph: DynamicGraph, keys: np.ndarray) -> None:
        self.graph = graph
        self.keys = keys.astype(np.float64).copy()
        self.sorted_neighbors: List[List[Tuple[float, int]]] = []
        self.rebuild_all()

    def rebuild_all(self) -> None:
        self.sorted_neighbors = [
            sorted((float(self.keys[v]), int(v)) for v in self.graph.adj[u])
            for u in range(self.graph.num_nodes)
        ]

    def range_neighbors(self, u: int, low: float, high: float) -> List[int]:
        arr = self.sorted_neighbors[u]
        left = bisect.bisect_left(arr, (float(low), -1))
        right = bisect.bisect_right(arr, (float(high), 10**18))
        return [v for _, v in arr[left:right]]

    def selected_set(self, u: int, delta: float) -> Set[int]:
        center = self.keys[u]
        return set(self.range_neighbors(u, center - delta, center + delta))

    def _remove(self, owner: int, nbr: int, old_key: float) -> None:
        arr = self.sorted_neighbors[owner]
        item = (float(old_key), int(nbr))
        i = bisect.bisect_left(arr, item)
        if i < len(arr) and arr[i] == item:
            arr.pop(i)
            return
        for j, (_, node) in enumerate(arr):
            if node == nbr:
                arr.pop(j)
                return

    def _insert(self, owner: int, nbr: int, key: float) -> None:
        bisect.insort(self.sorted_neighbors[owner], (float(key), int(nbr)))

    def update_key(self, node: int, new_key: float) -> None:
        old = float(self.keys[node])
        if old == float(new_key):
            return
        for owner in list(self.graph.adj[node]):
            self._remove(owner, node, old)
            self._insert(owner, node, float(new_key))
        self.keys[node] = float(new_key)

    def add_edge(self, u: int, v: int) -> None:
        self._insert(u, v, self.keys[v])
        self._insert(v, u, self.keys[u])

    def memory_estimate_mb(self) -> float:
        pair_count = sum(len(a) for a in self.sorted_neighbors)
        # rough estimate: float key + int id, 8 bytes each
        return pair_count * 16 / (1024 ** 2)


# ============================================================
# Range-conditioned GraphSAGE
# ============================================================

class RangeSAGELayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.neighbor_linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, h_self: Tensor, mean_nbr: Tensor) -> Tensor:
        return F.relu(self.self_linear(h_self) + self.neighbor_linear(mean_nbr))


class RangeGraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.layer1 = RangeSAGELayer(in_dim, hidden_dim)
        self.layer2 = RangeSAGELayer(hidden_dim, out_dim)

    @staticmethod
    def aggregate(h: Tensor, ri: RangeIndex, delta: float) -> Tensor:
        out = torch.zeros_like(h)
        for u in range(h.size(0)):
            c = ri.keys[u]
            nbrs = ri.range_neighbors(u, c - delta, c + delta)
            if nbrs:
                idx = torch.tensor(nbrs, dtype=torch.long, device=h.device)
                out[u] = h[idx].mean(dim=0)
        return out

    def full_cache(self, x: Tensor, ri: RangeIndex, delta: float) -> Dict[str, List[Tensor]]:
        h0 = x.clone()
        a1 = self.aggregate(h0, ri, delta)
        h1 = self.layer1(h0, a1)
        a2 = self.aggregate(h1, ri, delta)
        h2 = self.layer2(h1, a2)
        return {"H": [h0, h1, h2], "A": [None, a1, a2]}

    def recompute_l1(self, cache, nodes: Set[int], ri: RangeIndex, delta: float) -> Dict[int, Tensor]:
        changes = {}
        h0 = cache["H"][0]
        for u in sorted(nodes):
            old = cache["H"][1][u].clone()
            c = ri.keys[u]
            nbrs = ri.range_neighbors(u, c - delta, c + delta)
            mean = torch.zeros_like(h0[u])
            if nbrs:
                idx = torch.tensor(nbrs, dtype=torch.long, device=h0.device)
                mean = h0[idx].mean(dim=0)
            cache["A"][1][u] = mean
            cache["H"][1][u] = self.layer1(h0[u:u+1], mean[None, :]).squeeze(0)
            changes[u] = cache["H"][1][u] - old
        return changes

    def recompute_l2(self, cache, nodes: Set[int], ri: RangeIndex, delta: float) -> None:
        h1 = cache["H"][1]
        for u in sorted(nodes):
            c = ri.keys[u]
            nbrs = ri.range_neighbors(u, c - delta, c + delta)
            mean = torch.zeros_like(h1[u])
            if nbrs:
                idx = torch.tensor(nbrs, dtype=torch.long, device=h1.device)
                mean = h1[idx].mean(dim=0)
            cache["A"][2][u] = mean
            cache["H"][2][u] = self.layer2(h1[u:u+1], mean[None, :]).squeeze(0)


class DotDecoder(nn.Module):
    def forward(self, z: Tensor, edges: List[Tuple[int, int]]) -> Tensor:
        if not edges:
            return torch.empty(0, device=z.device)
        u = torch.tensor([e[0] for e in edges], dtype=torch.long, device=z.device)
        v = torch.tensor([e[1] for e in edges], dtype=torch.long, device=z.device)
        return (z[u] * z[v]).sum(dim=1)


# ============================================================
# Features, training, and prediction
# ============================================================

def make_node_features(keys: np.ndarray, graph: DynamicGraph, device: torch.device) -> Tensor:
    degrees = np.array([graph.degree(i) for i in range(graph.num_nodes)], dtype=np.float32)
    k = keys.astype(np.float32)
    if k.max() > k.min():
        k = (k - k.min()) / (k.max() - k.min())
    d = np.log1p(degrees)
    if d.max() > d.min():
        d = (d - d.min()) / (d.max() - d.min())
    x = np.stack([k, d, np.ones_like(k)], axis=1)
    return torch.tensor(x, dtype=torch.float32, device=device)


def update_one_node_feature(x: Tensor, node: int, key: float, graph: DynamicGraph) -> None:
    x[node, 0] = float(key)
    x[node, 1] = float(np.log1p(graph.degree(node)))


def negative_edges(num_nodes: int, forbidden: Set[Tuple[int, int]], count: int, rng: random.Random) -> List[Tuple[int, int]]:
    out = set()
    attempts = 0
    while len(out) < count and attempts < max(1000, count * 100):
        attempts += 1
        u = rng.randrange(num_nodes)
        v = rng.randrange(num_nodes)
        if u == v:
            continue
        e = edge_key(u, v)
        if e not in forbidden and e not in out:
            out.add(e)
    return list(out)


def train_initial_model(model, graph, ri, x, delta, epochs, lr, seed, device) -> None:
    if not graph.edge_hash:
        return
    rng = random.Random(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    decoder = DotDecoder().to(device)
    pos_pool = list(graph.edge_hash)

    for _ in tqdm(range(epochs), desc="Training initial model"):
        model.train()
        opt.zero_grad()
        m = min(len(pos_pool), 3000)
        pos = rng.sample(pos_pool, m)
        neg = negative_edges(graph.num_nodes, graph.edge_hash, m, rng)
        z = model.full_cache(x, ri, delta)["H"][2]
        logits = torch.cat([decoder(z, pos), decoder(z, neg)])
        labels = torch.cat([torch.ones(len(pos), device=device), torch.zeros(len(neg), device=device)])
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        opt.step()
    model.eval()


@torch.no_grad()
def evaluate_prediction(z: Tensor, positives: List[Tuple[int, int]], graph: DynamicGraph, seed: int) -> Dict[str, float]:
    if not positives:
        return {"roc_auc": float("nan"), "ap": float("nan")}
    rng = random.Random(seed)
    neg = negative_edges(graph.num_nodes, graph.edge_hash | set(positives), len(positives), rng)
    if not neg:
        return {"roc_auc": float("nan"), "ap": float("nan")}
    decoder = DotDecoder().to(z.device)
    edges = positives + neg
    scores = torch.sigmoid(decoder(z, edges)).cpu().numpy()
    labels = np.array([1] * len(positives) + [0] * len(neg))
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "ap": float(average_precision_score(labels, scores)),
    }


# ============================================================
# DynaRange engine
# ============================================================

@dataclass
class BatchStats:
    batch_id: int
    events: int
    edge_adds: int
    key_updates: int
    membership_changed: int
    l1_dirty: int
    l2_dirty: int
    local_ms: float
    full_ms: float
    speedup: float
    max_error: float
    embedding_cache_mb: float
    range_index_mb: float
    trace_peak_mb: float
    dynarange_roc_auc: float
    dynarange_ap: float
    full_roc_auc: float
    full_ap: float


class DynaRangeEngine:
    def __init__(self, model, graph, ri, x, delta, device) -> None:
        self.model = model
        self.graph = graph
        self.ri = ri
        self.x = x
        self.delta = float(delta)
        self.device = device
        self.cache = self.model.full_cache(self.x, self.ri, self.delta)

    def embedding_cache_mb(self) -> float:
        total = 0
        for h in self.cache["H"]:
            total += h.numel() * h.element_size()
        for a in self.cache["A"]:
            if a is not None:
                total += a.numel() * a.element_size()
        return total / (1024 ** 2)

    def process_batch(
        self,
        batch_id: int,
        events: List[TemporalEvent],
        key_updates: List[FeatureKeyUpdate],
        eval_pos: List[Tuple[int, int]],
        eval_seed: int,
    ) -> BatchStats:

        dyn_metrics = evaluate_prediction(self.cache["H"][2], eval_pos, self.graph, eval_seed)

        feature_nodes = {ku.node for ku in key_updates}
        endpoints = {n for e in events for n in (e.src, e.dst)}
        candidates = set(feature_nodes) | set(endpoints)
        for v in list(candidates):
            candidates.update(self.graph.adj[v])

        old_selected = {u: self.ri.selected_set(u, self.delta) for u in candidates}
        old_adj_feature = {v: set(self.graph.adj[v]) for v in feature_nodes}

        tracemalloc.start()
        t0 = time.perf_counter()

        # Apply key updates.
        for ku in key_updates:
            self.ri.update_key(ku.node, ku.new_key)
            update_one_node_feature(self.x, ku.node, ku.new_key, self.graph)
            self.cache["H"][0][ku.node] = self.x[ku.node]

        # Add temporal edges.
        adds = 0
        for e in events:
            if e.src == e.dst:
                continue
            if not self.graph.has_edge(e.src, e.dst):
                if self.graph.add_edge(e.src, e.dst):
                    self.ri.add_edge(e.src, e.dst)
                    adds += 1
                    update_one_node_feature(self.x, e.src, self.ri.keys[e.src], self.graph)
                    update_one_node_feature(self.x, e.dst, self.ri.keys[e.dst], self.graph)
                    self.cache["H"][0][e.src] = self.x[e.src]
                    self.cache["H"][0][e.dst] = self.x[e.dst]

        self.graph.check()

        for v in list(feature_nodes | endpoints):
            candidates.update(self.graph.adj[v])
            if v in old_adj_feature:
                candidates.update(old_adj_feature[v])

        new_selected = {u: self.ri.selected_set(u, self.delta) for u in candidates}
        membership_changed = {u for u in candidates if old_selected.get(u, set()) != new_selected[u]}

        l1_dirty = set(feature_nodes) | set(endpoints) | set(membership_changed)
        for v in feature_nodes:
            owners = old_adj_feature.get(v, set()) | self.graph.adj[v] | {v}
            for u in owners:
                if v in old_selected.get(u, set()) or v in new_selected.get(u, set()):
                    l1_dirty.add(u)

        delta_h1 = self.model.recompute_l1(self.cache, l1_dirty, self.ri, self.delta)
        h1_changed = {u for u, d in delta_h1.items() if torch.max(torch.abs(d)).item() > 1e-12}

        l2_dirty = set(h1_changed) | set(endpoints) | set(membership_changed)
        for h in h1_changed:
            l2_dirty.add(h)
            for u in self.graph.adj[h]:
                c = self.ri.keys[u]
                if c - self.delta <= self.ri.keys[h] <= c + self.delta:
                    l2_dirty.add(u)

        self.model.recompute_l2(self.cache, l2_dirty, self.ri, self.delta)

        local_ms = (time.perf_counter() - t0) * 1000.0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        t1 = time.perf_counter()
        full_cache = self.model.full_cache(self.x, self.ri, self.delta)
        full_ms = (time.perf_counter() - t1) * 1000.0

        max_error = torch.max(torch.abs(self.cache["H"][2] - full_cache["H"][2])).item()
        full_metrics = evaluate_prediction(full_cache["H"][2], eval_pos, self.graph, eval_seed)

        return BatchStats(
            batch_id=batch_id,
            events=len(events),
            edge_adds=adds,
            key_updates=len(key_updates),
            membership_changed=len(membership_changed),
            l1_dirty=len(l1_dirty),
            l2_dirty=len(l2_dirty),
            local_ms=local_ms,
            full_ms=full_ms,
            speedup=full_ms / max(local_ms, 1e-12),
            max_error=max_error,
            embedding_cache_mb=self.embedding_cache_mb(),
            range_index_mb=self.ri.memory_estimate_mb(),
            trace_peak_mb=peak / (1024 ** 2),
            dynarange_roc_auc=dyn_metrics["roc_auc"],
            dynarange_ap=dyn_metrics["ap"],
            full_roc_auc=full_metrics["roc_auc"],
            full_ap=full_metrics["ap"],
        )


# ============================================================
# Result plots
# ============================================================

def plot_results(df: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(df["batch_id"], df["local_ms"], label="DynaRange local")
    plt.plot(df["batch_id"], df["full_ms"], label="Full recomputation")
    plt.xlabel("Batch")
    plt.ylabel("Time (ms)")
    plt.title("Time: full recomputation vs DynaRangeGNN-Stream")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "06_time_full_vs_dynarange.png", dpi=300)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(df["batch_id"], df["speedup"])
    plt.xlabel("Batch")
    plt.ylabel("Speedup")
    plt.title("Speedup over full recomputation")
    plt.tight_layout()
    plt.savefig(out_dir / "07_speedup.png", dpi=300)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(df["batch_id"], df["l1_dirty"], label="Layer 1 dirty")
    plt.plot(df["batch_id"], df["l2_dirty"], label="Layer 2 dirty")
    plt.xlabel("Batch")
    plt.ylabel("Dirty nodes")
    plt.title("Dirty nodes per batch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "08_dirty_nodes.png", dpi=300)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(df["batch_id"], df["max_error"])
    plt.xlabel("Batch")
    plt.ylabel("Max absolute error")
    plt.title("Embedding error: DynaRange local vs full recomputation")
    plt.tight_layout()
    plt.savefig(out_dir / "09_embedding_error.png", dpi=300)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(df["batch_id"], df["embedding_cache_mb"], label="Embedding cache")
    plt.plot(df["batch_id"], df["range_index_mb"], label="Range index estimate")
    plt.xlabel("Batch")
    plt.ylabel("Memory (MB)")
    plt.title("Space estimate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "10_space_usage.png", dpi=300)
    plt.close()

    eval_df = df.dropna(subset=["dynarange_roc_auc"])
    if len(eval_df):
        plt.figure(figsize=(9, 5))
        plt.plot(eval_df["batch_id"], eval_df["dynarange_roc_auc"], label="DynaRange")
        plt.plot(eval_df["batch_id"], eval_df["full_roc_auc"], label="Full recomputation")
        plt.xlabel("Batch")
        plt.ylabel("ROC-AUC")
        plt.title("Prediction ROC-AUC")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "11_prediction_roc_auc.png", dpi=300)
        plt.close()


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="tgbl-wiki")
    parser.add_argument("--root", default="./tgb_data")
    parser.add_argument("--out_dir", default="tgb_dynarange_outputs")
    parser.add_argument("--max_events", type=int, default=10000)
    parser.add_argument("--initial_ratio", type=float, default=0.40)
    parser.add_argument("--num_batches", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--activity_decay", type=float, default=0.995)
    parser.add_argument("--activity_boost", type=float, default=1.0)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--out_dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--eval_pos", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    print(f"Loading TGB dataset: {args.dataset}")
    events, num_nodes, meta = load_tgb_events(args.dataset, args.root, args.max_events)
    print("Dataset meta:", meta)

    visualize_temporal_graph(events, num_nodes, out_dir)

    n_initial = int(len(events) * args.initial_ratio)
    initial_events = events[:n_initial]
    stream_events = events[n_initial:n_initial + args.num_batches * args.batch_size]
    if not stream_events:
        raise RuntimeError("No stream events selected. Increase max_events or reduce initial_ratio.")

    activity = ActivityKeyManager(num_nodes, args.activity_decay, args.activity_boost)
    graph = DynamicGraph(num_nodes)

    current_t = 0.0
    for e in initial_events:
        current_t = e.t
        activity.touch(e.src, e.t)
        activity.touch(e.dst, e.t)
        graph.add_edge(e.src, e.dst)

    keys = activity.snapshot(current_t)
    ri = RangeIndex(graph, keys)
    x = make_node_features(keys, graph, device)

    model = RangeGraphSAGE(in_dim=x.size(1), hidden_dim=args.hidden_dim, out_dim=args.out_dim).to(device)
    train_initial_model(model, graph, ri, x, args.delta, args.epochs, args.lr, args.seed, device)

    engine = DynaRangeEngine(model, graph, ri, x, args.delta, device)

    rows = []
    print("Running streaming benchmark...")
    for batch_id in tqdm(range(args.num_batches)):
        s = batch_id * args.batch_size
        e = min(s + args.batch_size, len(stream_events))
        batch = stream_events[s:e]
        if not batch:
            break

        eval_pos = []
        if batch_id % args.eval_every == 0:
            for ev in batch[:args.eval_pos]:
                if ev.src != ev.dst and not engine.graph.has_edge(ev.src, ev.dst):
                    eval_pos.append(edge_key(ev.src, ev.dst))

        key_updates = []
        for ev in batch:
            key_updates.append(FeatureKeyUpdate(ev.src, activity.touch(ev.src, ev.t)))
            key_updates.append(FeatureKeyUpdate(ev.dst, activity.touch(ev.dst, ev.t)))

        stats = engine.process_batch(batch_id, batch, key_updates, eval_pos, args.seed + batch_id)
        rows.append(stats.__dict__)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "12_benchmark_results.csv", index=False)
    plot_results(df, out_dir)

    summary = pd.DataFrame([{
        "dataset": args.dataset,
        "num_loaded_events": len(events),
        "num_nodes": num_nodes,
        "num_initial_events": len(initial_events),
        "num_stream_batches": len(df),
        "mean_local_ms": float(df["local_ms"].mean()),
        "mean_full_ms": float(df["full_ms"].mean()),
        "mean_speedup": float(df["speedup"].mean()),
        "max_embedding_error": float(df["max_error"].max()),
        "mean_l1_dirty": float(df["l1_dirty"].mean()),
        "mean_l2_dirty": float(df["l2_dirty"].mean()),
        "mean_membership_changed": float(df["membership_changed"].mean()),
        "mean_embedding_cache_mb": float(df["embedding_cache_mb"].mean()),
        "mean_range_index_mb": float(df["range_index_mb"].mean()),
        "mean_dynarange_roc_auc": float(df["dynarange_roc_auc"].dropna().mean()) if df["dynarange_roc_auc"].notna().any() else float("nan"),
        "mean_full_roc_auc": float(df["full_roc_auc"].dropna().mean()) if df["full_roc_auc"].notna().any() else float("nan"),
        "mean_dynarange_ap": float(df["dynarange_ap"].dropna().mean()) if df["dynarange_ap"].notna().any() else float("nan"),
        "mean_full_ap": float(df["full_ap"].dropna().mean()) if df["full_ap"].notna().any() else float("nan"),
    }])
    summary.to_csv(out_dir / "13_summary_results.csv", index=False)

    print("\nBenchmark summary")
    print(summary.T)
    print(f"\nSaved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
