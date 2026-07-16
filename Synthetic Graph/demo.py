"""
demo.py
-------
Runs two demonstrations:

  1. THE TOY EXAMPLE FROM THE PROPOSAL (Sections 3 and 12): the six-student
     graph A-F, the F: 55 -> 75 activity-score event, the resulting
     range-membership change for E, and a proof that DynaDeltaGNN's local
     update is byte-for-byte identical to a full recomputation.

  2. A SCALING BENCHMARK on a larger random graph, timing a full
     recompute vs. a delta update after a single feature change -- the
     concrete efficiency number for RQ1.

Run with:  python3 demo.py
"""

import random
import torch

from graph_store import DynamicGraphStore
from order_statistic_tree import GlobalOrderStatisticTree
from gnn_model import GraphSAGEMean
from delta_engine import DynaDeltaGNN


def retrieval_key_fn(feature_vector):
    """Recover the raw activity score from the (normalized) feature."""
    return round(feature_vector[0].item() * 100, 4)


def build_toy_graph():
    scores = {"A": 70, "B": 82, "C": 78, "D": 90, "E": 74, "F": 55}
    edges = [("A", "B"), ("A", "C"), ("B", "D"), ("B", "E"), ("C", "E"), ("E", "F")]

    store = DynamicGraphStore(node_ids=scores.keys(), feature_dim=1)
    for node, score in scores.items():
        store.set_feature(node, [score / 100.0])
    for u, v in edges:
        store.add_edge(u, v)
    return store


def print_stats(engine, label):
    s = engine.last_stats
    print(f"\n--- {label} ---")
    print(f"  mode               : {s['mode']}")
    if "event" in s:
        print(f"  event              : {s['event']}")
    print(f"  nodes recomputed   : {s['recomputed_nodes']} / {s['total_nodes']}")
    if "per_layer_dirty_counts" in s:
        print(f"  dirty count/layer  : {s['per_layer_dirty_counts']}")
    print(f"  wall time (sec)    : {s['time_sec']:.6f}")


def section1_toy_example():
    print("=" * 70)
    print("PART 1 - TOY EXAMPLE FROM THE PROPOSAL (six-student graph)")
    print("=" * 70)

    store = build_toy_graph()
    print(store)

    torch.manual_seed(0)
    model = GraphSAGEMean(in_dim=1, hidden_dim=4, num_layers=2)

    index = GlobalOrderStatisticTree()
    engine = DynaDeltaGNN(model, store, retrieval_index=index, retrieval_key_fn=retrieval_key_fn)
    engine.initial_run()
    print_stats(engine, "Initial full run")

    def neighbors_in_range(node, low, high):
        """Combines the global order-statistic tree with the node's own
        adjacency list -- exactly NeighborsInRange(node, low, high)
        from Section 7.3 of the proposal."""
        in_range = set(index.range_query(low, high))
        return sorted(in_range & store.neighbors(node))

    before = neighbors_in_range("E", 64, 84)
    print(f"\nNeighborsInRange(E, 64, 84) BEFORE the event : {before}")
    assert before == ["B", "C"], "Expected {B, C} before F's score changes"

    print("\nStreaming event: F's activity score changes 55 -> 75")
    engine.on_feature_update("F", [0.75])
    print_stats(engine, "Delta update after feature change")

    after = neighbors_in_range("E", 64, 84)
    print(f"\nNeighborsInRange(E, 64, 84) AFTER the event  : {after}")
    assert after == ["B", "C", "F"], "Expected {B, C, F} after F's score changes"
    print("Range-membership change detected correctly: F entered E's selected set.")

    ok, max_diff = engine.verify_exactness()
    print(f"\nExactness check vs. full recompute: {'PASS' if ok else 'FAIL'} "
          f"(max abs diff = {max_diff:.2e})")
    assert ok, "Delta-maintained embeddings must exactly match full recomputation"

    print("\nNow trying an edge event: inserting edge B---F")
    engine.on_edge_insert("B", "F")
    print_stats(engine, "Delta update after edge insertion")
    ok, max_diff = engine.verify_exactness()
    print(f"Exactness check vs. full recompute: {'PASS' if ok else 'FAIL'} "
          f"(max abs diff = {max_diff:.2e})")
    assert ok


def build_random_graph(num_nodes, avg_degree, feature_dim, seed=0):
    rng = random.Random(seed)
    node_ids = [f"n{i}" for i in range(num_nodes)]
    store = DynamicGraphStore(node_ids=node_ids, feature_dim=feature_dim)
    for n in node_ids:
        store.set_feature(n, [rng.random() for _ in range(feature_dim)])

    target_edges = num_nodes * avg_degree // 2
    added = 0
    while added < target_edges:
        u, v = rng.choice(node_ids), rng.choice(node_ids)
        if u != v and store.add_edge(u, v):
            added += 1
    return store


def section2_scaling_benchmark():
    print("\n" + "=" * 70)
    print("PART 2 - SCALING BENCHMARK (random graph)")
    print("=" * 70)

    NUM_NODES = 5000
    AVG_DEGREE = 8
    FEATURE_DIM = 8

    torch.manual_seed(0)
    store = build_random_graph(NUM_NODES, AVG_DEGREE, FEATURE_DIM)
    print(store)

    model = GraphSAGEMean(in_dim=FEATURE_DIM, hidden_dim=16, num_layers=2)
    engine = DynaDeltaGNN(model, store)
    engine.initial_run()
    print_stats(engine, "Initial full run")

    changed_node = store.node_ids[0]
    new_feature = [0.99] * FEATURE_DIM
    engine.on_feature_update(changed_node, new_feature)
    print_stats(engine, f"Delta update after changing {changed_node}'s feature")

    ok, max_diff = engine.verify_exactness()
    print(f"\nExactness check vs. full recompute: {'PASS' if ok else 'FAIL'} "
          f"(max abs diff = {max_diff:.2e})")
    assert ok

    full_time = None
    for s in [engine.last_stats]:
        pass
    # Time a fresh full recompute for direct comparison
    import time
    t0 = time.perf_counter()
    engine.full_recompute()
    full_time = time.perf_counter() - t0

    delta_time = engine.last_stats["time_sec"]
    recomputed = engine.last_stats["recomputed_nodes"]
    speedup = full_time / delta_time if delta_time > 0 else float("inf")

    print("\nSummary")
    print(f"  total nodes                 : {NUM_NODES}")
    print(f"  nodes recomputed by delta    : {recomputed} "
          f"({100 * recomputed / NUM_NODES:.2f}% of the graph)")
    print(f"  full recompute time (sec)    : {full_time:.6f}")
    print(f"  delta update time (sec)      : {delta_time:.6f}")
    print(f"  speedup                      : {speedup:.1f}x")


if __name__ == "__main__":
    section1_toy_example()
    section2_scaling_benchmark()
    print("\nAll demonstrations completed successfully.")
