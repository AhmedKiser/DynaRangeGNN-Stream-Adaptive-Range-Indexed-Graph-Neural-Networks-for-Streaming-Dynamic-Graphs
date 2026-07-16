"""
test_correctness.py
--------------------
Plain-assert test suite (no pytest dependency required).

Covers:
  - DynaDeltaGNN exactness after a feature update, an edge insertion,
    an edge deletion, and a sequence of mixed events (RQ1 / Section 13).
  - GlobalOrderStatisticTree correctness against a brute-force scan
    (Section 11.1 / Phase 2).

Run with:  python3 test_correctness.py
"""

import random
import torch

from graph_store import DynamicGraphStore
from order_statistic_tree import GlobalOrderStatisticTree
from gnn_model import GraphSAGEMean
from delta_engine import DynaDeltaGNN


def make_engine(num_nodes=30, avg_degree=4, feature_dim=5, seed=0):
    rng = random.Random(seed)
    torch.manual_seed(seed)
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

    model = GraphSAGEMean(in_dim=feature_dim, hidden_dim=6, num_layers=2)
    engine = DynaDeltaGNN(model, store)
    engine.initial_run()
    return engine, store, rng


def test_feature_update_exact():
    engine, store, rng = make_engine(seed=1)
    for _ in range(15):
        node = rng.choice(store.node_ids)
        new_feat = [rng.random() for _ in range(store.feature_dim)]
        engine.on_feature_update(node, new_feat)
        ok, diff = engine.verify_exactness()
        assert ok, f"feature update on {node} broke exactness (diff={diff})"
    print("test_feature_update_exact: PASS")


def test_edge_insert_exact():
    engine, store, rng = make_engine(seed=2)
    tries, done = 0, 0
    while done < 15 and tries < 200:
        tries += 1
        u, v = rng.choice(store.node_ids), rng.choice(store.node_ids)
        if u == v:
            continue
        changed = engine.on_edge_insert(u, v)
        if not changed:
            continue
        ok, diff = engine.verify_exactness()
        assert ok, f"edge insert ({u},{v}) broke exactness (diff={diff})"
        done += 1
    assert done == 15
    print("test_edge_insert_exact: PASS")


def test_edge_delete_exact():
    engine, store, rng = make_engine(seed=3)
    existing = [tuple(e) for e in (frozenset(x) for x in store.edge_hash)]
    existing = [tuple(e) for e in store.edge_hash]
    rng.shuffle(existing)
    done = 0
    for e in existing:
        if len(e) != 2:
            continue
        u, v = tuple(e)
        changed = engine.on_edge_delete(u, v)
        if not changed:
            continue
        ok, diff = engine.verify_exactness()
        assert ok, f"edge delete ({u},{v}) broke exactness (diff={diff})"
        done += 1
        if done >= 10:
            break
    assert done >= 1, "no edges were available to delete"
    print("test_edge_delete_exact: PASS")


def test_mixed_event_sequence_exact():
    engine, store, rng = make_engine(seed=4)
    for _ in range(30):
        choice = rng.choice(["feature", "insert", "delete"])
        if choice == "feature":
            node = rng.choice(store.node_ids)
            engine.on_feature_update(node, [rng.random() for _ in range(store.feature_dim)])
        elif choice == "insert":
            u, v = rng.choice(store.node_ids), rng.choice(store.node_ids)
            if u != v:
                engine.on_edge_insert(u, v)
        else:
            if store.edge_hash:
                u, v = tuple(rng.choice(list(store.edge_hash)))
                engine.on_edge_delete(u, v)
        ok, diff = engine.verify_exactness()
        assert ok, f"mixed sequence broke exactness at step (diff={diff})"
    print("test_mixed_event_sequence_exact: PASS")


def test_order_statistic_tree_matches_brute_force():
    rng = random.Random(7)
    tree = GlobalOrderStatisticTree()
    truth = {}  # node_id -> key

    node_ids = [f"n{i}" for i in range(200)]
    for n in node_ids:
        key = round(rng.uniform(0, 100), 3)
        tree.insert(n, key)
        truth[n] = key

    def brute_force_range(low, high):
        return sorted(n for n, k in truth.items() if low <= k <= high)

    for _ in range(50):
        low = round(rng.uniform(0, 90), 3)
        high = round(low + rng.uniform(0, 20), 3)
        assert sorted(tree.range_query(low, high)) == brute_force_range(low, high)
        assert tree.range_count(low, high) == len(brute_force_range(low, high))

    # Random updates, re-check
    for _ in range(100):
        n = rng.choice(node_ids)
        new_key = round(rng.uniform(0, 100), 3)
        tree.update_key(n, new_key)
        truth[n] = new_key

    for _ in range(50):
        low = round(rng.uniform(0, 90), 3)
        high = round(low + rng.uniform(0, 20), 3)
        assert sorted(tree.range_query(low, high)) == brute_force_range(low, high)

    # Deletions
    to_delete = node_ids[:20]
    for n in to_delete:
        tree.delete(n)
        del truth[n]
    assert len(tree) == len(truth)
    assert sorted(tree.range_query(0, 100)) == sorted(truth.keys())

    print("test_order_statistic_tree_matches_brute_force: PASS")


if __name__ == "__main__":
    test_feature_update_exact()
    test_edge_insert_exact()
    test_edge_delete_exact()
    test_mixed_event_sequence_exact()
    test_order_statistic_tree_matches_brute_force()
    print("\nAll tests passed.")
