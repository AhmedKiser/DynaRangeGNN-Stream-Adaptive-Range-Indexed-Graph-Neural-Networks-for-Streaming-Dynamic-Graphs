"""
run_evaluation.py
------------------
Ties everything together:

  PART 1 - Toy graph (the six-student example): visualize it before/after
           the F: 55 -> 75 event, illustrate the range-membership change,
           and show a (untrained-weights) link-prediction score just to
           demonstrate the plumbing works end-to-end.

  PART 2 - A synthetic 300-node dataset: visualize its degree
           distribution, properly TRAIN a GraphSAGE-Mean link predictor
           (edges held out, negatives sampled), evaluate it with
           ROC-AUC / AP / Precision / Recall / F1 / Accuracy, and
           visualize the trained embeddings + ROC/PR curves.

Run with:  python3 run_evaluation.py
All figures are written to ./figures/
"""

import os
import random
import torch

from graph_store import DynamicGraphStore
from gnn_model import GraphSAGEMean, full_forward
from delta_engine import DynaDeltaGNN
from order_statistic_tree import GlobalOrderStatisticTree
from link_predictor import DotProductDecoder
from train_eval import (split_edges, build_train_store, sample_negative_edges,
                         train_link_predictor, evaluate_link_prediction)
from visualize import (plot_toy_graph, plot_range_query_illustration,
                        plot_degree_distribution, plot_embeddings_pca,
                        plot_roc_pr_curves, plot_training_loss)

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


def retrieval_key_fn(feature_vector):
    return round(feature_vector[0].item() * 100, 4)


def build_toy_graph():
    scores = {"A": 70, "B": 82, "C": 78, "D": 90, "E": 74, "F": 55}
    edges = [("A", "B"), ("A", "C"), ("B", "D"), ("B", "E"), ("C", "E"), ("E", "F")]
    store = DynamicGraphStore(node_ids=scores.keys(), feature_dim=1)
    for node, score in scores.items():
        store.set_feature(node, [score / 100.0])
    for u, v in edges:
        store.add_edge(u, v)
    return store, scores


def part1_toy_graph():
    print("=" * 70)
    print("PART 1 - TOY GRAPH: DATASET VISUALIZATION")
    print("=" * 70)
    store, scores = build_toy_graph()

    plot_toy_graph(store, scores, f"{FIG_DIR}/01_toy_graph_before.png",
                    title="Toy Graph BEFORE F's score change")
    plot_range_query_illustration(scores, "E", 64, 84, f"{FIG_DIR}/02_range_query_before.png",
                                   title="NeighborsInRange(E, 64, 84) BEFORE")

    torch.manual_seed(0)
    model = GraphSAGEMean(in_dim=1, hidden_dim=4, num_layers=2)
    decoder = DotProductDecoder()
    index = GlobalOrderStatisticTree()
    engine = DynaDeltaGNN(model, store, retrieval_index=index, retrieval_key_fn=retrieval_key_fn)
    engine.initial_run()

    prob_before = decoder.probs(engine.embeddings(), [("A", "E")]).item()
    print(f"P(A-E link) BEFORE F's score change: {prob_before:.4f}  (random/untrained weights -- "
          f"see Part 2 for a properly trained + evaluated model)")

    engine.on_feature_update("F", [0.75])
    scores_after = dict(scores)
    scores_after["F"] = 75

    plot_toy_graph(store, scores_after, f"{FIG_DIR}/03_toy_graph_after.png",
                    title="Toy Graph AFTER F's score change")
    plot_range_query_illustration(scores_after, "E", 64, 84, f"{FIG_DIR}/04_range_query_after.png",
                                   title="NeighborsInRange(E, 64, 84) AFTER")

    prob_after = decoder.probs(engine.embeddings(), [("A", "E")]).item()
    print(f"P(A-E link) AFTER F's score change:  {prob_after:.4f}")

    ok, diff = engine.verify_exactness()
    print(f"Exactness check after the event: {'PASS' if ok else 'FAIL'} (max diff = {diff:.2e})")
    print(f"\nSaved: 01_toy_graph_before.png, 02_range_query_before.png, "
          f"03_toy_graph_after.png, 04_range_query_after.png")


def build_community_graph(num_nodes, num_communities, avg_degree, feature_dim,
                           homophily=10.0, seed=0):
    """A synthetic dataset with actual learnable structure (a simple
    stochastic-block-model-style generator): each node belongs to a
    community, its feature vector is a noisy version of that community's
    centroid, and edges are `homophily`-times more likely WITHIN a
    community than across communities. A pure Erdos-Renyi random graph
    (features and edges both independent noise) gives a GNN nothing to
    learn -- this gives it a real, checkable signal to pick up."""
    rng = random.Random(seed)
    node_ids = [f"n{i}" for i in range(num_nodes)]
    community_of = {n: rng.randrange(num_communities) for n in node_ids}

    store = DynamicGraphStore(node_ids=node_ids, feature_dim=feature_dim)
    centroids = [[rng.uniform(-1, 1) for _ in range(feature_dim)] for _ in range(num_communities)]
    for n in node_ids:
        c = centroids[community_of[n]]
        store.set_feature(n, [c[i] + rng.gauss(0, 0.15) for i in range(feature_dim)])

    target_edges = num_nodes * avg_degree // 2
    p_in = 0.9
    p_out = p_in / homophily
    added, tries, max_tries = 0, 0, target_edges * 300
    while added < target_edges and tries < max_tries:
        tries += 1
        u, v = rng.choice(node_ids), rng.choice(node_ids)
        if u == v:
            continue
        accept_prob = p_in if community_of[u] == community_of[v] else p_out
        if rng.random() > accept_prob:
            continue
        if store.add_edge(u, v):
            added += 1
    return store, community_of


def part2_train_and_evaluate():
    print("\n" + "=" * 70)
    print("PART 2 - SYNTHETIC DATASET: TRAIN + EVALUATE LINK PREDICTION")
    print("=" * 70)

    NUM_NODES, NUM_COMMUNITIES, AVG_DEGREE, FEATURE_DIM = 300, 4, 6, 8

    full_store, community_of = build_community_graph(
        NUM_NODES, NUM_COMMUNITIES, AVG_DEGREE, FEATURE_DIM, homophily=10.0, seed=1
    )
    print(full_store)
    print(f"Communities: {NUM_COMMUNITIES} (nodes are ~10x more likely to connect "
          f"within their own community than across communities)")
    plot_degree_distribution(full_store, f"{FIG_DIR}/05_degree_distribution.png",
                              title=f"Degree Distribution ({NUM_NODES}-node synthetic graph)")

    train_edges, val_edges, test_edges = split_edges(full_store, val_frac=0.1, test_frac=0.2, seed=1)
    print(f"Edges -> train: {len(train_edges)}, val: {len(val_edges)}, test: {len(test_edges)}")

    train_store = build_train_store(full_store, train_edges)

    rng = random.Random(2)
    all_true_edges = train_edges + val_edges + test_edges
    val_neg = sample_negative_edges(full_store, len(val_edges), exclude_edges=all_true_edges, rng=rng)
    test_neg = sample_negative_edges(full_store, len(test_edges),
                                      exclude_edges=all_true_edges + val_neg, rng=rng)

    torch.manual_seed(0)
    model = GraphSAGEMean(in_dim=FEATURE_DIM, hidden_dim=16, num_layers=2)
    decoder = DotProductDecoder()

    print("\nTraining (autograd forward pass, not the delta engine -- the delta")
    print("engine is for fast INFERENCE-time updates AFTER training is done)...")
    history = train_link_predictor(model, decoder, train_store, train_edges, all_true_edges,
                                     epochs=150, lr=0.01, neg_ratio=1, seed=3)
    print(f"Loss: {history[0]:.4f} -> {history[-1]:.4f} over {len(history)} epochs")
    plot_training_loss(history, f"{FIG_DIR}/06_training_loss.png")

    val_metrics = evaluate_link_prediction(model, decoder, train_store, val_edges, val_neg)
    test_metrics = evaluate_link_prediction(model, decoder, train_store, test_edges, test_neg)

    print("\nValidation metrics:")
    for k in ["roc_auc", "average_precision", "accuracy", "precision", "recall", "f1"]:
        print(f"  {k:18s}: {val_metrics[k]:.4f}")
    print("\nTest metrics (this is the full-neighbor GraphSAGE baseline"
          " -- Section 16.3 of the proposal):")
    for k in ["roc_auc", "average_precision", "accuracy", "precision", "recall", "f1"]:
        print(f"  {k:18s}: {test_metrics[k]:.4f}")

    plot_roc_pr_curves(test_metrics["y_true"], test_metrics["y_score"],
                        f"{FIG_DIR}/07_link_prediction_roc_pr.png", title_prefix="Test Set - ")

    with torch.no_grad():
        features = {n: train_store.get_feature(n) for n in train_store.node_ids}
        embeddings = full_forward(model, train_store.node_ids, features, train_store.adjacency)
    plot_embeddings_pca(embeddings, community_of, f"{FIG_DIR}/08_embeddings_pca.png",
                         title="Trained Node Embeddings (PCA), colored by true community",
                         color_label="community id")

    print(f"\nSaved: 05_degree_distribution.png, 06_training_loss.png, "
          f"07_link_prediction_roc_pr.png, 08_embeddings_pca.png")
    return model, train_store, test_metrics


if __name__ == "__main__":
    part1_toy_graph()
    model, train_store, test_metrics = part2_train_and_evaluate()
    print("\nAll evaluation + visualization steps completed successfully.")
    print(f"Figures are in ./{FIG_DIR}/")
