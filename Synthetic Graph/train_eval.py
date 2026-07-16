"""
train_eval.py
--------------
Phase 6 groundwork ("Add the link-prediction decoder") plus the accuracy
side of the evaluation plan (Section 16.4 of the proposal): ROC-AUC,
Average Precision, Precision, Recall, F1, Accuracy.

Establishes a FULL-NEIGHBOR GraphSAGE baseline -- the same reference
point the proposal's baseline table (Section 16.3) calls for -- against
which later range-aware sampling can be compared.

Important evaluation detail: validation/test edges are held out of the
message-passing graph entirely (a fresh DynamicGraphStore built only from
the training edges), so the model can't "see" the edges it's being asked
to predict. This is standard practice for transductive link prediction.
"""

import random
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score,
)

from graph_store import DynamicGraphStore
from gnn_model import full_forward


def split_edges(store, val_frac=0.1, test_frac=0.2, seed=0):
    """Randomly partitions the graph's edges into train/val/test sets."""
    edges = [tuple(e) for e in store.edge_hash]
    rng = random.Random(seed)
    rng.shuffle(edges)
    n = len(edges)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test_edges = edges[:n_test]
    val_edges = edges[n_test:n_test + n_val]
    train_edges = edges[n_test + n_val:]
    return train_edges, val_edges, test_edges


def build_train_store(full_store, train_edges):
    """A fresh graph store containing only the training edges (same nodes
    and features as the full graph) -- this is what the GNN is allowed
    to use for message passing during both training and evaluation."""
    train_store = DynamicGraphStore(node_ids=full_store.node_ids, feature_dim=full_store.feature_dim)
    train_store.features = full_store.features.clone()
    for u, v in train_edges:
        train_store.add_edge(u, v)
    return train_store


def sample_negative_edges(store, num, exclude_edges, rng):
    """Random pairs of distinct nodes that are NOT true edges in
    `exclude_edges` (which should include train+val+test positives)."""
    exclude = {frozenset(e) for e in exclude_edges}
    node_ids = store.node_ids
    negatives = []
    tries, max_tries = 0, max(200, num * 50)
    while len(negatives) < num and tries < max_tries:
        tries += 1
        u, v = rng.choice(node_ids), rng.choice(node_ids)
        if u == v:
            continue
        key = frozenset((u, v))
        if key in exclude:
            continue
        negatives.append((u, v))
        exclude.add(key)
    return negatives


def train_link_predictor(model, decoder, train_store, train_pos_edges, exclude_edges,
                          epochs=150, lr=0.01, neg_ratio=1, seed=0):
    """Trains the GNN + dot-product decoder JOINTLY (including the
    decoder's bias term) with binary cross-entropy on positive vs.
    sampled-negative edges. Uses full_forward() (autograd enabled) --
    NOT the delta engine, which is an inference-time cache, not a
    training-time computation graph."""
    params = list(model.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    rng = random.Random(seed)
    node_ids = train_store.node_ids
    features = {n: train_store.get_feature(n) for n in node_ids}

    history = []
    for _epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        embeddings = full_forward(model, node_ids, features, train_store.adjacency)

        neg_edges = sample_negative_edges(
            train_store, len(train_pos_edges) * neg_ratio, exclude_edges, rng
        )
        pos_logits = decoder.logits(embeddings, train_pos_edges)
        neg_logits = decoder.logits(embeddings, neg_edges)

        logits = torch.cat([pos_logits, neg_logits])
        labels = torch.cat([torch.ones(len(pos_logits)), torch.zeros(len(neg_logits))])

        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()
        history.append(loss.item())

    return history


def evaluate_link_prediction(model, decoder, store, pos_edges, neg_edges):
    """Full-batch evaluation. Returns a dict of metrics plus the raw
    y_true / y_score arrays (handy for plotting ROC / PR curves)."""
    model.eval()
    features = {n: store.get_feature(n) for n in store.node_ids}
    with torch.no_grad():
        embeddings = full_forward(model, store.node_ids, features, store.adjacency)
        probs_pos = decoder.probs(embeddings, pos_edges)
        probs_neg = decoder.probs(embeddings, neg_edges)

    y_true = [1] * len(pos_edges) + [0] * len(neg_edges)
    y_score = torch.cat([probs_pos, probs_neg]).tolist()
    y_pred = [1 if s >= 0.5 else 0 for s in y_score]

    return {
        "roc_auc": roc_auc_score(y_true, y_score),
        "average_precision": average_precision_score(y_true, y_score),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "y_true": y_true,
        "y_score": y_score,
    }
