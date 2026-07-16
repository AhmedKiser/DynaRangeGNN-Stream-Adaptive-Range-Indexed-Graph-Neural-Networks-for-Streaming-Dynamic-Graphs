# DynaRangeGNN-Stream — Phase 1–2 Prototype

A working implementation of the first two phases of the `DynaRangeGNN-Stream`
roadmap:

- **Phase 1** — exact local ("delta") updates for a GraphSAGE-Mean GNN
  (`delta_engine.py`, `gnn_model.py`)
- **Phase 2** — a global order-statistic tree over each node's retrieval
  key, ready to feed the range-query layer built in later phases
  (`order_statistic_tree.py`)

It reproduces the six-student worked example from the project proposal
end-to-end and proves, by direct comparison against a from-scratch
recomputation, that the local updates are exact.

## Files

| File | Purpose |
|---|---|
| `graph_store.py` | Dynamic graph store: adjacency lists, edge hash table, feature storage |
| `order_statistic_tree.py` | Sorted retrieval-key index (`insert`, `delete`, `update_key`, `range_query`) |
| `gnn_model.py` | 2-layer GraphSAGE-Mean encoder in PyTorch, plus `full_forward()` for training |
| `delta_engine.py` | `DynaDeltaGNN`: dirty-frontier local update engine + exactness check |
| `link_predictor.py` | `DotProductDecoder`: learnable-bias dot-product link scorer |
| `train_eval.py` | Edge splitting, negative sampling, training loop, evaluation metrics |
| `visualize.py` | Matplotlib/networkx plotting utilities (graph, range queries, embeddings, ROC/PR) |
| `demo.py` | Reproduces the proposal's running example + a scaling benchmark |
| `run_evaluation.py` | Trains + evaluates link prediction, saves all figures to `./figures/` |
| `test_correctness.py` | Automated exactness tests (feature/edge events, mixed sequences) |
| `requirements.txt` | `torch`, `sortedcontainers`, `scikit-learn`, `networkx`, `matplotlib` |

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python3 demo.py              # walks through the toy example + scaling benchmark
python3 test_correctness.py  # automated correctness tests
python3 run_evaluation.py    # dataset visualization + link-prediction training/evaluation
```

## `run_evaluation.py` — what it produces

**Part 1 (toy graph).** Visualizes the six-student graph before/after the
`F: 55 -> 75` event, plus a number-line illustration of
`NeighborsInRange(E, 64, 84)` before and after — a direct picture of the
range-membership change from Section 4.3 of the proposal.

**Part 2 (synthetic dataset, 300 nodes / 4 communities).** A graph is
generated with real learnable structure (nodes belong to a community,
features are noisy community centroids, edges are ~10x more likely
within a community than across one) — a plain Erdos-Renyi random graph
has no structure for a GNN to learn from, so this is intentionally more
realistic. Edges are split into train/val/test, negatives are sampled,
and a GraphSAGE-Mean encoder + dot-product decoder are trained with
binary cross-entropy, then evaluated with ROC-AUC, Average Precision,
Accuracy, Precision, Recall, and F1 — this is the **full-neighbor
GraphSAGE baseline** called for in Section 16.3 of the proposal, which
later range-aware sampling should be compared against.

All figures are written to `./figures/`:

| File | Shows |
|---|---|
| `01/03_toy_graph_*.png` | The toy graph, node color = activity score, before/after the event |
| `02/04_range_query_*.png` | `NeighborsInRange(E, 64, 84)` before/after, as a number line |
| `05_degree_distribution.png` | Degree histogram of the synthetic dataset |
| `06_training_loss.png` | Training loss curve |
| `07_link_prediction_roc_pr.png` | ROC curve and Precision-Recall curve on the held-out test edges |
| `08_embeddings_pca.png` | Trained embeddings projected to 2D (PCA), colored by true community — a working GNN should visibly cluster communities together |

## A modeling detail worth knowing

`DotProductDecoder` includes a learnable bias term. Without it, every
prediction would come out "positive": GraphSAGE-Mean's layers end in a
ReLU, so embeddings are always non-negative, which makes every raw dot
product non-negative too, and `sigmoid(x) >= 0.5` whenever `x >= 0`.
That shows up as recall stuck at 1.0 and accuracy stuck at the class
base rate, even though threshold-free metrics (ROC-AUC, AP) look fine.
The bias lets training move the decision boundary to where it actually
belongs.

## What "exactness" means here

Per Section 13 of the proposal, `DynaDeltaGNN`'s exactness claim is:
**the local update equals a full recomputation of the same GraphSAGE-Mean
model over the current graph.** `verify_exactness()` checks this directly
by recomputing every embedding from scratch and comparing it, node by
node, against the delta-maintained cache (`max abs diff` should be `0.0`
up to floating-point noise).

## Known scope limits (by design, for Phase 1–2)

- Node set is fixed; only **edges** and **features** are dynamic (node
  insertion/removal is out of scope until later phases).
- The order-statistic tree is used directly for range queries in the demo
  (equivalent to the proposal's "light-node" path). The **heavy-node
  ordered range index** and **workload-adaptive indexing** rule (Phase 3–4)
  are not implemented yet — see the accompanying report for next steps.
- `sortedcontainers.SortedList` is used in place of a hand-rolled balanced
  BST/treap; it gives the same amortized O(log n) behaviour and is easier
  to review and trust in a first prototype.
