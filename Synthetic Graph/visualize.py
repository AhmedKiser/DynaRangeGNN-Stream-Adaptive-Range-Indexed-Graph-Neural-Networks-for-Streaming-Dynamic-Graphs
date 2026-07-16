"""
visualize.py
------------
Dataset and result visualization. Produces PNG figures for:

  - the toy graph itself (nodes colored by activity score)
  - a number-line style illustration of a range query (Section 7.3)
  - the degree distribution of a synthetic dataset
  - a 2D (PCA) projection of learned node embeddings
  - ROC and Precision-Recall curves for the link-prediction evaluation

All functions save a PNG to `filename` and don't display interactively,
so they work headlessly in the sandbox.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, precision_recall_curve, auc

ACCENT = "#2E5C8A"
ACCENT2 = "#B5651D"
GREY = "#888888"


def plot_toy_graph(store, scores, filename, title="Graph"):
    G = nx.Graph()
    G.add_nodes_from(store.node_ids)
    for e in store.edge_hash:
        u, v = tuple(e)
        G.add_edge(u, v)
    pos = nx.spring_layout(G, seed=0)

    node_colors = [scores[n] for n in G.nodes()]
    fig, ax = plt.subplots(figsize=(6, 5))
    nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap="viridis",
                                    node_size=900, ax=ax, vmin=40, vmax=100)
    nx.draw_networkx_edges(G, pos, ax=ax, width=1.5, edge_color=GREY)
    nx.draw_networkx_labels(G, pos, ax=ax, font_color="white", font_weight="bold")
    cbar = fig.colorbar(nodes, ax=ax)
    cbar.set_label("Activity score")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_range_query_illustration(scores, target_node, low, high, filename, title="Range Query"):
    """Number-line illustration of NeighborsInRange(target_node, low, high)."""
    fig, ax = plt.subplots(figsize=(8, 2.6))
    names = list(scores.keys())
    values = [scores[n] for n in names]

    ax.axvspan(low, high, color="#EAF1F8", zorder=0)
    ax.axvline(low, color=ACCENT, linestyle="--", linewidth=1)
    ax.axvline(high, color=ACCENT, linestyle="--", linewidth=1)

    for n, v in zip(names, values):
        in_range = (low <= v <= high) and n != target_node
        is_target = n == target_node
        if is_target:
            color = "#B5651D"
        elif in_range:
            color = ACCENT
        else:
            color = "#AAAAAA"
        ax.scatter(v, 0, s=280, color=color, zorder=3, edgecolor="black")
        ax.annotate(n, (v, 0), textcoords="offset points", xytext=(0, 14),
                    ha="center", fontweight="bold")
        ax.annotate(f"{v:.0f}", (v, 0), textcoords="offset points", xytext=(0, -20),
                    ha="center", fontsize=8, color="#555")

    ax.set_yticks([])
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel("Activity score")
    ax.set_title(f"{title}\n(orange = query node, blue = in range, grey = out of range)", fontsize=10)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_degree_distribution(store, filename, title="Degree Distribution"):
    degrees = [store.degree(n) for n in store.node_ids]
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = range(min(degrees), max(degrees) + 2)
    ax.hist(degrees, bins=bins, color=ACCENT, edgecolor="white")
    ax.set_xlabel("Node degree")
    ax.set_ylabel("Number of nodes")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_embeddings_pca(embeddings, color_values, filename, title="Node Embeddings (PCA)", color_label="value"):
    node_ids = list(embeddings.keys())
    X = np.stack([embeddings[n].detach().numpy() for n in node_ids])
    X2 = PCA(n_components=2, random_state=0).fit_transform(X) if X.shape[1] > 2 else X

    colors = [color_values[n] for n in node_ids]
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(X2[:, 0], X2[:, 1], c=colors, cmap="viridis", s=70,
                     edgecolor="black", linewidth=0.4)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(color_label)
    ax.set_title(title)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_roc_pr_curves(y_true, y_score, filename, title_prefix=""):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(rec, prec)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(fpr, tpr, color=ACCENT, lw=2, label=f"AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"{title_prefix}ROC Curve")
    axes[0].legend(loc="lower right")

    axes[1].plot(rec, prec, color=ACCENT2, lw=2, label=f"AUC \u2248 {pr_auc:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"{title_prefix}Precision-Recall Curve")
    axes[1].legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_training_loss(history, filename, title="Training Loss"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(history) + 1), history, color=ACCENT, lw=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary cross-entropy loss")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
