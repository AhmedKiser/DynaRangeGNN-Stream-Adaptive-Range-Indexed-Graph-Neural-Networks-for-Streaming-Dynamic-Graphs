"""
Cora visualization + full GraphSAGE computation demo.

This script does three things:
1) Loads and describes the Cora citation dataset.
2) Saves clear visualizations of the dataset.
3) Demonstrates full 2-layer GraphSAGE-Mean computation with caches.

Install:
    pip install torch torch-geometric numpy pandas matplotlib scikit-learn networkx

Run:
    python visualize_cora_graphsage.py --out_dir outputs --layout_nodes 200 --ego_node 0

Outputs:
    01_class_distribution.png
    02_degree_distribution.png
    03_feature_pca_by_class.png
    04_cora_sample_subgraph.png
    05_graphsage_full_computation_flow.png
    06_graphsage_ego_neighborhood.png
    cora_dataset_summary.csv
    cora_degree_by_class.csv
    graphsage_full_computation_shapes.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.decomposition import PCA


def load_cora(root: str):
    """Load Cora using PyTorch Geometric."""
    try:
        from torch_geometric.datasets import Planetoid
        from torch_geometric.transforms import NormalizeFeatures
    except ImportError as exc:
        raise ImportError(
            "PyTorch Geometric is required. Install with: pip install torch-geometric"
        ) from exc

    dataset = Planetoid(root=root, name="Cora", transform=NormalizeFeatures())
    return dataset, dataset[0]


def undirected_edges(edge_index: Tensor) -> List[Tuple[int, int]]:
    """Convert PyG directed edge_index to unique undirected edges."""
    edges = set()
    for u, v in edge_index.cpu().t().tolist():
        if u == v:
            continue
        a, b = (int(u), int(v)) if u < v else (int(v), int(u))
        edges.add((a, b))
    return sorted(edges)


def build_adjacency(num_nodes: int, edges: Iterable[Tuple[int, int]]) -> List[Set[int]]:
    adj = [set() for _ in range(num_nodes)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def build_networkx_graph(num_nodes: int, edges: List[Tuple[int, int]]) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    g.add_edges_from(edges)
    return g


def summarize_cora(dataset, data, edges: List[Tuple[int, int]], adj: List[Set[int]], out_dir: Path) -> None:
    x = data.x.cpu()
    y = data.y.cpu().numpy()
    degrees = np.array([len(nbrs) for nbrs in adj])
    nonzero_features = (x > 0).sum(dim=1).numpy()

    summary = {
        "num_nodes": int(data.num_nodes),
        "num_directed_edges_in_pyg_edge_index": int(data.edge_index.size(1)),
        "num_unique_undirected_edges": int(len(edges)),
        "num_features_per_node": int(data.num_node_features),
        "num_classes": int(dataset.num_classes),
        "average_degree": float(degrees.mean()),
        "median_degree": float(np.median(degrees)),
        "max_degree": int(degrees.max()),
        "min_degree": int(degrees.min()),
        "average_nonzero_features_per_node": float(nonzero_features.mean()),
        "median_nonzero_features_per_node": float(np.median(nonzero_features)),
        "train_nodes": int(data.train_mask.sum().item()) if hasattr(data, "train_mask") else None,
        "validation_nodes": int(data.val_mask.sum().item()) if hasattr(data, "val_mask") else None,
        "test_nodes": int(data.test_mask.sum().item()) if hasattr(data, "test_mask") else None,
    }

    pd.DataFrame([summary]).to_csv(out_dir / "cora_dataset_summary.csv", index=False)

    rows = []
    for c in range(dataset.num_classes):
        idx = np.where(y == c)[0]
        rows.append({
            "class_id": c,
            "num_nodes": int(len(idx)),
            "mean_degree": float(degrees[idx].mean()) if len(idx) else 0.0,
            "median_degree": float(np.median(degrees[idx])) if len(idx) else 0.0,
            "mean_nonzero_features": float(nonzero_features[idx].mean()) if len(idx) else 0.0,
        })
    pd.DataFrame(rows).to_csv(out_dir / "cora_degree_by_class.csv", index=False)

    print("\nCora dataset summary")
    print("--------------------")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\nPlain meaning")
    print("-------------")
    print("Node = one paper")
    print("Edge = citation link between two papers")
    print("Feature vector = normalized bag-of-words paper representation")
    print("Label = research topic/class of the paper")
    print("Full GraphSAGE recomputes every node at every layer.")


def plot_class_distribution(data, num_classes: int, out_dir: Path) -> None:
    labels = data.y.cpu().numpy()
    counts = np.bincount(labels, minlength=num_classes)
    plt.figure(figsize=(8, 5))
    plt.bar([str(i) for i in range(num_classes)], counts)
    plt.xlabel("Class ID")
    plt.ylabel("Number of papers")
    plt.title("Cora class distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "01_class_distribution.png", dpi=300)
    plt.close()


def plot_degree_distribution(adj: List[Set[int]], out_dir: Path) -> None:
    degrees = np.array([len(nbrs) for nbrs in adj])
    plt.figure(figsize=(8, 5))
    plt.hist(degrees, bins=40)
    plt.xlabel("Degree")
    plt.ylabel("Number of nodes")
    plt.title("Cora degree distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "02_degree_distribution.png", dpi=300)
    plt.close()


def plot_feature_pca(data, out_dir: Path) -> None:
    x = data.x.cpu().numpy()
    y = data.y.cpu().numpy()
    xy = PCA(n_components=2, random_state=42).fit_transform(x)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(xy[:, 0], xy[:, 1], c=y, s=10)
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.title("Cora node features projected with PCA")
    plt.colorbar(scatter, label="Class ID")
    plt.tight_layout()
    plt.savefig(out_dir / "03_feature_pca_by_class.png", dpi=300)
    plt.close()


def plot_sample_subgraph(g: nx.Graph, data, out_dir: Path, layout_nodes: int = 200) -> None:
    degrees = dict(g.degree())
    nodes = sorted(degrees, key=degrees.get, reverse=True)[:layout_nodes]
    sg = g.subgraph(nodes).copy()
    labels = data.y.cpu().numpy()
    node_classes = [int(labels[n]) for n in sg.nodes()]
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(sg, seed=42, iterations=50)
    nx.draw_networkx_edges(sg, pos, width=0.4, alpha=0.4)
    nx.draw_networkx_nodes(sg, pos, node_size=35, node_color=node_classes)
    plt.title(f"Cora sample subgraph: top {layout_nodes} high-degree nodes")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "04_cora_sample_subgraph.png", dpi=300)
    plt.close()


def plot_graphsage_flow(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")
    boxes = [
        ("Input features\nH[0] = X", 0.05, 0.55),
        ("Layer 1 neighbor mean\nmean(N(v), H[0])", 0.28, 0.55),
        ("Layer 1 embedding\nH[1]", 0.51, 0.55),
        ("Layer 2 neighbor mean\nmean(N(v), H[1])", 0.28, 0.20),
        ("Layer 2 embedding\nH[2]", 0.51, 0.20),
        ("Prediction\nscore(u, v)", 0.75, 0.37),
    ]
    for text, x, y in boxes:
        rect = plt.Rectangle((x, y), 0.18, 0.18, fill=False)
        ax.add_patch(rect)
        ax.text(x + 0.09, y + 0.09, text, ha="center", va="center", fontsize=10)
    arrows = [
        ((0.23, 0.64), (0.28, 0.64)),
        ((0.46, 0.64), (0.51, 0.64)),
        ((0.60, 0.55), (0.37, 0.38)),
        ((0.46, 0.29), (0.51, 0.29)),
        ((0.69, 0.29), (0.75, 0.42)),
        ((0.69, 0.64), (0.75, 0.50)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->"))
    ax.text(
        0.05, 0.05,
        "Full computation means every node passes through Layer 1 and Layer 2 again.\n"
        "DynaDeltaGNN/DynaRangeGNN later avoids this by recomputing only dirty nodes.",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "05_graphsage_full_computation_flow.png", dpi=300)
    plt.close()


def plot_ego_neighborhood(g: nx.Graph, data, ego_node: int, out_dir: Path) -> None:
    ego = nx.ego_graph(g, ego_node, radius=2)
    labels = data.y.cpu().numpy()
    node_classes = [int(labels[n]) for n in ego.nodes()]
    plt.figure(figsize=(8, 7))
    pos = nx.spring_layout(ego, seed=42, iterations=80)
    nx.draw_networkx_edges(ego, pos, width=0.8, alpha=0.5)
    nx.draw_networkx_nodes(ego, pos, node_size=80, node_color=node_classes)
    nx.draw_networkx_nodes(ego, pos, nodelist=[ego_node], node_size=250)
    nx.draw_networkx_labels(ego, pos, labels={ego_node: f"target\n{ego_node}"}, font_size=8)
    plt.title(f"Two-hop neighborhood for a 2-layer GraphSAGE model: target node {ego_node}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "06_graphsage_ego_neighborhood.png", dpi=300)
    plt.close()


class MeanSAGELayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.neighbor_linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, h_self: Tensor, neighbor_mean: Tensor) -> Tensor:
        return F.relu(self.self_linear(h_self) + self.neighbor_linear(neighbor_mean))


class TwoLayerGraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.layer1 = MeanSAGELayer(in_dim, hidden_dim)
        self.layer2 = MeanSAGELayer(hidden_dim, out_dim)

    @staticmethod
    def neighbor_mean(h: Tensor, adj: List[Set[int]]) -> Tensor:
        out = torch.zeros_like(h)
        for v, nbrs in enumerate(adj):
            if nbrs:
                idx = torch.tensor(sorted(nbrs), dtype=torch.long, device=h.device)
                out[v] = h[idx].mean(dim=0)
        return out

    def full_forward_with_cache(self, x: Tensor, adj: List[Set[int]]) -> Dict[str, List[Tensor]]:
        h0 = x
        mean1 = self.neighbor_mean(h0, adj)
        h1 = self.layer1(h0, mean1)
        mean2 = self.neighbor_mean(h1, adj)
        h2 = self.layer2(h1, mean2)
        return {"H": [h0, h1, h2], "neighbor_mean": [None, mean1, mean2]}


def demonstrate_full_graphsage(data, adj: List[Set[int]], out_dir: Path) -> None:
    torch.manual_seed(42)
    x = data.x.float()
    model = TwoLayerGraphSAGE(in_dim=x.size(1), hidden_dim=32, out_dim=16)
    model.eval()
    with torch.no_grad():
        cache = model.full_forward_with_cache(x, adj)
    rows = [
        {"object": "Input features H[0]", "shape": str(tuple(cache["H"][0].shape)), "meaning": "Original Cora paper features"},
        {"object": "Layer 1 neighbor mean", "shape": str(tuple(cache["neighbor_mean"][1].shape)), "meaning": "Mean of neighbor input features"},
        {"object": "Layer 1 embeddings H[1]", "shape": str(tuple(cache["H"][1].shape)), "meaning": "First learned representation"},
        {"object": "Layer 2 neighbor mean", "shape": str(tuple(cache["neighbor_mean"][2].shape)), "meaning": "Mean of neighbor layer-1 embeddings"},
        {"object": "Layer 2 embeddings H[2]", "shape": str(tuple(cache["H"][2].shape)), "meaning": "Final node embeddings"},
    ]
    pd.DataFrame(rows).to_csv(out_dir / "graphsage_full_computation_shapes.csv", index=False)
    print("\nFull GraphSAGE computation")
    print("--------------------------")
    for row in rows:
        print(f"{row['object']}: {row['shape']} -> {row['meaning']}")
    print("\nFull computation recomputes Layer 1 and Layer 2 for every node.")
    print("This is correct but wasteful after a small graph event.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./data")
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--layout_nodes", type=int, default=200)
    parser.add_argument("--ego_node", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset, data = load_cora(args.root)
    edges = undirected_edges(data.edge_index)
    adj = build_adjacency(data.num_nodes, edges)
    g = build_networkx_graph(data.num_nodes, edges)

    summarize_cora(dataset, data, edges, adj, out_dir)
    plot_class_distribution(data, dataset.num_classes, out_dir)
    plot_degree_distribution(adj, out_dir)
    plot_feature_pca(data, out_dir)
    plot_sample_subgraph(g, data, out_dir, layout_nodes=args.layout_nodes)
    plot_graphsage_flow(out_dir)
    plot_ego_neighborhood(g, data, ego_node=args.ego_node, out_dir=out_dir)
    demonstrate_full_graphsage(data, adj, out_dir)

    print("\nSaved outputs to:", out_dir.resolve())
    print("\nGenerated files:")
    for path in sorted(out_dir.iterdir()):
        print("-", path.name)


if __name__ == "__main__":
    main()
