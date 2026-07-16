# Cora + GraphSAGE Visualization Code

This code first visualizes and describes the Cora dataset, then shows how full GraphSAGE computation works.

## Install

```bash
pip install torch torch-geometric numpy pandas matplotlib scikit-learn networkx
```

## Run

```bash
python visualize_cora_graphsage.py --out_dir outputs --layout_nodes 200 --ego_node 0
```

## Outputs

- `01_class_distribution.png`
- `02_degree_distribution.png`
- `03_feature_pca_by_class.png`
- `04_cora_sample_subgraph.png`
- `05_graphsage_full_computation_flow.png`
- `06_graphsage_ego_neighborhood.png`
- `cora_dataset_summary.csv`
- `cora_degree_by_class.csv`
- `graphsage_full_computation_shapes.csv`
