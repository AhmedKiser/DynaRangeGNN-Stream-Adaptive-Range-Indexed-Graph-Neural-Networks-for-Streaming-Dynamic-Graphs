# DynaDeltaGNN

This code implements :
- Static 2-layer GraphSAGE-Mean reference model
- Graph store using adjacency list + edge hash + degree table
- Embedding cache `H[0], H[1], H[2]`
- Neighbor-sum cache `M[1], M[2]`
- Exact local update engine for:
  - node-feature update
  - edge insertion
  - edge deletion
  - mixed event batch
  - high-degree hub stress test
- Correctness tests against full recomputation

## Setup

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
pytest -q
python run_demo.py
```

## Main success criterion

For every event batch:

```text
max_abs_error(local_update, full_recompute) < 1e-6
```
