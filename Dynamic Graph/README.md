# TGB DynaRangeGNN-Stream Auto Runner

This version creates a virtual environment, installs dependencies, and runs the TGB benchmark automatically.

## Files

- `tgb_dynarangegnn_benchmark.py` — main benchmark
- `requirements.txt` — required packages
- `setup_and_run_tgb_dynarange.py` — creates venv, installs packages, runs benchmark
- `run_windows.ps1` — Windows PowerShell runner
- `run_windows.bat` — Windows CMD runner
- `run_linux_mac.sh` — Linux/macOS runner

## Windows PowerShell

```powershell
powershell -ExecutionPolicy Bypass -File .\run_windows.ps1
```

## Windows CMD

```cmd
run_windows.bat
```

## Linux/macOS

```bash
chmod +x run_linux_mac.sh
./run_linux_mac.sh
```

## Direct Python

```bash
python setup_and_run_tgb_dynarange.py
```

## Custom benchmark arguments

Everything after `--` is passed to the benchmark.

```bash
python setup_and_run_tgb_dynarange.py -- --dataset tgbl-wiki --max_events 10000 --num_batches 80 --batch_size 100 --epochs 10 --device cpu
```

## Default quick run

```text
--dataset tgbl-wiki
--max_events 5000
--num_batches 50
--batch_size 100
--epochs 5
--device cpu
--out_dir tgb_dynarange_outputs
```

## Outputs

Results are saved in:

```text
tgb_dynarange_outputs/
```

Most important files:

- `13_summary_results.csv`
- `12_benchmark_results.csv`
- `06_time_full_vs_dynarange.png`
- `07_speedup.png`
- `08_dirty_nodes.png`
- `09_embedding_error.png`
- `10_space_usage.png`
- `11_prediction_roc_auc.png`

## GPU note

This setup uses normal pip installation and is safest for CPU runs. For CUDA, install the PyTorch build that matches your CUDA version manually, then run the benchmark.
