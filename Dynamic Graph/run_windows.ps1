# Windows PowerShell
# Default:
#   powershell -ExecutionPolicy Bypass -File .\run_windows.ps1
#
# Custom:
#   powershell -ExecutionPolicy Bypass -File .\run_windows.ps1 -- --dataset tgbl-wiki --max_events 10000 --epochs 10

$ErrorActionPreference = "Stop"
python .\setup_and_run_tgb_dynarange.py @args
