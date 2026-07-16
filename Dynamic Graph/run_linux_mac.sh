#!/usr/bin/env bash
set -e

# Linux/macOS
# Default:
#   chmod +x run_linux_mac.sh
#   ./run_linux_mac.sh
#
# Custom:
#   ./run_linux_mac.sh -- --dataset tgbl-wiki --max_events 10000 --epochs 10

python3 setup_and_run_tgb_dynarange.py "$@"
