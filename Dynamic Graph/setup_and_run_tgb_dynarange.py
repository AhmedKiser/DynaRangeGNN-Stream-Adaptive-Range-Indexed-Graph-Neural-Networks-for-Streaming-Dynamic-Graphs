
from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
VENV_DIR = PROJECT_DIR / ".venv_tgb_dynarange"
REQUIREMENTS = PROJECT_DIR / "requirements.txt"
BENCHMARK = PROJECT_DIR / "tgb_dynarangegnn_benchmark.py"

DEFAULT_ARGS = [
    "--dataset", "tgbl-wiki",
    "--max_events", "5000",
    "--num_batches", "50",
    "--batch_size", "100",
    "--epochs", "5",
    "--device", "cpu",
    "--out_dir", "tgb_dynarange_outputs",
]

def run(cmd: list[str]) -> None:
    print("\n[RUN]", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(PROJECT_DIR))

def venv_python() -> Path:
    if platform.system().lower().startswith("win"):
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"

def create_venv() -> None:
    py = venv_python()
    if py.exists():
        print(f"[OK] Virtual environment already exists: {VENV_DIR}")
        return
    print(f"[SETUP] Creating virtual environment: {VENV_DIR}")
    run([sys.executable, "-m", "venv", str(VENV_DIR)])

def install_requirements() -> None:
    py = venv_python()
    print("[SETUP] Upgrading pip, setuptools, wheel")
    run([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    print("[SETUP] Installing project requirements")
    run([str(py), "-m", "pip", "install", "-r", str(REQUIREMENTS)])

def parse_args(argv: list[str]) -> list[str]:
    # Anything after -- is passed to the benchmark script.
    if "--" in argv:
        i = argv.index("--")
        custom = argv[i + 1:]
        if custom:
            return custom
    return DEFAULT_ARGS

def run_benchmark(args: list[str]) -> None:
    py = venv_python()
    if not BENCHMARK.exists():
        raise FileNotFoundError(f"Missing benchmark file: {BENCHMARK}")

    print("\n[START] Running benchmark")
    print("[ARGS]", " ".join(args))
    run([str(py), str(BENCHMARK), *args])

    out_dir = PROJECT_DIR / "tgb_dynarange_outputs"
    if out_dir.exists():
        print("\n[DONE] Results are saved in:")
        print(out_dir.resolve())
        for p in sorted(out_dir.iterdir()):
            print(" -", p.name)

def main() -> None:
    print("=" * 72)
    print("TGB DynaRangeGNN-Stream Auto Setup + Run")
    print("=" * 72)
    print(f"Project folder: {PROJECT_DIR}")
    print(f"Launcher Python: {sys.executable}")
    create_venv()
    install_requirements()
    benchmark_args = parse_args(sys.argv[1:])
    run_benchmark(benchmark_args)

if __name__ == "__main__":
    main()
