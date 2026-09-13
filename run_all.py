from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(script, *args):
    cmd = [sys.executable, str(HERE / script), *map(str, args)]
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    run("context_dependence.py")
    run("duplicate_symmetry.py")
    run("controlled_order.py", "--reference-orders", "1000", "--explain-n", "300")
    run("rff_sensitivity.py", "--n", "240", "--orders", "100")


if __name__ == "__main__":
    main()
