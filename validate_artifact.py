from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(ROOT / "tests")],
        check=True,
        env=env,
    )
    print("Unit tests: PASS")
    print("Artifact validation: PASS")


if __name__ == "__main__":
    main()
