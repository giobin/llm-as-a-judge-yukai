from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: container_python_entry.py <module> [args...]")

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))

    venv_lib = root / ".venv" / "lib"
    for site_packages in sorted(venv_lib.glob("python*/site-packages")):
        sys.path.append(str(site_packages))

    module = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
