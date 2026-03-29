"""
clean.py -- Clear cached models, figures, reports, or all outputs.

Usage:
    python -m asset_allocation.clean --what cache
    python -m asset_allocation.clean --what figures
    python -m asset_allocation.clean --what reports
    python -m asset_allocation.clean --what all
"""

from __future__ import annotations
import argparse
import shutil
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "outputs"

DIRS = {
    "cache": OUTPUT_DIR / "cache",
    "figures": OUTPUT_DIR / "figures",
    "reports": OUTPUT_DIR / "reports",
    "testoutput": OUTPUT_DIR / "testoutput",
}


def clean(what: str, verbose: bool = True):
    """Clean specified output directory."""
    if what == "all":
        targets = list(DIRS.values())
    elif what in DIRS:
        targets = [DIRS[what]]
    else:
        print(f"Unknown target: {what}. Options: {', '.join(list(DIRS.keys()) + ['all'])}")
        return

    for d in targets:
        if d.exists():
            n_files = sum(1 for _ in d.glob("*") if _.is_file())
            shutil.rmtree(str(d))
            d.mkdir(parents=True, exist_ok=True)
            if verbose:
                print(f"  Cleared {d.name}/ ({n_files} files)")
        else:
            d.mkdir(parents=True, exist_ok=True)
            if verbose:
                print(f"  Created {d.name}/ (was missing)")


def main():
    parser = argparse.ArgumentParser(description="Clean output directories")
    parser.add_argument("--what", "-w", required=True,
                        help="What to clean: cache, figures, reports, testoutput, all")
    args = parser.parse_args()
    clean(args.what)


if __name__ == "__main__":
    main()
