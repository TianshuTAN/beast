"""Download IBL camera timestamp files for a session via the ONE API.

Saves `_ibl_leftCamera.times.{eid}.npy` and `_ibl_rightCamera.times.{eid}.npy`
into the destination directory so that `extract_neural_data.py` can read them.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
from one.api import ONE

logging.basicConfig(level=logging.INFO)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eid", required=True, type=str)
    ap.add_argument("--one_cache_path", required=True, type=str)
    ap.add_argument("--output_path", required=True, type=str)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        username="intbrainlab",
        password="international",
        silent=True,
        cache_dir=args.one_cache_path,
    )

    for cam in ("left", "right"):
        dataset = f"alf/_ibl_{cam}Camera.times.npy"
        logging.info("Downloading %s for EID %s", dataset, args.eid)
        local_path = one.load_dataset(args.eid, dataset, download_only=True)
        local_path = Path(local_path)
        arr = np.load(local_path)
        dst = out_dir / f"_ibl_{cam}Camera.times.{args.eid}.npy"
        shutil.copyfile(local_path, dst)
        logging.info("Saved %s (shape=%s)", dst, arr.shape)


if __name__ == "__main__":
    main()
