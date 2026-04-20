"""Download raw left/right camera videos for an IBL session via the ONE API.

Copies the downloaded mp4 files into the destination directory using names
that include the EID, so downstream tools can locate them.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from one.api import ONE

logging.basicConfig(level=logging.INFO)

VIDEO_DATASETS = {
    "left": "raw_video_data/_iblrig_leftCamera.raw.mp4",
    "right": "raw_video_data/_iblrig_rightCamera.raw.mp4",
}


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

    for cam, dataset in VIDEO_DATASETS.items():
        logging.info("Downloading %s for EID %s", dataset, args.eid)
        local_path = one.load_dataset(args.eid, dataset, download_only=True)
        local_path = Path(local_path)
        dst = out_dir / f"_iblrig_{cam}Camera.raw.{args.eid}.mp4"
        shutil.copyfile(local_path, dst)
        size_gb = dst.stat().st_size / 1e9
        logging.info("Saved %s (%.2f GB)", dst, size_gb)


if __name__ == "__main__":
    main()
