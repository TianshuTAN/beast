from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from one.api import ONE

from beast.data.ibl_data_utils import (
    prepare_data,
    select_brain_regions,
    create_intervals,
    list_brain_regions,
    bin_spiking_data,
    bin_behaviors,
    align_data,
)

logging.basicConfig(level=logging.INFO)

DYNAMIC_VARS = [
    "wheel-speed",
    "licks",
    "left-whisker-motion-energy",
    "right-whisker-motion-energy",
    "left-nose-speed",
    "right-nose-speed",
    "left-paw-speed",
    "right-paw-speed",
]


def _beh_to_npz_keys(prefix: str, beh_dict: dict) -> dict:
    out = {}
    for name, arr in beh_dict.items():
        key = f"{prefix}_{str(name).replace('-', '_')}"
        out[key] = np.asarray(arr, dtype=float)
    return out


def _save_extract_bundle(
    *,
    out_root: Path,
    session_eid: str,
    spikes: np.ndarray,
    train_i: np.ndarray,
    val_i: np.ndarray,
    test_i: np.ndarray,
    train_int: np.ndarray,
    val_int: np.ndarray,
    test_int: np.ndarray,
    train_b: dict,
    val_b: dict,
    test_b: dict,
    meta: dict,
    extra_params: dict,
) -> None:
    out_dir = out_root / session_eid
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_payload = {
        "train_spikes": spikes[train_i],
        "val_spikes": spikes[val_i],
        "test_spikes": spikes[test_i],
        "train_intervals": np.asarray(train_int, dtype=np.float64),
        "val_intervals": np.asarray(val_int, dtype=np.float64),
        "test_intervals": np.asarray(test_int, dtype=np.float64),
        **_beh_to_npz_keys("train", train_b),
        **_beh_to_npz_keys("val", val_b),
        **_beh_to_npz_keys("test", test_b),
    }

    np.savez_compressed(out_dir / f"{session_eid}_aligned.npz", **npz_payload)

    with (out_dir / f"{session_eid}_meta.pkl").open("wb") as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)

    with (out_dir / "params.json").open("w", encoding="utf-8") as f:
        json.dump(extra_params, f, indent=2, default=str)

    logging.info("Saved bundle under %s", out_dir)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eid", type=str)
    ap.add_argument("--video_timestamps", type=str)
    ap.add_argument("--num_trials", type=int, default=400)
    ap.add_argument("--one_cache_path", type=str)
    ap.add_argument("--output_path", type=str, help="Directory to write npz/json/pkl bundle.")
    ap.add_argument("--n_workers", type=int, default=1)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.eid is None:
        raise ValueError("Session EID is required.")
    if args.one_cache_path is None:
        raise ValueError("--one_cache_path is required for ONE cache_dir.")
    if args.output_path is None:
        raise ValueError("--output_path is required to save the extract bundle.")
    if args.video_timestamps is None:
        raise ValueError("--video_timestamps is required.")
    if args.num_trials is None:
        raise ValueError("--num_trials is required to sample data from each session.")

    eid = args.eid

    params = {
        "interval_len": 1,
        "binsize": 1 / 60,  # 60 frames per second
        "single_region": False,
        "fr_thresh": 0.2,
    }

    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        username="intbrainlab",
        password="international",
        silent=True,
        cache_dir=args.one_cache_path,
    )

    logging.info("EID %s", eid)

    neural_dict, behave_dict, meta_dict, trials_dict, _ = prepare_data(
        one, eid, params, n_workers=args.n_workers
    )

    if neural_dict is None:
        logging.info("Skip EID %s Due to Missing Spike Data!", eid)
        sys.exit(0)

    regions, beryl_reg = list_brain_regions(neural_dict, **params)
    region_cluster_ids = select_brain_regions(neural_dict, beryl_reg, regions, **params)

    video_timestamps = Path(args.video_timestamps)
    left_ts_path = video_timestamps / f"_ibl_leftCamera.times.{eid}.npy"
    right_ts_path = video_timestamps / f"_ibl_rightCamera.times.{eid}.npy"
    left_timestamps = np.load(left_ts_path) if left_ts_path.is_file() else None
    right_timestamps = np.load(right_ts_path) if right_ts_path.is_file() else None

    ts_arrays = [a for a in (left_timestamps, right_timestamps) if a is not None]
    if not ts_arrays:
        logging.info(
            "Skip EID %s: missing %s and %s",
            eid,
            left_ts_path.name,
            right_ts_path.name,
        )
        sys.exit(0)
    if len(ts_arrays) == 2 and not np.array_equal(ts_arrays[0], ts_arrays[1]):
        raise AssertionError("Left and right camera timestamps must be synchronized.")
    min_timestamp = min(a.min() for a in ts_arrays)
    max_timestamp = max(a.max() for a in ts_arrays)
    min_timestamp = int(round(min_timestamp)) + 1
    max_timestamp = int(round(max_timestamp)) - 1

    intervals = create_intervals(min_timestamp, max_timestamp, params["interval_len"])

    assert len(intervals) >= args.num_trials, "Not enough intervals to sample from."

    rng = np.random.default_rng(42)
    trial_idxs = rng.choice(np.arange(len(intervals)), args.num_trials, replace=False)
    intervals = intervals[trial_idxs]

    bin_spikes, clusters_used_in_bins = bin_spiking_data(
        region_cluster_ids,
        neural_dict,
        intervals=intervals,
        trials_df=None,
        n_workers=args.n_workers,
        **params,
    )

    logging.info("Binned Spike Data: %s", bin_spikes.shape)

    mean_fr = bin_spikes.sum(1).mean(0) / params["interval_len"]
    keep_unit_idxs = np.argwhere(mean_fr > 1 / params["fr_thresh"]).flatten()
    bin_spikes = bin_spikes[..., keep_unit_idxs]
    logging.info(
        "# Responsive Units: %s / %s",
        bin_spikes.shape[-1],
        len(mean_fr),
    )

    meta_dict["cluster_regions"] = [meta_dict["cluster_regions"][idx] for idx in keep_unit_idxs]
    meta_dict["cluster_channels"] = [meta_dict["cluster_channels"][idx] for idx in keep_unit_idxs]
    meta_dict["cluster_depths"] = [meta_dict["cluster_depths"][idx] for idx in keep_unit_idxs]
    meta_dict["good_clusters"] = [meta_dict["good_clusters"][idx] for idx in keep_unit_idxs]
    meta_dict["uuids"] = [meta_dict["uuids"][idx] for idx in keep_unit_idxs]

    bin_beh, beh_mask = bin_behaviors(
        one,
        eid,
        DYNAMIC_VARS,
        intervals=intervals,
        trials_df=None,
        allow_nans=True,
        n_workers=args.n_workers,
        **params,
    )

    try:
        align_bin_spikes, align_bin_beh, target_mask, bad_trial_idxs = align_data(
            bin_spikes,
            bin_beh,
            list(bin_beh.keys()),
        )
    except ValueError as e:
        logging.info("Skip EID %s due to error: %s", eid, e)
        sys.exit(0)

    _bad = np.asarray(bad_trial_idxs, dtype=np.intp).ravel()
    aligned_intervals = np.delete(np.asarray(intervals), _bad, axis=0)
    if aligned_intervals.shape[0] != len(align_bin_spikes):
        raise RuntimeError(
            f"aligned_intervals length {aligned_intervals.shape[0]} != "
            f"align_bin_spikes trials {len(align_bin_spikes)}"
        )

    num_trials = len(aligned_intervals)
    rng = np.random.default_rng(42)
    perm = rng.choice(np.arange(num_trials), num_trials, replace=False)

    train_idxs = perm[: int(0.7 * num_trials)]
    val_idxs = perm[int(0.7 * num_trials) : int(0.8 * num_trials)]
    test_idxs = perm[int(0.8 * num_trials) :]

    train_beh, val_beh, test_beh = {}, {}, {}
    for beh in align_bin_beh.keys():
        train_beh[beh] = align_bin_beh[beh][train_idxs]
        val_beh[beh] = align_bin_beh[beh][val_idxs]
        test_beh[beh] = align_bin_beh[beh][test_idxs]

    train_intervals = aligned_intervals[train_idxs]
    val_intervals = aligned_intervals[val_idxs]
    test_intervals = aligned_intervals[test_idxs]

    _save_extract_bundle(
        out_root=Path(args.output_path),
        session_eid=eid,
        spikes=align_bin_spikes,
        train_i=train_idxs,
        val_i=val_idxs,
        test_i=test_idxs,
        train_int=train_intervals,
        val_int=val_intervals,
        test_int=test_intervals,
        train_b=train_beh,
        val_b=val_beh,
        test_b=test_beh,
        meta=meta_dict,
        extra_params=params,
    )

    logging.info("Finished EID: %s", eid)


if __name__ == "__main__":
    main()
