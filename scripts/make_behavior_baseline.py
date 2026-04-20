"""Build a `z_trials.npz` from behavior traces in `*_aligned.npz`.

The encoding/decoding eval (`encoding_decoding_code/test.py`) reads a latent
file with key `z_trials_time` of shape `(K, T, V, D)` and slices it into
train/val/test splits using the lengths of `train_intervals`/`val_intervals`/
`test_intervals` from the neural npz. By stacking the per-split behavior
traces in [train, val, test] order and reshaping to `(K, T, 1, D)`, we get a
drop-in "behavior-traces" baseline that can be compared against learned video
latents under the same evaluation pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Behavior name sets (as they appear in `*_aligned.npz`, per split prefix).
BEHAVIOR_SETS = {
    "7d": (
        "wheel_speed",
        "left_whisker_motion_energy",
        "right_whisker_motion_energy",
        "left_nose_speed",
        "right_nose_speed",
        "left_camera_left_paw_speed",
        "left_camera_right_paw_speed",
    ),
    # Mentor's 6d: paw 2d per view × 2 views + nose 1d per view × 2 views
    "6d": (
        "left_camera_left_paw_speed",
        "left_camera_right_paw_speed",
        "right_camera_left_paw_speed",
        "right_camera_right_paw_speed",
        "left_nose_speed",
        "right_nose_speed",
    ),
}
BEHAVIORS = BEHAVIOR_SETS["7d"]  # default, overridden by --behavior_set


def _split_block(npz, split: str, behaviors) -> np.ndarray:
    cols = [np.asarray(npz[f"{split}_{b}"], dtype=np.float32) for b in behaviors]
    # each shape (K_split, T); stack to (K_split, T, D)
    return np.stack(cols, axis=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eid", required=True)
    ap.add_argument("--neural_input_dir", required=True)
    ap.add_argument("--latent_output_dir", required=True)
    ap.add_argument("--behavior_set", choices=list(BEHAVIOR_SETS.keys()), default="7d")
    args = ap.parse_args()
    behaviors = BEHAVIOR_SETS[args.behavior_set]

    neural_path = Path(args.neural_input_dir) / args.eid / f"{args.eid}_aligned.npz"
    if not neural_path.is_file():
        raise FileNotFoundError(neural_path)

    npz = np.load(neural_path, allow_pickle=True)

    train_block = _split_block(npz, "train", behaviors)
    val_block = _split_block(npz, "val", behaviors)
    test_block = _split_block(npz, "test", behaviors)

    # NaN-fill (some sessions have missing behavior frames; encoder normalization
    # will choke on NaNs). Fall back to zero — these rows are rare and stable.
    for blk in (train_block, val_block, test_block):
        np.nan_to_num(blk, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # Concatenate in the order test.py expects: [train, val, test].
    z_concat = np.concatenate([train_block, val_block, test_block], axis=0)

    # Add a fake "view" axis so the shape matches (K, T, V, D).
    z_trials_time = z_concat[:, :, None, :]

    trial_split = np.array(
        [len(train_block), len(val_block), len(test_block)], dtype=np.int64
    )

    out_dir = Path(args.latent_output_dir) / args.eid
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "z_trials.npz"
    np.savez(out_path, z_trials_time=z_trials_time, trial_split=trial_split)

    K, T, V, D = z_trials_time.shape
    print(f"Saved {out_path}")
    print(f"  z_trials_time shape: {z_trials_time.shape}")
    print(f"  trial_split: train={trial_split[0]} val={trial_split[1]} test={trial_split[2]}")
    print(f"  behaviors ({D}): {', '.join(behaviors)}")


if __name__ == "__main__":
    main()
