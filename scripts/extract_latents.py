"""Extract BEAST-AE bottleneck latents over the eval frames for one session.

Loads a trained left-camera and right-camera checkpoint, runs each through the
matching `extracted_frames/eval/.../{train,val,test}/intervalNtimebinM.png`
directories, and assembles a `z_trials.npz` compatible with
`encoding_decoding_code/test.py`.

Output schema:
    z_trials_time : (K, T, V, D) float32      # K=400, T=60, V=2, D=num_latents
    trial_split   : (3,) int64                # [n_train, n_val, n_test]

Trial order in z_trials_time is `[train, val, test]` to match the slicing
done by `test.py`, which uses `len(neural_data['*_intervals'])` rather than
`trial_split`.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

from beast.models.vits import VisionTransformer

_FRAME_RE = re.compile(r"interval(\d+)timebin(\d+)\.png$")
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_IMG_TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224), antialias=True),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


class _FrameDataset(torch.utils.data.Dataset):
    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return _IMG_TF(Image.open(self.paths[idx]).convert("RGB"))


def _list_frames_sorted(split_dir: Path) -> tuple[list[Path], int, int]:
    """Return frame paths sorted by (interval_idx, timebin_idx)."""
    triples: list[tuple[int, int, Path]] = []
    for p in split_dir.glob("*.png"):
        m = _FRAME_RE.match(p.name)
        if not m:
            continue
        triples.append((int(m.group(1)), int(m.group(2)), p))
    if not triples:
        raise FileNotFoundError(f"No interval*timebin*.png found in {split_dir}")
    triples.sort()
    n_intervals = max(t[0] for t in triples) + 1
    n_timebins = max(t[1] for t in triples) + 1
    expected = n_intervals * n_timebins
    if len(triples) != expected:
        raise RuntimeError(
            f"{split_dir}: expected {expected} frames "
            f"({n_intervals} intervals × {n_timebins} timebins), got {len(triples)}"
        )
    # confirm dense (interval i has all timebins 0..n_timebins-1)
    for k, (iv, tb, _p) in enumerate(triples):
        if iv * n_timebins + tb != k:
            raise RuntimeError(f"frame ordering mismatch at {split_dir}/{_p.name}")
    return [p for _, _, p in triples], n_intervals, n_timebins


def _load_model(ckpt_dir: Path, device: torch.device) -> VisionTransformer:
    # config.yaml is saved at the training output root, not inside tb_logs/version_X
    cfg_candidates = [ckpt_dir / "config.yaml", ckpt_dir.parent.parent / "config.yaml"]
    cfg_path = next((p for p in cfg_candidates if p.is_file()), None)
    if cfg_path is None:
        raise FileNotFoundError(
            f"config.yaml not found at any of: {[str(p) for p in cfg_candidates]}"
        )
    ckpt_path = ckpt_dir / "checkpoints" / "last.ckpt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    with cfg_path.open() as f:
        config = yaml.safe_load(f)

    # skip HuggingFace download — our state_dict will overwrite the weights anyway
    config["model"]["model_params"]["random_init"] = True

    model = VisionTransformer(config)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state["state_dict"], strict=False)
    if missing:
        print(f"  warn: missing keys: {len(missing)} (first 3: {missing[:3]})")
    if unexpected:
        print(f"  warn: unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})")

    model.to(device).eval()
    model.vit_mae.config.mask_ratio = 0
    return model


@torch.no_grad()
def _run_inference(
    model: VisionTransformer,
    paths: list[Path],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    mode: str = "bottleneck",
) -> np.ndarray:
    loader = torch.utils.data.DataLoader(
        _FrameDataset(paths),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )
    chunks: list[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        out = model(batch)
        if mode == "bottleneck":
            chunks.append(out["bottleneck_latents"].cpu().numpy())
        elif mode == "cls":
            # out["latents"] shape (B, T, H); CLS token is at position 0
            chunks.append(out["latents"][:, 0].cpu().numpy())
        else:
            raise ValueError(f"unknown mode: {mode}")
    return np.concatenate(chunks, axis=0).astype(np.float32)


def _process_camera(
    cam_dir: Path,
    ckpt_dir: Path,
    splits: tuple[str, str, str],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    mode: str = "bottleneck",
) -> tuple[np.ndarray, list[int]]:
    """Return (latents_concat (K_total, D), [n_train, n_val, n_test] interval counts)."""
    print(f"Loading model from {ckpt_dir}")
    model = _load_model(ckpt_dir, device)

    per_split_arrays: list[np.ndarray] = []
    interval_counts: list[int] = []
    for split in splits:
        split_dir = cam_dir / split
        paths, n_intervals, n_timebins = _list_frames_sorted(split_dir)
        print(f"  {split}: {len(paths)} frames ({n_intervals} intervals × {n_timebins} timebins)")
        flat = _run_inference(model, paths, batch_size, num_workers, device, mode=mode)
        D = flat.shape[1]
        reshaped = flat.reshape(n_intervals, n_timebins, D)  # (K_split, T, D)
        per_split_arrays.append(reshaped)
        interval_counts.append(n_intervals)

    # free GPU memory
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    cam_latents = np.concatenate(per_split_arrays, axis=0)  # (K_total, T, D)
    return cam_latents, interval_counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eid", required=True)
    ap.add_argument("--eval_frames_dir", required=True,
                    help="Parent of _iblrig_{left,right}Camera.raw.{eid}/ split dirs.")
    ap.add_argument("--left_ckpt_dir", required=True,
                    help="tb_logs/version_X dir for left model.")
    ap.add_argument("--right_ckpt_dir", required=True,
                    help="tb_logs/version_X dir for right model.")
    ap.add_argument("--output_dir", required=True,
                    help="Will write <output_dir>/<eid>/z_trials.npz")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--mode", choices=["bottleneck", "cls"], default="bottleneck",
                    help="bottleneck: extract BEAST-AE bottleneck_latents (needs num_latents set); "
                         "cls: extract CLS token from full encoder output (works for any ViT-MAE)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    eval_root = Path(args.eval_frames_dir)
    left_dir = eval_root / f"_iblrig_leftCamera.raw.{args.eid}"
    right_dir = eval_root / f"_iblrig_rightCamera.raw.{args.eid}"
    if not left_dir.is_dir() or not right_dir.is_dir():
        raise FileNotFoundError(f"missing camera dir under {eval_root}")

    splits = ("train", "val", "test")

    print(f"=== LEFT CAMERA (mode={args.mode}) ===")
    left_lat, left_counts = _process_camera(
        left_dir, Path(args.left_ckpt_dir), splits,
        args.batch_size, args.num_workers, device, mode=args.mode,
    )
    print(f"  left latents shape: {left_lat.shape}")

    print(f"=== RIGHT CAMERA (mode={args.mode}) ===")
    right_lat, right_counts = _process_camera(
        right_dir, Path(args.right_ckpt_dir), splits,
        args.batch_size, args.num_workers, device, mode=args.mode,
    )
    print(f"  right latents shape: {right_lat.shape}")

    if left_counts != right_counts:
        raise RuntimeError(
            f"Interval counts mismatch: left={left_counts} right={right_counts}"
        )

    # stack along view axis: (K, T, D, V) → transpose to (K, T, V, D)
    z_trials_time = np.stack([left_lat, right_lat], axis=2)  # (K, T, 2, D)
    trial_split = np.array(left_counts, dtype=np.int64)

    out_dir = Path(args.output_dir) / args.eid
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "z_trials.npz"
    np.savez(out_path, z_trials_time=z_trials_time, trial_split=trial_split)
    print(f"Saved {out_path}")
    print(f"  z_trials_time shape: {z_trials_time.shape} dtype={z_trials_time.dtype}")
    print(f"  trial_split: train={trial_split[0]} val={trial_split[1]} test={trial_split[2]}")


if __name__ == "__main__":
    main()
