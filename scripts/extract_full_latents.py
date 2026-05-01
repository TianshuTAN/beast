"""Extract full 197-token ViT-MAE encoder output + ids_restore for one camera.

Used by the video-decoding pipeline (`scripts/decode_video.py`). Unlike
`extract_latents.py`, this saves the FULL encoder output (B, 197, 768) plus the
permutation indices the decoder needs (`ids_restore`, shape (B, 196)).

Output is HDF5 with one dataset per (split × field) pair — chunked + LZF-
compressed to keep ~14 GB raw down to ~5-7 GB on disk.

Schema:
    latents_<split>     : (K_split, T, 197, 768) float32
    ids_restore_<split> : (K_split, T, 196)      int64
    trial_split         : (3,) int64    # [n_train, n_val, n_test]
    n_timebins          : ()  int64
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py
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
            f"({n_intervals}x{n_timebins}), got {len(triples)}"
        )
    for k, (iv, tb, _p) in enumerate(triples):
        if iv * n_timebins + tb != k:
            raise RuntimeError(f"frame ordering mismatch at {split_dir}/{_p.name}")
    return [p for _, _, p in triples], n_intervals, n_timebins


def _load_model(ckpt_dir: Path, device: torch.device) -> VisionTransformer:
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
def _run_split(
    model: VisionTransformer,
    paths: list[Path],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (latents (N, 197, 768) float32, ids_restore (N, 196) int64)."""
    loader = torch.utils.data.DataLoader(
        _FrameDataset(paths),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )
    lat_chunks: list[np.ndarray] = []
    ids_chunks: list[np.ndarray] = []
    n_done = 0
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        # call ViT directly so we can also grab ids_restore
        outputs = model.vit_mae.vit(pixel_values=batch)
        # last_hidden_state: (B, 197, 768) — CLS at position 0, then 196 patches in shuffled order
        # ids_restore:      (B, 196)      — inverse permutation to put patches back to spatial order
        lat_chunks.append(outputs.last_hidden_state.cpu().numpy().astype(np.float32))
        ids_chunks.append(outputs.ids_restore.cpu().numpy().astype(np.int64))
        n_done += batch.size(0)
        if n_done % (batch_size * 20) == 0:
            print(f"    {n_done} / {len(paths)} frames")
    return np.concatenate(lat_chunks, axis=0), np.concatenate(ids_chunks, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eid", required=True)
    ap.add_argument("--eval_frames_dir", required=True,
                    help="Parent of _iblrig_<cam>Camera.raw.<eid>/ split dirs.")
    ap.add_argument("--cam", choices=["left", "right"], required=True)
    ap.add_argument("--ckpt_dir", required=True,
                    help="tb_logs/version_X dir for the chosen cam's stage-1 ckpt.")
    ap.add_argument("--output_h5", required=True,
                    help="Output HDF5 path (will be overwritten).")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}, eid={args.eid}, cam={args.cam}")

    cam_dir = Path(args.eval_frames_dir) / f"_iblrig_{args.cam}Camera.raw.{args.eid}"
    if not cam_dir.is_dir():
        raise FileNotFoundError(cam_dir)

    model = _load_model(Path(args.ckpt_dir), device)

    out_path = Path(args.output_h5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    splits = ("train", "val", "test")
    interval_counts: list[int] = []
    n_timebins: int | None = None

    with h5py.File(out_path, "w") as h5:
        for split in splits:
            split_dir = cam_dir / split
            paths, n_intervals, T = _list_frames_sorted(split_dir)
            if n_timebins is None:
                n_timebins = T
            elif n_timebins != T:
                raise RuntimeError(f"timebin count mismatch: {n_timebins} vs {T}")
            print(f"  {split}: {len(paths)} frames ({n_intervals} intervals x {T} timebins)")
            lat, ids = _run_split(model, paths, args.batch_size, args.num_workers, device)
            assert lat.shape == (len(paths), 197, 768), f"latent shape {lat.shape}"
            assert ids.shape == (len(paths), 196), f"ids_restore shape {ids.shape}"
            lat_4d = lat.reshape(n_intervals, T, 197, 768)
            ids_3d = ids.reshape(n_intervals, T, 196)
            # chunk one trial at a time so PCA bulk reads stream linearly
            h5.create_dataset(
                f"latents_{split}", data=lat_4d,
                chunks=(1, T, 197, 768),
                compression="lzf",
            )
            h5.create_dataset(
                f"ids_restore_{split}", data=ids_3d,
                chunks=(1, T, 196),
                compression="lzf",
            )
            interval_counts.append(n_intervals)
            print(f"    wrote latents_{split} {lat_4d.shape}, ids_restore_{split} {ids_3d.shape}")
        h5.create_dataset("trial_split", data=np.array(interval_counts, dtype=np.int64))
        h5.create_dataset("n_timebins", data=np.int64(n_timebins))

    size_gb = out_path.stat().st_size / 1024**3
    print(f"\nWrote {out_path} ({size_gb:.2f} GB)")


if __name__ == "__main__":
    main()
