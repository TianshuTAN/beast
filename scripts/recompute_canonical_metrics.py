"""Recompute canonical PSNR + SSIM on existing decode_video.py runs.

For each (eid, cam) output dir under beast_data/decoding_video/<eid>/pca<K>_unshuffled_<cam>/:
  - Load test latents + ids from <eid>/<cam>_full_unshuffled.h5
  - Load Z_avg / Z_std / pca_mean / pca_components from pca.npz
  - Load Zc_pred from z_compressed_pred.npz
  - Reconstruct Z_pred via PCA inverse + denorm
  - Run ViT-MAE decoder for GT (from latents_test) and pred (from Z_pred)
  - Resize to 320×320 (PIL BICUBIC), compute canonical PSNR + SSIM
    (data_range=1.0), save <out_dir>/psnr_ssim_metrics.npz
  - Patch summary.yaml in place: append ssim_mean, ssim_std, replace psnr_mean/std

Usage:
  python scripts/recompute_canonical_metrics.py \
    --runs <path1> [<path2> ...] \
    --eval_frames_dir <unused, kept for symmetry>
  Or:
  python scripts/recompute_canonical_metrics.py --auto_5sess  # iterate all 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "encoding_decoding_code"
for _p in (SRC_ROOT, SCRIPTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import h5py
import numpy as np
import torch
import yaml
from PIL import Image

from beast.models.vits import VisionTransformer
from decoding_metrics import (
    _psnr_per_image,
    _ssim_per_image,
    save_psnr_ssim_metrics_npz,
)
# reuse helpers from decode_video.py via import
from decode_video import _to_unit_range, _resize_to_320, _batched_metric, _decode_to_images, _load_decoder_model

EIDS = [
    ("s1", "4b00df29-3769-43be-bb40-128b1cba6d35"),
    ("s2", "72cb5550-43b4-4ef0-add5-e4adfdfb5e02"),
    ("s3", "781b35fd-e1f0-4d14-b2bb-95b7263082bb"),
    ("s4", "ecb5520d-1358-434c-95ec-93687ecd1396"),
    ("s5", "f312aaec-3b6f-44b3-86b4-3a0c119c0438"),
]
PCA_K = 6
DEC_BASE = Path("/home/haoshen/Disk5/Encoding_Decoding/3DProject/beast_data/decoding_video")
CKPT_BASE = Path("/home/haoshen/Disk5/Encoding_Decoding/3DProject/beast_data/checkpoints")
NEURAL_BASE = Path("/home/haoshen/Disk5/Encoding_Decoding/3DProject/beast_data/neural_data_400_yizi")


def process_run(eid: str, cam: str, device: torch.device) -> dict | None:
    out_dir = DEC_BASE / eid / f"pca{PCA_K}_unshuffled_{cam}"
    if cam == "left" and not out_dir.exists():
        # fallback to legacy s1 left dir name (no _left suffix)
        legacy = DEC_BASE / eid / f"pca{PCA_K}_unshuffled"
        if legacy.exists():
            out_dir = legacy
    h5_path = DEC_BASE / eid / f"{cam}_full_unshuffled.h5"
    pca_path = out_dir / "pca.npz"
    zcp_path = out_dir / "z_compressed_pred.npz"
    ckpt_dir = CKPT_BASE / f"beast_{cam}_noaug_{eid}" / "tb_logs" / "version_0"
    neural_npz = NEURAL_BASE / eid / f"{eid}_aligned.npz"

    for p in (h5_path, pca_path, zcp_path, neural_npz):
        if not p.exists():
            print(f"  [skip] missing {p}")
            return None
    if not (ckpt_dir / "checkpoints").exists():
        print(f"  [skip] missing ckpt at {ckpt_dir}")
        return None

    print(f"\n=== {cam} {eid[:8]} ===")
    print(f"  out_dir: {out_dir}")

    # ── load saved artifacts ────────────────────────────────────────────────
    with h5py.File(h5_path, "r") as h5:
        Z_test = h5["latents_test"][:]            # (K, T, 197, 768) — already unshuffled
        ids_test = h5["ids_restore_test"][:]      # (K, T, 196) — identity
        T = int(h5["n_timebins"][()])
    n_test = Z_test.shape[0]
    L, D = Z_test.shape[2], Z_test.shape[3]
    print(f"  Z_test {Z_test.shape}, T={T}")

    pca = np.load(pca_path)
    Z_avg = pca["Z_avg"]
    Z_std = pca["Z_std"] if pca["Z_std"].size > 0 else None
    pca_mean = pca["pca_mean"]
    pca_components = pca["pca_components"]
    K_pca = int(pca["n_components"])
    center_only = bool(pca["center_only"])
    print(f"  PCA k={K_pca}, center_only={center_only}, Z_std={'None' if Z_std is None else Z_std.shape}")

    Zc_pred = np.load(zcp_path)["pred"]           # (K, T, 197, K_pca)
    flat = Zc_pred.reshape(-1, K_pca)
    Zn_pred = (flat @ pca_components + pca_mean).reshape(n_test, T, L, D).astype(np.float32)
    if center_only:
        Z_pred = Zn_pred + Z_avg
    else:
        Z_pred = Zn_pred * Z_std + Z_avg

    # ── decode GT + pred ────────────────────────────────────────────────────
    model = _load_decoder_model(ckpt_dir, device)
    print("  GT pass...")
    gt_imgs = _decode_to_images(model, Z_test, ids_test, device, batch_size=32)
    print(f"    gt_imgs {gt_imgs.shape}")
    print("  pred pass...")
    pred_imgs = _decode_to_images(model, Z_pred, ids_test, device, batch_size=32)
    print(f"    pred_imgs {pred_imgs.shape}")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── canonical metrics ──────────────────────────────────────────────────
    gt_unit = _to_unit_range(gt_imgs)
    pred_unit = _to_unit_range(pred_imgs)
    print("  resize 224 → 320 (PIL BICUBIC)...")
    gt_320 = _resize_to_320(gt_unit)
    pred_320 = _resize_to_320(pred_unit)

    gt_t = torch.from_numpy(gt_320).reshape(n_test * T, 1, 3, 320, 320)
    pred_t = torch.from_numpy(pred_320).reshape(n_test * T, 1, 3, 320, 320).clamp_(0, 1)
    if device.type == "cuda":
        gt_t = gt_t.to(device);  pred_t = pred_t.to(device)
    psnr_flat = _batched_metric(_psnr_per_image, pred_t, gt_t, batch_size=32)
    ssim_flat = _batched_metric(_ssim_per_image, pred_t, gt_t, batch_size=32)
    psnr_KTV = psnr_flat.numpy().reshape(n_test, T, 1).astype(np.float32)
    ssim_KTV = ssim_flat.numpy().reshape(n_test, T, 1).astype(np.float32)
    psnr_per_frame = psnr_KTV[..., 0]
    ssim_per_frame = ssim_KTV[..., 0]
    psnr_mean = float(np.nanmean(psnr_per_frame))
    psnr_std  = float(np.nanstd(psnr_per_frame))
    ssim_mean = float(np.nanmean(ssim_per_frame))
    ssim_std  = float(np.nanstd(ssim_per_frame))
    print(f"  PSNR canon mean={psnr_mean:.3f} dB std={psnr_std:.3f}  SSIM mean={ssim_mean:.4f} std={ssim_std:.4f}")

    save_psnr_ssim_metrics_npz(
        out_dir / "psnr_ssim_metrics.npz",
        psnr_blocks=[psnr_KTV],
        ssim_blocks=[ssim_KTV],
        neural_trial_blocks=[np.arange(n_test, dtype=np.int64)],
        neural_bin_blocks=[np.tile(np.arange(T, dtype=np.int64), (n_test, 1))],
        trial_split_blocks=[np.full((n_test,), "test", dtype=str)],
        source_file_rows=[str(neural_npz)] * n_test,
        view_names=(cam,),
    )
    print(f"  wrote {out_dir / 'psnr_ssim_metrics.npz'}")

    # patch summary.yaml: add canonical metrics, keep old-PSNR fields renamed
    sm_path = out_dir / "summary.yaml"
    sm = yaml.safe_load(sm_path.open())
    sm["psnr_mean_legacy_224"] = sm.get("psnr_mean")
    sm["psnr_std_legacy_224"] = sm.get("psnr_std")
    sm["psnr_mean"] = psnr_mean
    sm["psnr_std"] = psnr_std
    sm["ssim_mean"] = ssim_mean
    sm["ssim_std"] = ssim_std
    sm["metric_resize_to"] = 320
    sm["metric_resize_method"] = "PIL.Image.Resampling.BICUBIC"
    sm["metric_data_range"] = 1.0
    yaml.safe_dump(sm, sm_path.open("w"), sort_keys=False)
    print(f"  patched {sm_path}")

    return {
        "tag_eid": (eid, cam),
        "psnr_mean": psnr_mean, "psnr_std": psnr_std,
        "ssim_mean": ssim_mean, "ssim_std": ssim_std,
        "psnr_legacy": sm["psnr_mean_legacy_224"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto_5sess", action="store_true",
                    help="Iterate all 5 sessions × {left, right}")
    ap.add_argument("--eid", type=str, default=None)
    ap.add_argument("--cam", choices=["left", "right"], default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    pairs: list[tuple[str, str]] = []
    if args.auto_5sess:
        for tag, eid in EIDS:
            for cam in ("left", "right"):
                pairs.append((eid, cam))
    elif args.eid and args.cam:
        pairs.append((args.eid, args.cam))
    else:
        ap.error("provide --auto_5sess OR (--eid AND --cam)")

    summary = []
    for eid, cam in pairs:
        try:
            r = process_run(eid, cam, device)
            if r is not None:
                summary.append(r)
        except Exception as e:
            import traceback
            print(f"  [FAIL] {eid[:8]} {cam}: {type(e).__name__}: {e}")
            traceback.print_exc()

    print("\n=== Backfill summary ===")
    print(f"{'sess/eid':<14} {'cam':<6} {'PSNR canon':>11} {'SSIM canon':>11} {'PSNR legacy':>12}")
    print("-" * 60)
    tag_lookup = {eid: tag for tag, eid in EIDS}
    for r in summary:
        eid, cam = r["tag_eid"]
        tag = tag_lookup.get(eid, eid[:8])
        print(f"{tag:<14} {cam:<6} {r['psnr_mean']:>8.3f} dB "
              f"{r['ssim_mean']:>9.4f}    {r['psnr_legacy']:>9.3f} dB")


if __name__ == "__main__":
    main()
