"""Decode video frames from neural data via PCA-3 of full ViT-MAE patch tokens.

Per mentor's spec (2026-04-28):
  1. Load video latents Z (K, T, 197, 768) and ids_restore (K, T, 196).
  2. Compute frame avg Z_avg (1, 1, 197, 768) and std Z_std on TRAIN split.
  3. Z_norm = (Z - Z_avg) / Z_std.
  4. PCA(n_components=3) on Z_norm[train].reshape(-1, 768) (pooled over K, T, 197).
     Save mean and components for inverse transform.
  5. Use neural data X (K, T, C) to predict Z_compressed (K, T, 197, 3) via
     train_cnn_decoder_with_tune (KeypointsNetwork + Ray Tune).
  6. Unproject: Z_pred = (Z_compressed_pred @ components + mean) * Z_std + Z_avg.
  7. Run model.vit_mae.decoder(Z_pred, ids_restore) + unpatchify -> images.
  8. Per-frame PSNR between ViT-decoded GT and ViT-decoded pred (both go through
     decoder so PSNR measures TCN-prediction loss, not encoder/decoder loss).
  9. Save 5 random + 5 best test trials' GT/pred PNGs for visualization.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "encoding_decoding_code"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import h5py
import numpy as np
import torch
import yaml
from PIL import Image
from sklearn.decomposition import PCA

from analyses.utils.decoder import train_cnn_decoder_with_tune
from beast.models.vits import VisionTransformer

# inverse ImageNet normalization for PSNR + image saving
_IN_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
_IN_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def _load_decoder_model(ckpt_dir: Path, device: torch.device) -> VisionTransformer:
    cfg_candidates = [ckpt_dir / "config.yaml", ckpt_dir.parent.parent / "config.yaml"]
    cfg_path = next((p for p in cfg_candidates if p.is_file()), None)
    if cfg_path is None:
        raise FileNotFoundError(f"config.yaml not found near {ckpt_dir}")
    ckpt_path = ckpt_dir / "checkpoints" / "last.ckpt"
    with cfg_path.open() as f:
        config = yaml.safe_load(f)
    config["model"]["model_params"]["random_init"] = True
    model = VisionTransformer(config)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"], strict=False)
    model.to(device).eval()
    model.vit_mae.config.mask_ratio = 0
    return model


@torch.no_grad()
def _decode_to_images(
    model: VisionTransformer,
    latents: np.ndarray,        # (K, T, 197, 768) float32
    ids_restore: np.ndarray,    # (K, T, 196) int64
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Run vit.decoder + unpatchify over all (K*T) frames. Returns (K, T, 3, 224, 224) float32."""
    K, T, L, D = latents.shape
    flat_lat = latents.reshape(K * T, L, D)
    flat_ids = ids_restore.reshape(K * T, -1)
    out_chunks: list[np.ndarray] = []
    for i in range(0, flat_lat.shape[0], batch_size):
        lat_b = torch.from_numpy(flat_lat[i:i + batch_size]).to(device, dtype=torch.float32)
        ids_b = torch.from_numpy(flat_ids[i:i + batch_size]).to(device, dtype=torch.long)
        decoder_outputs = model.vit_mae.decoder(lat_b, ids_b)
        logits = decoder_outputs.logits
        imgs = model.vit_mae.unpatchify(logits)  # (B, 3, 224, 224)
        out_chunks.append(imgs.cpu().numpy().astype(np.float32))
    flat_imgs = np.concatenate(out_chunks, axis=0)  # (K*T, 3, 224, 224)
    return flat_imgs.reshape(K, T, 3, 224, 224)


def _to_unit_range(imgs: np.ndarray) -> np.ndarray:
    """Inverse ImageNet normalization, then clip to [0, 1]. Input shape (..., 3, H, W)."""
    out = imgs * _IN_STD + _IN_MEAN  # broadcast over leading dims
    return np.clip(out, 0.0, 1.0)


def _psnr(gt: np.ndarray, pred: np.ndarray, axis_per_frame: tuple[int, ...]) -> np.ndarray:
    """PSNR per frame between gt and pred, both in [0, 1]. Reduce MSE along axis_per_frame."""
    mse = ((gt - pred) ** 2).mean(axis=axis_per_frame)
    eps = 1e-12
    psnr = 10.0 * np.log10(1.0 / np.maximum(mse, eps))
    return psnr


def _save_pngs(imgs: np.ndarray, out_dir: Path, prefix: str) -> None:
    """imgs shape (T, 3, H, W) in [0, 1]. Saves <prefix>_timebinM.png."""
    out_dir.mkdir(parents=True, exist_ok=True)
    T = imgs.shape[0]
    for t in range(T):
        # (3, H, W) -> (H, W, 3) uint8
        arr = (imgs[t].transpose(1, 2, 0) * 255.0).round().clip(0, 255).astype(np.uint8)
        Image.fromarray(arr).save(out_dir / f"{prefix}_timebin{t:02d}.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eid", required=True)
    ap.add_argument("--latents_h5", required=True)
    ap.add_argument("--neural_npz", required=True,
                    help="Path to <eid>_aligned.npz with train/val/test_spikes.")
    ap.add_argument("--ckpt_dir", required=True,
                    help="tb_logs/version_X dir for the encoder ckpt (decoder weights loaded from it).")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--pca_components", type=int, default=3)
    ap.add_argument("--num_samples", type=int, default=20,
                    help="Ray Tune samples for TCN decoder.")
    ap.add_argument("--decode_batch_size", type=int, default=32)
    ap.add_argument("--center_only", action="store_true", default=False,
                    help="If set, only subtract train mean (no /std). "
                         "Per co-authors 2026-04-30: mean-only latents are more decodable.")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}, eid={args.eid}, output={out}")

    # ── Phase 1: load latents ────────────────────────────────────────────────
    print("\n[1] Loading latents from", args.latents_h5)
    with h5py.File(args.latents_h5, "r") as h5:
        Z_train = h5["latents_train"][:]
        Z_val = h5["latents_val"][:]
        Z_test = h5["latents_test"][:]
        ids_test = h5["ids_restore_test"][:]
        trial_split = h5["trial_split"][:]
        T = int(h5["n_timebins"][()])
    n_train, n_val, n_test = [int(x) for x in trial_split]
    L, D = Z_train.shape[2], Z_train.shape[3]
    assert (L, D) == (197, 768), f"unexpected token shape {(L, D)}"
    print(f"  Z_train {Z_train.shape}, Z_val {Z_val.shape}, Z_test {Z_test.shape}")
    print(f"  ids_test {ids_test.shape}, T={T}, splits={trial_split.tolist()}")

    # ── Phase 2: frame avg (and optionally std) on train ─────────────────────
    print(f"\n[2] Computing Z_avg{' only' if args.center_only else ', Z_std'} on train split (per-token-position)")
    Z_avg = Z_train.mean(axis=(0, 1), keepdims=True).astype(np.float32)  # (1, 1, 197, 768)
    if args.center_only:
        Z_std = None
        print(f"  Z_avg {Z_avg.shape} (mean-only mode)")
    else:
        Z_std = Z_train.std(axis=(0, 1), keepdims=True).astype(np.float32)
        Z_std = np.maximum(Z_std, 1e-6)
        print(f"  Z_avg {Z_avg.shape}, Z_std {Z_std.shape}")

    # ── Phase 3: PCA along D=768 (pooled over K, T, 197 in train) ───────────
    print(f"\n[3] PCA(n={args.pca_components}) along feature dim 768"
          f" ({'mean-only' if args.center_only else 'mean+std'} normalization)")
    if args.center_only:
        Zn_train = Z_train - Z_avg
        Zn_val = Z_val - Z_avg
        Zn_test = Z_test - Z_avg
    else:
        Zn_train = (Z_train - Z_avg) / Z_std
        Zn_val = (Z_val - Z_avg) / Z_std
        Zn_test = (Z_test - Z_avg) / Z_std
    flat_train = Zn_train.reshape(-1, D)  # (n_train * T * 197, 768)
    print(f"  fitting PCA on {flat_train.shape[0]:,} rows × {D}")
    pca = PCA(n_components=args.pca_components, svd_solver="randomized", random_state=0)
    pca.fit(flat_train)
    evr = pca.explained_variance_ratio_
    print(f"  explained variance: {evr.tolist()}, cumulative={evr.sum():.4f}")

    def _project(Zn: np.ndarray, K: int) -> np.ndarray:
        return pca.transform(Zn.reshape(-1, D)).reshape(K, T, L, args.pca_components).astype(np.float32)

    Zc_train = _project(Zn_train, n_train)
    Zc_val = _project(Zn_val, n_val)
    Zc_test = _project(Zn_test, n_test)
    print(f"  Z_compressed shapes: train {Zc_train.shape}, val {Zc_val.shape}, test {Zc_test.shape}")

    np.savez(
        out / "pca.npz",
        Z_avg=Z_avg,
        Z_std=(np.zeros(0, dtype=np.float32) if Z_std is None else Z_std),
        pca_mean=pca.mean_.astype(np.float32),
        pca_components=pca.components_.astype(np.float32),
        explained_variance_ratio=evr.astype(np.float32),
        n_components=args.pca_components,
        center_only=np.bool_(args.center_only),
    )
    np.savez(
        out / "z_compressed.npz",
        train=Zc_train, val=Zc_val, test=Zc_test,
        trial_split=trial_split,
    )
    # free Zn intermediates
    del flat_train, Zn_train, Zn_val, Zn_test

    # ── Phase 4: TCN decoder via Ray Tune ────────────────────────────────────
    print("\n[4] TCN decoder X→Z_compressed (Ray Tune)")
    print(f"  loading neural data from {args.neural_npz}")
    neural = np.load(args.neural_npz, allow_pickle=True)
    train_X = neural["train_spikes"].astype(np.float32)
    val_X = neural["val_spikes"].astype(np.float32)
    test_X = neural["test_spikes"].astype(np.float32)
    print(f"  X shapes: train {train_X.shape}, val {val_X.shape}, test {test_X.shape}")
    embed_size = L * args.pca_components  # 197 * 3 = 591
    data_dict = {args.eid: {
        "X": [train_X, val_X, test_X],
        "y": [
            Zc_train.reshape(n_train, T, embed_size),
            Zc_val.reshape(n_val, T, embed_size),
            Zc_test.reshape(n_test, T, embed_size),
        ],
        "setup": {},
    }}
    res = train_cnn_decoder_with_tune(data_dict, num_samples=args.num_samples)
    pred_flat = res[args.eid]["pred"]  # (n_test, T, embed_size) — denormalized to Z_compressed scale
    Zc_pred = pred_flat.reshape(n_test, T, L, args.pca_components).astype(np.float32)
    r2 = res[args.eid]["r2"]
    print(f"  TCN test R²={r2:.4f}, Z_compressed_pred shape {Zc_pred.shape}")

    # save
    np.save(out / "decoder_results.npy", res)
    np.savez(out / "z_compressed_pred.npz", pred=Zc_pred)

    # ── Phase 5: unproject Z_compressed_pred → Z_pred (197×768) ─────────────
    print("\n[5] Unprojecting Z_compressed_pred → Z_pred (197 × 768)")
    flat_pred = Zc_pred.reshape(-1, args.pca_components)  # (n_test*T*197, K)
    Zn_pred_flat = flat_pred @ pca.components_ + pca.mean_  # (n_test*T*197, 768)
    Zn_pred = Zn_pred_flat.reshape(n_test, T, L, D).astype(np.float32)
    if args.center_only:
        Z_pred = Zn_pred + Z_avg
    else:
        Z_pred = Zn_pred * Z_std + Z_avg  # back to encoder-output scale
    print(f"  Z_pred {Z_pred.shape}, dtype {Z_pred.dtype}")

    # ── Phase 6: ViT-MAE decoder + unpatchify (GT and pred) ─────────────────
    print("\n[6] ViT-MAE decoder + unpatchify (test set, GT and pred)")
    model = _load_decoder_model(Path(args.ckpt_dir), device)
    print(f"  decoding {n_test} trials × {T} timebins = {n_test * T} frames")
    print("  GT pass...")
    gt_imgs = _decode_to_images(model, Z_test, ids_test, device, args.decode_batch_size)
    print(f"    gt_imgs {gt_imgs.shape}")
    print("  pred pass...")
    pred_imgs = _decode_to_images(model, Z_pred, ids_test, device, args.decode_batch_size)
    print(f"    pred_imgs {pred_imgs.shape}")

    # ── Phase 7: PSNR per frame ─────────────────────────────────────────────
    print("\n[7] PSNR per frame")
    gt_unit = _to_unit_range(gt_imgs)
    pred_unit = _to_unit_range(pred_imgs)
    psnr_per_frame = _psnr(gt_unit, pred_unit, axis_per_frame=(2, 3, 4))  # over (3, H, W)
    psnr_mean = float(psnr_per_frame.mean())
    psnr_std = float(psnr_per_frame.std())
    print(f"  PSNR mean={psnr_mean:.3f} dB, std={psnr_std:.3f} dB over {psnr_per_frame.size} frames")
    np.savez(
        out / "psnr.npz",
        per_frame=psnr_per_frame.astype(np.float32),
        mean=psnr_mean, std=psnr_std,
        eid=args.eid,
    )

    # ── Phase 8: save 5 random + 5 best test trials ─────────────────────────
    print("\n[8] Saving visualization PNGs (5 random + 5 best test trials)")
    rng = np.random.default_rng(42)
    rand_idx = sorted(rng.choice(n_test, size=5, replace=False).tolist())
    per_trial_psnr = psnr_per_frame.mean(axis=1)  # (n_test,)
    best_idx = sorted(np.argsort(per_trial_psnr)[-5:].tolist())
    print(f"  random trials: {rand_idx}")
    print(f"  best trials  : {best_idx} (PSNR {[round(float(per_trial_psnr[i]), 2) for i in best_idx]})")
    vis_root = out / "vis_frames"
    for label, idxs in [("random", rand_idx), ("best", best_idx)]:
        for i in idxs:
            tdir = vis_root / f"{label}_trial{i:03d}"
            _save_pngs(gt_unit[i], tdir, "gt")
            _save_pngs(pred_unit[i], tdir, "pred")
    overlap = sorted(set(rand_idx) & set(best_idx))
    print(f"  saved {(len(rand_idx)+len(best_idx)) * T * 2} PNGs (random ∩ best = {overlap})")

    # ── done ────────────────────────────────────────────────────────────────
    summary = {
        "eid": args.eid,
        "n_test": n_test,
        "T": T,
        "pca_components": args.pca_components,
        "center_only": bool(args.center_only),
        "pca_explained_variance": evr.tolist(),
        "tcn_test_r2": float(r2),
        "psnr_mean": psnr_mean,
        "psnr_std": psnr_std,
        "vis_random_trials": rand_idx,
        "vis_best_trials": best_idx,
    }
    with (out / "summary.yaml").open("w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    print(f"\nALL DONE. Summary at {out / 'summary.yaml'}")


if __name__ == "__main__":
    main()
