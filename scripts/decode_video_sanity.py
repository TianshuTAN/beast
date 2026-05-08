"""Sanity checks for the video-decoding pipeline.

Runs two independent checks the mentor requested (2026-05-02):

  Check 1 — NO COMPRESSION:
     Take encoder output Z_test (197, 768) directly through the ViT-MAE
     decoder + unpatchify. If the model loaded correctly and the
     latent-extraction pipeline isn't garbled, the decoded image should look
     like the input frame (PSNR easily >20 dB for an MAE pretrained on
     natural images).

  Check 2 — PCA ROUND-TRIP (no TCN):
     For k in {3, 8, 50}, do PCA forward + inverse on Z_test (along D=768,
     same recipe as decode_video.py), then decode. Compares the decoded
     round-tripped image against decoded raw Z_test. Tells us how much
     information PCA itself is throwing away — independent of TCN.

Both checks run on a small set of test frames (default: 5 random + 5 best by
GT-vs-input PSNR if input frames are available). Outputs side-by-side PNGs
and prints PSNR tables.

Usage:
  python scripts/decode_video_sanity.py \
    --eid 4b00df29-3769-43be-bb40-128b1cba6d35 \
    --cam left \
    --latents_h5 beast_data/decoding_video/4b00df29-.../left_full.h5 \
    --ckpt_dir   beast_data/checkpoints/beast_left_noaug_4b00df29-.../tb_logs/version_0 \
    --eval_frames_dir beast_data/extracted_frames_400_yizi/eval \
    --output_dir beast_data/decoding_video/4b00df29-.../sanity_check_2026-05-02 \
    --pca_ks 3 8 50 \
    --num_trials 5
"""

from __future__ import annotations

import argparse
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
from torchvision import transforms

from beast.models.vits import VisionTransformer

_IN_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
_IN_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_IMG_TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224), antialias=True),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


def _load_model(ckpt_dir: Path, device: torch.device) -> VisionTransformer:
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

    missing, unexpected = model.load_state_dict(state["state_dict"], strict=False)
    print(f"  state_dict load: missing={len(missing)}  unexpected={len(unexpected)}")
    if missing:
        print(f"    first 3 missing: {missing[:3]}")
    if unexpected:
        print(f"    first 3 unexpected: {unexpected[:3]}")

    n_params = sum(p.numel() for p in model.parameters())
    n_trained = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params: {n_params:,} total, {n_trained:,} trainable")
    sample_param = model.vit_mae.vit.embeddings.patch_embeddings.projection.weight
    print(f"  patch_embed.proj.weight stats: mean={sample_param.mean().item():+.4f} "
          f"std={sample_param.std().item():.4f}  norm={sample_param.norm().item():.2f}")

    model.to(device).eval()
    model.vit_mae.config.mask_ratio = 0
    return model


@torch.no_grad()
def _decode_batch(model, latents_np, ids_restore_np, device):
    """latents_np (B, 197, 768) → images (B, 3, 224, 224) float32 in ImageNet-normalized space."""
    lat = torch.from_numpy(latents_np).to(device, dtype=torch.float32)
    ids = torch.from_numpy(ids_restore_np).to(device, dtype=torch.long)
    decoder_outputs = model.vit_mae.decoder(lat, ids)
    logits = decoder_outputs.logits
    imgs = model.vit_mae.unpatchify(logits)  # (B, 3, 224, 224)
    return imgs.cpu().numpy().astype(np.float32)


def _to_unit(imgs):
    """ImageNet-normalized (..., 3, H, W) → [0, 1] (..., 3, H, W)."""
    return np.clip(imgs * _IN_STD + _IN_MEAN, 0, 1)


def _psnr(gt, pred):
    """Both inputs in [0, 1], same shape (..., 3, H, W). Returns scalar PSNR."""
    mse = ((gt - pred) ** 2).mean()
    return float(10 * np.log10(1 / max(mse, 1e-12)))


def _save_side_by_side(panels, labels, out_path):
    """panels: list of (3, H, W) arrays in [0, 1]. Save horizontally concatenated."""
    H, W = panels[0].shape[1], panels[0].shape[2]
    sep = np.ones((3, H, 4), dtype=np.float32)
    parts = []
    for i, p in enumerate(panels):
        parts.append(p)
        if i < len(panels) - 1:
            parts.append(sep)
    composite = np.concatenate(parts, axis=2)
    arr = (composite.transpose(1, 2, 0) * 255).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(out_path)


def _list_test_frames(eval_frames_dir: Path, eid: str, cam: str):
    """Return list of (interval_idx, timebin_idx, path) for the test split, sorted."""
    import re
    pat = re.compile(r"interval(\d+)timebin(\d+)\.png$")
    test_dir = eval_frames_dir / f"_iblrig_{cam}Camera.raw.{eid}" / "test"
    triples = []
    for p in test_dir.glob("*.png"):
        m = pat.match(p.name)
        if m:
            triples.append((int(m.group(1)), int(m.group(2)), p))
    triples.sort()
    return triples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eid", required=True)
    ap.add_argument("--cam", choices=["left", "right"], required=True)
    ap.add_argument("--latents_h5", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--eval_frames_dir", required=True,
                    help="Parent of _iblrig_<cam>Camera.raw.<eid>/test/ — used to load original frames for PSNR.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--pca_ks", type=int, nargs="+", default=[3, 8, 50])
    ap.add_argument("--num_trials", type=int, default=5)
    ap.add_argument("--decode_batch_size", type=int, default=16)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}\noutput={out}")

    # ── Load model + show diagnostics ───────────────────────────────────────
    print("\n[load model]")
    model = _load_model(Path(args.ckpt_dir), device)

    # ── Load latents ────────────────────────────────────────────────────────
    print("\n[load latents]")
    with h5py.File(args.latents_h5, "r") as h5:
        Z_train = h5["latents_train"][:]
        Z_test = h5["latents_test"][:]
        ids_test = h5["ids_restore_test"][:]
        T = int(h5["n_timebins"][()])
    n_test, _, L, D = Z_test.shape
    print(f"  Z_train {Z_train.shape}  Z_test {Z_test.shape}  ids_test {ids_test.shape}")
    print(f"  Z_test stats: mean={Z_test.mean():+.3f}  std={Z_test.std():.3f}")

    # Pick trial indices: first num_trials test trials, timebin 30 (mid-trial)
    pick_trials = list(range(min(args.num_trials, n_test)))
    pick_t = T // 2
    print(f"  picking trials {pick_trials} at timebin {pick_t}")

    # ── Load original input frames for ground-truth PSNR ────────────────────
    print("\n[load original input frames for these trials]")
    triples = _list_test_frames(Path(args.eval_frames_dir), args.eid, args.cam)
    by_iv = {iv: [(tb, p) for (iv2, tb, p) in triples if iv2 == iv] for iv in pick_trials}
    orig_imgs = []
    for iv in pick_trials:
        match = [p for tb, p in by_iv[iv] if tb == pick_t]
        if not match:
            raise FileNotFoundError(f"no frame at trial={iv} timebin={pick_t}")
        img_t = _IMG_TF(Image.open(match[0]).convert("RGB"))  # (3, 224, 224) ImageNet-norm
        orig_imgs.append(img_t.numpy())
    orig_arr = np.stack(orig_imgs, axis=0)  # (N, 3, 224, 224) ImageNet-norm
    orig_unit = _to_unit(orig_arr)  # (N, 3, 224, 224) [0, 1]
    print(f"  loaded {len(orig_imgs)} input frames")

    # Pick the matching latents
    lat_pick = Z_test[pick_trials, pick_t]               # (N, 197, 768)
    ids_pick = ids_test[pick_trials, pick_t]             # (N, 196)

    # ── Check 1 — direct decode (no compression) ────────────────────────────
    print("\n" + "=" * 60)
    print("CHECK 1: NO COMPRESSION — Z_test → decoder → image")
    print("=" * 60)
    raw_decoded = _decode_batch(model, lat_pick, ids_pick, device)
    raw_unit = _to_unit(raw_decoded)
    psnr_raw_vs_orig = [_psnr(orig_unit[i], raw_unit[i]) for i in range(len(pick_trials))]
    print(f"  PSNR(decoded(Z_test) vs original input):")
    for i, p in zip(pick_trials, psnr_raw_vs_orig):
        print(f"    trial {i:3d}: {p:6.2f} dB")
    print(f"  mean PSNR = {np.mean(psnr_raw_vs_orig):.2f} dB  "
          f"(>20 dB ≈ model + extract pipeline are sound)")
    print(f"  decoded(Z_test) pixel stats: mean={raw_unit.mean():.3f}  std={raw_unit.std():.3f}  "
          f"(orig: mean={orig_unit.mean():.3f}  std={orig_unit.std():.3f})")

    # ── Check 2 — PCA round-trip (no TCN) ───────────────────────────────────
    print("\n" + "=" * 60)
    print("CHECK 2: PCA ROUND-TRIP — Z → PCA(k) → PCA⁻¹ → decoder → image")
    print("=" * 60)
    # Replicate decode_video.py normalization recipe (mean+std on train, per-token-position)
    Z_avg = Z_train.mean(axis=(0, 1), keepdims=True).astype(np.float32)  # (1,1,197,768)
    Z_std = Z_train.std(axis=(0, 1), keepdims=True).astype(np.float32)
    Z_std = np.maximum(Z_std, 1e-6)
    print(f"  Z_avg shape {Z_avg.shape}  Z_std shape {Z_std.shape}")
    print(f"  Z_avg stats: mean={Z_avg.mean():+.3f}  std={Z_avg.std():.3f}")
    print(f"  Z_std stats: min={Z_std.min():.3f}  median={np.median(Z_std):.3f}  max={Z_std.max():.3f}")

    Zn_train = (Z_train - Z_avg) / Z_std
    Zn_pick = (lat_pick - Z_avg[0, 0]) / Z_std[0, 0]   # (N, 197, 768)

    composite_panels_per_trial = {i: [orig_unit[idx], raw_unit[idx]] for idx, i in enumerate(pick_trials)}
    composite_labels = ["original", "decode(Z)"]

    print(f"\n  PSNR(decoded(PCA roundtrip) vs decoded(Z_test)):")
    print(f"  {'k':>4} | {'EVR':>7} | " + " ".join(f"trial{i:3d}" for i in pick_trials) + " |  mean")
    for k in args.pca_ks:
        pca = PCA(n_components=k, svd_solver="randomized", random_state=0)
        pca.fit(Zn_train.reshape(-1, D))
        evr = pca.explained_variance_ratio_.sum()
        # forward + inverse on the picked latents
        Zn_pick_flat = Zn_pick.reshape(-1, D)
        Zc = pca.transform(Zn_pick_flat)                       # (N*197, k)
        Zn_back = Zc @ pca.components_ + pca.mean_             # (N*197, 768)
        # restore (N, 197, 768) and apply per-token denorm
        Zn_back_3d = Zn_back.reshape(lat_pick.shape)
        Z_back = Zn_back_3d * Z_std[0, 0] + Z_avg[0, 0]        # (N, 197, 768)
        decoded_back = _decode_batch(model, Z_back.astype(np.float32), ids_pick, device)
        decoded_back_unit = _to_unit(decoded_back)
        psnrs = [_psnr(raw_unit[i], decoded_back_unit[i]) for i in range(len(pick_trials))]
        # also vs original
        psnrs_orig = [_psnr(orig_unit[i], decoded_back_unit[i]) for i in range(len(pick_trials))]
        print(f"  {k:>4} | {evr:>7.4f} | " + " ".join(f"{p:6.2f}dB" for p in psnrs)
              + f" | {np.mean(psnrs):6.2f}dB")
        print(f"        |   (vs orig)| " + " ".join(f"{p:6.2f}dB" for p in psnrs_orig)
              + f" | {np.mean(psnrs_orig):6.2f}dB")
        for idx, i in enumerate(pick_trials):
            composite_panels_per_trial[i].append(decoded_back_unit[idx])
        composite_labels.append(f"PCA-{k}")

    # ── Save composite PNGs ──────────────────────────────────────────────────
    print("\n[save composites]")
    print(f"  panels: {composite_labels}")
    for i in pick_trials:
        out_path = out / f"trial{i:03d}_t{pick_t:02d}_sanity.png"
        _save_side_by_side(composite_panels_per_trial[i], composite_labels, out_path)
        print(f"  wrote {out_path.name}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
