# Reproducing the video-decoding pipeline

End-to-end reproduction of the PCA-6 video-decoding results for **5 IBL
sessions × {left, right} cameras**: from raw IBL access through stage-1
BEAST training to per-frame PSNR + SSIM.

All numbers below were observed on this run; treat them as sanity checks at
each stage. Wall-clock times are for 2× RTX A6000.

## Sessions

These five EIDs are the canonical set used throughout the project (also
hard-coded in `scripts/beast_noaug_rerun.sh` and
`scripts/decode_video_5sess_pca6.sh`):

```
s1: 4b00df29-3769-43be-bb40-128b1cba6d35
s2: 72cb5550-43b4-4ef0-add5-e4adfdfb5e02
s3: 781b35fd-e1f0-4d14-b2bb-95b7263082bb
s4: ecb5520d-1358-434c-95ec-93687ecd1396
s5: f312aaec-3b6f-44b3-86b4-3a0c119c0438
```

## Key invariants

These shapes/values are assumed throughout. If any step produces something
else, stop and investigate.

| Quantity | Value |
|---|---|
| Trials per session | **400** (`--num_trials 400`) |
| Train / val / test split | **280 / 40 / 80** |
| Timebins per trial | **60** |
| Frame size (model input) | **224 × 224** RGB |
| ViT-MAE encoder hidden dim | **768** |
| Tokens per frame (encoder output) | **197** = 1 CLS + 196 patches |
| Stage-1 mask ratio | 0.75 (training) → 0 (latent extraction) |

## Environment

```bash
conda env create -f environment.yml      # or follow the existing setup
conda activate beast
pip install 'torchmetrics[image]'         # for canonical PSNR + SSIM
```

System requirements:
- CUDA-capable GPU (≥ 24 GB VRAM recommended for stage-1 training)
- ~50 GB free disk per session for raw IBL data + extracted frames
- `/usr/bin/ffmpeg` with libx264 if you want MP4 visualizations
  (conda's `libopenh264` is broken on Linux as of 2026-04 — use system ffmpeg)

---

## Step 1 — Download raw IBL data

### 1a. Videos (~7 GB / session)

```bash
cd scripts
bash download_videos.sh <ONE_CACHE_PATH> <OUTPUT_DIR>
```

Internally calls `download_videos.py` for the 5 EIDs above and saves
both leftCamera and rightCamera MP4s to
`<OUTPUT_DIR>/_iblrig_<cam>Camera.raw.<eid>.mp4`.

### 1b. Frame timestamps

```bash
bash download_timestamps.sh <ONE_CACHE_PATH> <TIMESTAMP_OUT>
```

Writes per-session `*_iblrig_<cam>Camera.times.<eid>.npy` files (one
timestamp per video frame). **Left and right share the same timestamps**
inside a session.

---

## Step 2 — Align neural + behavior data (400 trials)

```bash
cd scripts
bash batch_extract_neural_data.sh \
  <ONE_CACHE_PATH> \
  <TIMESTAMP_OUT> \
  <NEURAL_OUT> \
  400
```

Runs `extract_neural_data.sh` with `--num_trials 400` for each of the 5
EIDs. Output per session:

```
<NEURAL_OUT>/<eid>/<eid>_aligned.npz
```

### Expected schema (sanity check on s1)

```python
import numpy as np
d = np.load(f"<NEURAL_OUT>/{eid}/{eid}_aligned.npz")
assert d["train_spikes"].shape == (280, 60, N_neurons)   # s1 has 740 neurons
assert d["val_spikes"].shape   == (40,  60, N_neurons)
assert d["test_spikes"].shape  == (80,  60, N_neurons)
assert d["train_intervals"].shape == (280, 2)
# Plus 8 behavior keys (wheel_speed, licks, *whisker, *nose, *paw) per split
```

---

## Step 3 — Extract finetune frames (stage-1 training data)

```bash
cd scripts
bash extract.sh \
  <VIDEO_DIR>/_iblrig_leftCamera.raw.<eid>.mp4 \
  <FRAMES_FT>/_iblrig_leftCamera.raw.<eid> \
  pca_kmeans 700
bash extract.sh \
  <VIDEO_DIR>/_iblrig_rightCamera.raw.<eid>.mp4 \
  <FRAMES_FT>/_iblrig_rightCamera.raw.<eid> \
  precomputed 700
```

Frames are sampled to be visually diverse. Result: ~700 PNGs per camera per
session under `<FRAMES_FT>/_iblrig_<cam>Camera.raw.<eid>/`.

---

## Step 4 — Extract eval frames (timestamp method)

This is the **per-trial × per-timebin grid** used at inference time. One PNG
per (interval, timebin), where each interval = one trial and each timebin
spans (interval / 60) seconds.

```bash
cd scripts
bash extract.sh \
  <VIDEO_DIR>/_iblrig_leftCamera.raw.<eid>.mp4 \
  <FRAMES_EV>/_iblrig_leftCamera.raw.<eid> \
  timestamp 0 \
  <TIMESTAMP_OUT> \
  <NEURAL_OUT>
```

The `timestamp` method uses `<NEURAL_OUT>/<eid>/<eid>_aligned.npz` to find
the (train, val, test) intervals and samples 60 frames per interval. Output:

```
<FRAMES_EV>/_iblrig_<cam>Camera.raw.<eid>/
  train/intervalNtimebinM.png   (280 × 60 = 16800 PNGs)
  val/intervalNtimebinM.png     (40  × 60 = 2400 PNGs)
  test/intervalNtimebinM.png    (80  × 60 = 4800 PNGs)
```

---

## Step 5 — Train stage-1 BEAST per (session × cam)

10 separate trainings (5 sessions × 2 cameras), no augmentation. The wrapper
script `scripts/beast_noaug_rerun.sh` runs all 10 in pairs across 2 GPUs;
launch it with paths set as constants at the top of the script. The
underlying command per (eid, cam):

```bash
beast train \
  --config configs/vit_beast.yaml \
  --data   <FRAMES_FT>/_iblrig_<cam>Camera.raw.<eid> \
  --output <CKPT_BASE>/beast_<cam>_noaug_<eid>
```

`configs/vit_beast.yaml` key settings:
- `imgaug: none` (no augmentation)
- `train_batch_size: 32`, `test_batch_size: 32`
- `num_epochs: 400`
- `mask_ratio: 0.75`, `num_latents: null` (no AE bottleneck — raw 768-d output)
- `random_init: False` (start from `facebook/vit-mae-base`)

Output:

```
<CKPT_BASE>/beast_<cam>_noaug_<eid>/
  config.yaml
  tb_logs/version_0/
    checkpoints/last.ckpt
    checkpoints/{epoch}-{step}-best.ckpt
```

**Wall-clock**: ~10 h end-to-end for 10 trainings on 2 × A6000.

### Sanity check after step 5

`beast/scripts/decode_video_sanity.py` Check 1 should report mean PSNR
≈ **19 dB** when raw `decode(Z_test)` is run through the ViT-MAE decoder.
Lower than that suggests the ckpt didn't load.

---

## Step 6 — Extract unshuffled 197-token latents

**This step contains the shuffle fix** (commit `cd50b10`). HF
`ViTMAEModel.random_masking()` shuffles patches even at `mask_ratio=0`;
`extract_full_latents.py` now uses `ids_restore` to gather patches back to
canonical spatial order before saving and stores identity `ids_restore` so
the decoder's internal gather is a no-op. Mathematically equivalent to
running the encoder on input-in-spatial-order (PE added before shuffling,
attention permutation-equivariant) — but downstream PCA + TCN now see
positionally-stable tokens.

Per (eid, cam):

```bash
cd beast/
python scripts/extract_full_latents.py \
  --eid <eid> \
  --eval_frames_dir <FRAMES_EV> \
  --cam <left|right> \
  --ckpt_dir   <CKPT_BASE>/beast_<cam>_noaug_<eid>/tb_logs/version_0 \
  --output_h5  <DECODE_BASE>/<eid>/<cam>_full_unshuffled.h5 \
  --batch_size 32 --num_workers 4
```

### Expected output schema

```python
import h5py
with h5py.File(out, "r") as h5:
    assert h5["latents_train"].shape == (280, 60, 197, 768)
    assert h5["latents_val"  ].shape == (40,  60, 197, 768)
    assert h5["latents_test" ].shape == (80,  60, 197, 768)
    assert h5["ids_restore_test"].shape == (80, 60, 196)
    # ids_restore is identity arange(196) per (k, t) after the fix
    import numpy as np
    np.testing.assert_array_equal(h5["ids_restore_test"][0, 0], np.arange(196))
```

H5 size: ~5–7 GB per (eid, cam) with LZF compression.

### Sanity check after step 6

`scripts/decode_video_sanity.py` Check 2 (PCA round-trip without TCN):

| k | Expected EVR sum | PSNR vs raw decode |
|---|---|---|
| 3 | ≈ 10% | ≈ 26 dB |
| 8 | ≈ 18% | ≈ 27 dB |
| 50 | ≈ 43% | ≈ 32 dB |

If you see EVR ≈ 33% for k=3 and PSNR ≈ 17 dB at k=3, the latents are still
shuffled — re-run extract with the fixed script.

---

## Step 7 — Decode video (PCA-6 + TCN)

Per (eid, cam):

```bash
cd beast/
export PYTHONPATH=$(pwd)/encoding_decoding_code:${PYTHONPATH:-}
python scripts/decode_video.py \
  --eid <eid> \
  --latents_h5 <DECODE_BASE>/<eid>/<cam>_full_unshuffled.h5 \
  --neural_npz <NEURAL_OUT>/<eid>/<eid>_aligned.npz \
  --ckpt_dir   <CKPT_BASE>/beast_<cam>_noaug_<eid>/tb_logs/version_0 \
  --output_dir <DECODE_BASE>/<eid>/pca6_unshuffled_<cam> \
  --pca_components 6 --num_samples 20
```

Or use the 5-session orchestrator (parallelizes across 2 GPUs, resumable):

```bash
bash scripts/decode_video_5sess_pca6.sh
```

### Outputs per run

| File | Purpose |
|---|---|
| `pca.npz` | Saved PCA mean / components / Z_avg / Z_std |
| `z_compressed.npz` | PCA-6 coords for train/val/test (Zc) |
| `z_compressed_pred.npz` | TCN's PCA-6 prediction on test |
| `decoder_results.npy` | Full TCN results dict |
| `psnr.npz` | Legacy 224-px per-frame PSNR (kept for back-compat) |
| `psnr_ssim_metrics.npz` | **Canonical** PSNR + SSIM (320 px BICUBIC, data_range=1.0) |
| `summary.yaml` | Scalars (TCN R², PSNR mean/std, SSIM mean/std, EVR, …) |
| `vis_frames/{random,best}_trial<NNN>/{gt,pred}_timebin<TT>.png` | 60-frame trials |

### Expected numbers (PCA-6, post-fix)

```
sess  cam       TCN R²  PSNR_320  SSIM_320
-------------------------------------------
s1    left      0.379    25.006    0.9359
s1    right     0.060    22.262    0.9039
s2    left      0.371    22.373    0.9117
s2    right     0.013    21.770    0.9061
s3    left      0.467    20.266    0.8703
s3    right     0.038    19.863    0.8741
s4    left      0.395    23.622    0.9236
s4    right     0.128    22.690    0.9136
s5    left      0.369    22.335    0.9046
s5    right     0.089    22.649    0.9073
```

Mean **left**: R²=0.385, PSNR=22.7, SSIM=0.911
Mean **right**: R²=0.066, PSNR=21.9, SSIM=0.901

Differences within ±5% on a fresh re-run are expected (PCA SVD seed is fixed
to 0 but TCN initialization + Ray Tune scheduling have stochastic components).

**Wall-clock**: ~30 min per (eid, cam) on one GPU; ~2 h end-to-end with 2 GPUs.

---

## Backfilling canonical PSNR + SSIM on existing runs

If you already have decode_video runs from before the metric overhaul, you
can recompute canonical 320-px PSNR + SSIM without re-running TCN:

```bash
python scripts/recompute_canonical_metrics.py --auto_5sess
```

This reads each run's saved `pca.npz` + `z_compressed_pred.npz`,
reconstructs `Z_pred` via PCA inverse + denorm, runs the ViT-MAE decoder
for GT and pred, computes canonical metrics, writes
`psnr_ssim_metrics.npz`, and patches `summary.yaml` (legacy 224-px PSNR
preserved as `psnr_mean_legacy_224`).

---

## Recommended directory layout

```
<ONE_CACHE_PATH>                           ← step 1, raw IBL cache
<VIDEO_DIR>/                               ← step 1, downloaded MP4s
<TIMESTAMP_OUT>/                           ← step 1, frame timestamps
<NEURAL_OUT>/<eid>/<eid>_aligned.npz       ← step 2
<FRAMES_FT>/_iblrig_<cam>Camera.raw.<eid>/ ← step 3, finetune frames
<FRAMES_EV>/_iblrig_<cam>Camera.raw.<eid>/ ← step 4, eval frames (train/val/test)
<CKPT_BASE>/beast_<cam>_noaug_<eid>/       ← step 5, stage-1 ckpts
<DECODE_BASE>/<eid>/<cam>_full_unshuffled.h5  ← step 6, latents
<DECODE_BASE>/<eid>/pca6_unshuffled_<cam>/    ← step 7, decode results
```

Setting these to absolute paths at the top of `beast_noaug_rerun.sh` and
`decode_video_5sess_pca6.sh` means the wrappers automate steps 5–7
end-to-end.

---

## Common pitfalls

1. **`scripts/extract_full_latents.py` save the shuffled output** — fixed in
   commit `cd50b10`. Verify by checking that `ids_restore_test[0, 0]` in
   the output h5 equals `np.arange(196)`.

2. **`STD_MODE=per_feature` for encoding eval** — required env var when
   running `encoding_decoding_code/test.py`; not relevant for video
   decoding (`decode_video.py` does its own per-token-position normalization
   internally).

3. **Conda's libopenh264 is broken** — when generating MP4 visualizations
   from PNG frames, use `/usr/bin/ffmpeg` (system) with `libx264`, not
   conda's ffmpeg.

4. **MAE decoder distribution shift** — trained at `mask_ratio=0.75` (50
   visible + 147 mask tokens), used at `mask_ratio=0` (197 visible). This
   caps any decoded image at ≈ 19 dB PSNR vs original input. Predictions
   that score > 19 dB vs raw `decode(Z_test)` are normal — they're closer
   to the (already mildly blurry) decoder output, not to the original frame.

5. **PSNR / SSIM are not enough.** They're dominated by background /
   lighting / scene reconstruction quality. For "is the pred trial-specific":
   look at TCN R², the cross-trial L1 diagnostic, and within-trial temporal
   variation (see `decoding_review_2026-05-02/REPORT.md` §4–5 for full
   discussion).
