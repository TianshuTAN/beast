#!/bin/bash
# Video decoding for all 5 sessions × left+right cameras, PCA-6.
# Uses the unshuffled extract (extract_full_latents.py post-fix).
# Resumable: skips any (eid, cam) whose final summary.yaml already exists.
#
# Phase 1: ensure each (eid, cam) has an unshuffled h5
#   - s1 left/right: patch existing shuffled h5 (CPU, ~2.5min each)
#   - s2-s5 left/right: re-extract from frames (GPU, ~5-10min each)
# Phase 2: decode_video.py --pca_components 6 --num_samples 20 (~30min each)
# Phase 3: aggregate summary across all 10 runs
#
# Two GPUs in parallel where possible.

set -Eeuo pipefail

source /home/haoshen/anaconda3/etc/profile.d/conda.sh
conda activate beast
cd /home/haoshen/Disk5/Encoding_Decoding/3DProject/beast

REPO=/home/haoshen/Disk5/Encoding_Decoding/3DProject/beast
BD=/home/haoshen/Disk5/Encoding_Decoding/3DProject/beast_data
LOGS=$BD/logs
DEC=$BD/decoding_video
NEURAL=$BD/neural_data_400_yizi
FRAMES=$BD/extracted_frames_400_yizi/eval
CKPT_BASE=$BD/checkpoints

EIDS=(
  4b00df29-3769-43be-bb40-128b1cba6d35
  72cb5550-43b4-4ef0-add5-e4adfdfb5e02
  781b35fd-e1f0-4d14-b2bb-95b7263082bb
  ecb5520d-1358-434c-95ec-93687ecd1396
  f312aaec-3b6f-44b3-86b4-3a0c119c0438
)
TAGS=(s1 s2 s3 s4 s5)
CAMS=(left right)

PCA_K=6
NUM_SAMPLES=20

banner() { echo ""; echo "=== [$(date +'%F %T')] $* ==="; }

# ───────────────────────────────────────────────────────────────────────────
# PHASE 1: ensure unshuffled h5s exist
# ───────────────────────────────────────────────────────────────────────────
banner "PHASE 1: prepare unshuffled h5 for all (eid, cam)"

ensure_h5() {
  local eid=$1 cam=$2 gpu=$3
  local outdir="$DEC/$eid"
  local h5_unshuf="$outdir/${cam}_full_unshuffled.h5"
  local h5_old="$outdir/${cam}_full.h5"
  local ckpt="$CKPT_BASE/beast_${cam}_noaug_${eid}/tb_logs/version_0"
  mkdir -p "$outdir"

  if [ -f "$h5_unshuf" ]; then
    echo "  [skip] $h5_unshuf already exists"
    return 0
  fi

  if [ -f "$h5_old" ]; then
    # Patch existing shuffled h5 (CPU only)
    echo "  [patch] $h5_old → $h5_unshuf  (~2.5min, CPU)"
    python - <<PYEOF
import h5py, numpy as np, time
src, dst = "$h5_old", "$h5_unshuf"
t0 = time.time()
with h5py.File(src, "r") as s, h5py.File(dst, "w") as d:
    d.create_dataset("trial_split", data=s["trial_split"][:])
    d.create_dataset("n_timebins", data=s["n_timebins"][()])
    T = int(s["n_timebins"][()])
    for split in ("train", "val", "test"):
        lat = s[f"latents_{split}"][:]
        ids = s[f"ids_restore_{split}"][:]
        K = lat.shape[0]
        cls = lat[:, :, 0:1, :]
        patches_shuf = lat[:, :, 1:, :]
        idx = np.broadcast_to(ids[..., None], patches_shuf.shape)
        patches_spat = np.take_along_axis(patches_shuf, idx, axis=2)
        lat_canon = np.concatenate([cls, patches_spat], axis=2).astype(np.float32)
        ids_id = np.broadcast_to(np.arange(196, dtype=np.int64), (K, T, 196))
        d.create_dataset(f"latents_{split}", data=lat_canon, chunks=(1, T, 197, 768), compression="lzf")
        d.create_dataset(f"ids_restore_{split}", data=ids_id, chunks=(1, T, 196), compression="lzf")
print(f"    patched in {time.time()-t0:.1f}s")
PYEOF
    return 0
  fi

  # No h5 at all — re-extract using fixed extract_full_latents.py
  if [ ! -d "$ckpt/checkpoints" ]; then
    echo "  [FATAL] missing ckpt at $ckpt"
    return 1
  fi
  echo "  [extract] GPU $gpu → $h5_unshuf  (~5-10min)"
  CUDA_VISIBLE_DEVICES=$gpu python scripts/extract_full_latents.py \
    --eid "$eid" --eval_frames_dir "$FRAMES" --cam "$cam" \
    --ckpt_dir "$ckpt" --output_h5 "$h5_unshuf" \
    --batch_size 32 --num_workers 4 \
    > "$LOGS/decode5_extract_${cam}_${eid:0:8}.log" 2>&1
}

# Pair (eid, cam) jobs into 2-GPU parallel batches
phase1_jobs=()
for i in 0 1 2 3 4; do
  for cam in "${CAMS[@]}"; do
    h5="$DEC/${EIDS[i]}/${cam}_full_unshuffled.h5"
    [ -f "$h5" ] && continue
    phase1_jobs+=("${EIDS[i]}|$cam")
  done
done
echo "  ${#phase1_jobs[@]} (eid, cam) pairs need an unshuffled h5"

idx=0
while [ $idx -lt ${#phase1_jobs[@]} ]; do
  IFS='|' read -r e1 c1 <<< "${phase1_jobs[$idx]}"
  ensure_h5 "$e1" "$c1" 0 &
  P1=$!
  if [ $((idx + 1)) -lt ${#phase1_jobs[@]} ]; then
    IFS='|' read -r e2 c2 <<< "${phase1_jobs[$((idx + 1))]}"
    ensure_h5 "$e2" "$c2" 1 &
    P2=$!
    wait $P1 $P2
    idx=$((idx + 2))
  else
    wait $P1
    idx=$((idx + 1))
  fi
done
banner "PHASE 1 done"

# ───────────────────────────────────────────────────────────────────────────
# PHASE 2: decode_video.py --pca_components 6 for each (eid, cam)
# ───────────────────────────────────────────────────────────────────────────
banner "PHASE 2: decode_video PCA-${PCA_K} for all (eid, cam)"

run_decode() {
  local eid=$1 cam=$2 gpu=$3
  local outdir="$DEC/$eid/pca${PCA_K}_unshuffled_${cam}"
  local h5="$DEC/$eid/${cam}_full_unshuffled.h5"
  local ckpt="$CKPT_BASE/beast_${cam}_noaug_${eid}/tb_logs/version_0"
  local neural="$NEURAL/$eid/${eid}_aligned.npz"

  if [ -f "$outdir/summary.yaml" ]; then
    echo "  [skip] $outdir/summary.yaml exists"
    return 0
  fi
  echo "  [decode] GPU $gpu → $outdir  (~30min)"
  export PYTHONPATH=$REPO/encoding_decoding_code:${PYTHONPATH:-}
  CUDA_VISIBLE_DEVICES=$gpu python scripts/decode_video.py \
    --eid "$eid" --latents_h5 "$h5" --neural_npz "$neural" \
    --ckpt_dir "$ckpt" --output_dir "$outdir" \
    --pca_components $PCA_K --num_samples $NUM_SAMPLES \
    > "$LOGS/decode5_decode_${cam}_${eid:0:8}.log" 2>&1
}

# s1 left already done at $DEC/$EID_S1/pca6_unshuffled (no _left suffix).
# Copy/symlink to standardized name so the aggregator finds it.
S1=${EIDS[0]}
if [ -d "$DEC/$S1/pca${PCA_K}_unshuffled" ] && [ ! -e "$DEC/$S1/pca${PCA_K}_unshuffled_left" ]; then
  ln -s "pca${PCA_K}_unshuffled" "$DEC/$S1/pca${PCA_K}_unshuffled_left"
  echo "  [link] s1 left: pca${PCA_K}_unshuffled → pca${PCA_K}_unshuffled_left"
fi

phase2_jobs=()
for i in 0 1 2 3 4; do
  for cam in "${CAMS[@]}"; do
    od="$DEC/${EIDS[i]}/pca${PCA_K}_unshuffled_${cam}"
    [ -f "$od/summary.yaml" ] && continue
    phase2_jobs+=("${EIDS[i]}|$cam")
  done
done
echo "  ${#phase2_jobs[@]} (eid, cam) pairs need decode_video"

idx=0
while [ $idx -lt ${#phase2_jobs[@]} ]; do
  IFS='|' read -r e1 c1 <<< "${phase2_jobs[$idx]}"
  run_decode "$e1" "$c1" 0 &
  P1=$!
  if [ $((idx + 1)) -lt ${#phase2_jobs[@]} ]; then
    IFS='|' read -r e2 c2 <<< "${phase2_jobs[$((idx + 1))]}"
    run_decode "$e2" "$c2" 1 &
    P2=$!
    wait $P1 $P2
    idx=$((idx + 2))
  else
    wait $P1
    idx=$((idx + 1))
  fi
  echo "  [progress] $idx / ${#phase2_jobs[@]} done at $(date +%T)"
done
banner "PHASE 2 done"

# ───────────────────────────────────────────────────────────────────────────
# PHASE 3: aggregate summary across all 10 runs
# ───────────────────────────────────────────────────────────────────────────
banner "PHASE 3: aggregate"
python - <<'PYEOF' | tee "$LOGS/decode5_pca6_summary.txt"
import yaml
from pathlib import Path

DEC = Path("/home/haoshen/Disk5/Encoding_Decoding/3DProject/beast_data/decoding_video")
SESS = [("s1","4b00df29-3769-43be-bb40-128b1cba6d35"),
        ("s2","72cb5550-43b4-4ef0-add5-e4adfdfb5e02"),
        ("s3","781b35fd-e1f0-4d14-b2bb-95b7263082bb"),
        ("s4","ecb5520d-1358-434c-95ec-93687ecd1396"),
        ("s5","f312aaec-3b6f-44b3-86b4-3a0c119c0438")]

print("PCA-6 unshuffled video-decoding results, all 5 sessions × {left, right}")
print(f"{'sess':<5} {'cam':<6} {'TCN R2':>9} {'PSNR mean':>10} {'PSNR std':>9} {'EVR sum':>9}")
print("-" * 56)
for tag, eid in SESS:
    for cam in ("left", "right"):
        d = DEC / eid / f"pca6_unshuffled_{cam}"
        f = d / "summary.yaml"
        if not f.is_file():
            print(f"{tag:<5} {cam:<6}  (missing)")
            continue
        s = yaml.safe_load(f.open())
        evr = sum(s.get("pca_explained_variance", []))
        print(f"{tag:<5} {cam:<6} {s['tcn_test_r2']:>9.4f} "
              f"{s['psnr_mean']:>10.3f} {s['psnr_std']:>9.3f} {evr:>9.4f}")
PYEOF

banner "ALL DONE"
