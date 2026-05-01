#!/bin/bash
# BEAST stage-1 only, no image augmentation, new training config.
# Skips stage-2 entirely — extracts raw 768-d CLS from stage-1 ViT-MAE.
# Reuses existing aligned.npz (neural_data_400_yizi/) and eval frames
# (extracted_frames_400_yizi/eval/) — only stage-1 is retrained.
#
# Outputs:
#   beast_data/checkpoints/beast_<cam>_noaug_<eid>/        # stage-1 ckpts (new)
#   beast_data/latents_beast_768_noaug/<eid>/z_trials.npz  # joint (V=2, D=768)
#   beast_data/latents_beast_768_noaug_left/<eid>/         # left-only slice
#   beast_data/latents_beast_768_noaug_right/<eid>/        # right-only slice
#   beast_data/latents_beast_768_noaug{,_left,_right}/<eid>/encoding_results.npy
#
# Config edits (already applied):
#   beast/configs/vit_beast.yaml: imgaug=none, train_bs=32, test_bs=32, epochs=400
set -Eeuo pipefail

source /home/haoshen/anaconda3/etc/profile.d/conda.sh
conda activate beast
cd /home/haoshen/Disk5/Encoding_Decoding/3DProject/beast

BD=/home/haoshen/Disk5/Encoding_Decoding/3DProject/beast_data
LOGS=$BD/logs
CKPT=$BD/checkpoints
FRAMES_FT=$BD/extracted_frames/finetune
FRAMES_EV=$BD/extracted_frames_400_yizi/eval
NEURAL=$BD/neural_data_400_yizi

LAT_JOINT=$BD/latents_beast_768_noaug
LAT_LEFT=$BD/latents_beast_768_noaug_left
LAT_RIGHT=$BD/latents_beast_768_noaug_right

EIDS=(
  4b00df29-3769-43be-bb40-128b1cba6d35
  72cb5550-43b4-4ef0-add5-e4adfdfb5e02
  781b35fd-e1f0-4d14-b2bb-95b7263082bb
  ecb5520d-1358-434c-95ec-93687ecd1396
  f312aaec-3b6f-44b3-86b4-3a0c119c0438
)
TAGS=(s1 s2 s3 s4 s5)

banner() { echo ""; echo "=== $* at $(date +'%F %T') ==="; }
CUR=""; trap 'banner "FAILED at: $CUR"; exit 1' ERR

# Sanity check: vit_beast.yaml has imgaug: none
banner "STEP 0: sanity check vit_beast.yaml"
if ! grep -q "^  imgaug: none" configs/vit_beast.yaml; then
  echo "ERROR: configs/vit_beast.yaml does not have 'imgaug: none'"
  grep "imgaug" configs/vit_beast.yaml
  exit 1
fi
echo "OK — imgaug=none confirmed"
grep -E "imgaug|train_batch|test_batch|num_epochs" configs/vit_beast.yaml

# ─── STEP 1: stage-1 BEAST per session (L + R parallel on 2 GPUs) ────────────
CUR="step 1"
banner "STEP 1: stage-1 BEAST × 5 sessions × 2 cams (no-aug)"
for i in 0 1 2 3 4; do
  EID=${EIDS[i]}
  TAG=${TAGS[i]}
  banner "STEP 1.$TAG: stage-1 L (GPU0) || R (GPU1) for $EID"
  CUDA_VISIBLE_DEVICES=0 beast train \
    --config configs/vit_beast.yaml \
    --data   "$FRAMES_FT/_iblrig_leftCamera.raw.$EID" \
    --output "$CKPT/beast_left_noaug_$EID" \
    > "$LOGS/noaug_stage1_${TAG}_left.log" 2>&1 &
  P_L=$!
  CUDA_VISIBLE_DEVICES=1 beast train \
    --config configs/vit_beast.yaml \
    --data   "$FRAMES_FT/_iblrig_rightCamera.raw.$EID" \
    --output "$CKPT/beast_right_noaug_$EID" \
    > "$LOGS/noaug_stage1_${TAG}_right.log" 2>&1 &
  P_R=$!
  wait $P_L $P_R
  echo "  $TAG stage-1 done at $(date +%T)"
done
banner "STEP 1: all stage-1 done"

# ─── STEP 2: extract 768-d CLS over eval frames (joint per session) ──────────
CUR="step 2"
banner "STEP 2: extract 768-d CLS × 5 sessions"
for i in 0 1 2 3 4; do
  EID=${EIDS[i]}
  TAG=${TAGS[i]}
  banner "STEP 2.$TAG: CLS extract for $EID"
  python scripts/extract_latents.py \
    --eid "$EID" \
    --eval_frames_dir "$FRAMES_EV" \
    --left_ckpt_dir   "$CKPT/beast_left_noaug_$EID/tb_logs/version_0" \
    --right_ckpt_dir  "$CKPT/beast_right_noaug_$EID/tb_logs/version_0" \
    --output_dir      "$LAT_JOINT" \
    --mode cls \
    > "$LOGS/noaug_cls_extract_${TAG}.log" 2>&1
  echo "  $TAG CLS extract done at $(date +%T)"
done
banner "STEP 2: all CLS extracts done"

# ─── STEP 3: slice joint into left-only / right-only ─────────────────────────
CUR="step 3"
banner "STEP 3: slice joint into left/right per session"
python - <<'PYEOF'
import numpy as np, os
EIDS = ["4b00df29-3769-43be-bb40-128b1cba6d35",
        "72cb5550-43b4-4ef0-add5-e4adfdfb5e02",
        "781b35fd-e1f0-4d14-b2bb-95b7263082bb",
        "ecb5520d-1358-434c-95ec-93687ecd1396",
        "f312aaec-3b6f-44b3-86b4-3a0c119c0438"]
BD = "/home/haoshen/Disk5/Encoding_Decoding/3DProject/beast_data"
LJ = f"{BD}/latents_beast_768_noaug"
LL = f"{BD}/latents_beast_768_noaug_left"
LR = f"{BD}/latents_beast_768_noaug_right"
for eid in EIDS:
    src = np.load(f"{LJ}/{eid}/z_trials.npz")
    z = src["z_trials_time"]; ts = src["trial_split"]
    print(f"  {eid[:8]}: joint shape={z.shape} split={ts.tolist()}")
    for d, arr in [(LL, z[:, :, 0:1, :]), (LR, z[:, :, 1:2, :])]:
        os.makedirs(f"{d}/{eid}", exist_ok=True)
        np.savez(f"{d}/{eid}/z_trials.npz", z_trials_time=arr, trial_split=ts)
PYEOF
banner "STEP 3: slicing done"

# ─── STEP 4: encoding eval — joint first (5 runs, 2 GPUs in parallel) ────────
CUR="step 4 joint"
banner "STEP 4a: encoding eval × 5 sessions, JOINT"
JOINT_RUNS=()
for i in 0 1 2 3 4; do
  JOINT_RUNS+=("${TAGS[i]}|${EIDS[i]}|joint|$LAT_JOINT")
done
N=${#JOINT_RUNS[@]}
for ((i=0; i<N; i+=2)); do
  IFS='|' read -r T1 E1 C1 L1 <<< "${JOINT_RUNS[i]}"
  J=$((i+1))
  if [ $J -lt $N ]; then
    IFS='|' read -r T2 E2 C2 L2 <<< "${JOINT_RUNS[J]}"
    STD_MODE=per_feature CUDA_VISIBLE_DEVICES=0 python encoding_decoding_code/test.py \
      --eid "$E1" --neural_input_dir "$NEURAL" --latent_input_dir "$L1" \
      --eval_task encoding > "$LOGS/noaug_enc_${T1}_${C1}.log" 2>&1 &
    P1=$!
    STD_MODE=per_feature CUDA_VISIBLE_DEVICES=1 python encoding_decoding_code/test.py \
      --eid "$E2" --neural_input_dir "$NEURAL" --latent_input_dir "$L2" \
      --eval_task encoding > "$LOGS/noaug_enc_${T2}_${C2}.log" 2>&1 &
    P2=$!
    wait $P1 $P2
    echo "  joint round done at $(date +%T): $T1 + $T2"
  else
    STD_MODE=per_feature CUDA_VISIBLE_DEVICES=0 python encoding_decoding_code/test.py \
      --eid "$E1" --neural_input_dir "$NEURAL" --latent_input_dir "$L1" \
      --eval_task encoding > "$LOGS/noaug_enc_${T1}_${C1}.log" 2>&1
    echo "  joint solo done at $(date +%T): $T1"
  fi
done
banner "STEP 4a: joint eval done"

# ─── STEP 5: encoding eval — left + right (10 runs, 2 GPUs in parallel) ──────
CUR="step 5 left/right"
banner "STEP 4b: encoding eval × 5 sessions × {left, right}"
LR_RUNS=()
for i in 0 1 2 3 4; do
  LR_RUNS+=("${TAGS[i]}|${EIDS[i]}|left|$LAT_LEFT")
  LR_RUNS+=("${TAGS[i]}|${EIDS[i]}|right|$LAT_RIGHT")
done
N=${#LR_RUNS[@]}
for ((i=0; i<N; i+=2)); do
  IFS='|' read -r T1 E1 C1 L1 <<< "${LR_RUNS[i]}"
  J=$((i+1))
  IFS='|' read -r T2 E2 C2 L2 <<< "${LR_RUNS[J]}"
  STD_MODE=per_feature CUDA_VISIBLE_DEVICES=0 python encoding_decoding_code/test.py \
    --eid "$E1" --neural_input_dir "$NEURAL" --latent_input_dir "$L1" \
    --eval_task encoding > "$LOGS/noaug_enc_${T1}_${C1}.log" 2>&1 &
  P1=$!
  STD_MODE=per_feature CUDA_VISIBLE_DEVICES=1 python encoding_decoding_code/test.py \
    --eid "$E2" --neural_input_dir "$NEURAL" --latent_input_dir "$L2" \
    --eval_task encoding > "$LOGS/noaug_enc_${T2}_${C2}.log" 2>&1 &
  P2=$!
  wait $P1 $P2
  echo "  L/R round done at $(date +%T): $T1/$C1 + $T2/$C2"
done
banner "STEP 4b: left/right eval done"

# ─── STEP 6: summary ─────────────────────────────────────────────────────────
banner "STEP 5: summary"
python - <<'PYEOF' | tee "$LOGS/beast_noaug_summary.txt"
import numpy as np
BD = "/home/haoshen/Disk5/Encoding_Decoding/3DProject/beast_data"
SESS = {"s1":"4b00df29-3769-43be-bb40-128b1cba6d35",
        "s2":"72cb5550-43b4-4ef0-add5-e4adfdfb5e02",
        "s3":"781b35fd-e1f0-4d14-b2bb-95b7263082bb",
        "s4":"ecb5520d-1358-434c-95ec-93687ecd1396",
        "s5":"f312aaec-3b6f-44b3-86b4-3a0c119c0438"}
print("BEAST 768-d raw CLS — no augmentation, new training cfg, per_feature (nan-safe)")
print(f"{'Sess':<4} {'Cfg':<6} {'CNN BPS':>9} {'CNN R2':>9} {'RRR BPS':>9} {'RRR R2':>9}")
print("-"*52)
for tag, eid in SESS.items():
    for cfg, dir_ in [("joint", "latents_beast_768_noaug"),
                      ("left",  "latents_beast_768_noaug_left"),
                      ("right", "latents_beast_768_noaug_right")]:
        try:
            r = np.load(f"{BD}/{dir_}/{eid}/encoding_results.npy", allow_pickle=True).item()[eid]
            print(f"{tag:<4} {cfg:<6} {r['cnn']['bps']:>9.4f} {r['cnn']['r2']:>9.4f} {r['rrr']['bps']:>9.4f} {r['rrr']['r2']:>9.4f}")
        except FileNotFoundError:
            print(f"{tag:<4} {cfg:<6} (missing)")
PYEOF
banner "ALL DONE"
