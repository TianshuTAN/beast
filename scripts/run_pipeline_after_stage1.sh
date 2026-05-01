#!/usr/bin/env bash
# Chain: wait for stage-1 BEAST L+R → CLS extract → CLS-AE → BEAST-AE patchconv (L||R)
# → extract bottleneck latents → encoding+decoding eval → BEAST-AE warmstart (L||R)
# → extract + eval. Resumable via .done markers under $STATE_DIR.
#
# Launch example:
#   nohup bash scripts/run_pipeline_after_stage1.sh \
#     > /home/haoshen/Disk5/Encoding_Decoding/3DProject/beast_data/logs/pipeline.nohup.log 2>&1 &

set -euo pipefail

############################################################
# CONFIG — edit here if paths/EID change
############################################################
EID="4b00df29-3769-43be-bb40-128b1cba6d35"
ROOT="/home/haoshen/Disk5/Encoding_Decoding/3DProject"
REPO="${ROOT}/beast"
DATA_ROOT="${ROOT}/beast_data"

FINETUNE_ROOT="${DATA_ROOT}/extracted_frames/finetune"
EVAL_FRAMES_DIR="${DATA_ROOT}/extracted_frames/eval"
LEFT_DATA="${FINETUNE_ROOT}/_iblrig_leftCamera.raw.${EID}"
RIGHT_DATA="${FINETUNE_ROOT}/_iblrig_rightCamera.raw.${EID}"

STAGE1_L_OUT="${DATA_ROOT}/checkpoints/beast_left_${EID:0:8}"
STAGE1_R_OUT="${DATA_ROOT}/checkpoints/beast_right_${EID:0:8}"
STAGE1_L_CKPT_DIR="${STAGE1_L_OUT}/tb_logs/version_0"
STAGE1_R_CKPT_DIR="${STAGE1_R_OUT}/tb_logs/version_0"
STAGE1_L_CKPT="${STAGE1_L_CKPT_DIR}/checkpoints/last.ckpt"
STAGE1_R_CKPT="${STAGE1_R_CKPT_DIR}/checkpoints/last.ckpt"

CLS_LATENT_DIR="${DATA_ROOT}/latents_cls"
CLSAE_DIR="${DATA_ROOT}/clsae"
CLSAE_CAM0_CKPT="${CLSAE_DIR}/${EID}/clsae_cam0.pt"   # left
CLSAE_CAM1_CKPT="${CLSAE_DIR}/${EID}/clsae_cam1.pt"   # right

PATCHCONV_L_OUT="${DATA_ROOT}/checkpoints/beast_ae_patchconv_left_${EID:0:8}"
PATCHCONV_R_OUT="${DATA_ROOT}/checkpoints/beast_ae_patchconv_right_${EID:0:8}"
WARMSTART_L_OUT="${DATA_ROOT}/checkpoints/beast_ae_warmstart_left_${EID:0:8}"
WARMSTART_R_OUT="${DATA_ROOT}/checkpoints/beast_ae_warmstart_right_${EID:0:8}"

LATENTS_PATCHCONV_DIR="${DATA_ROOT}/latents_patchconv"
LATENTS_WARMSTART_DIR="${DATA_ROOT}/latents_warmstart"
NEURAL_DIR="${DATA_ROOT}/neural_data"

LOG_DIR="${DATA_ROOT}/logs"
STATE_DIR="${DATA_ROOT}/pipeline_state"
mkdir -p "${LOG_DIR}" "${STATE_DIR}"

CONDA_SH="/home/haoshen/anaconda3/etc/profile.d/conda.sh"
CONDA_ENV="beast"

POLL_SECS=60

############################################################
# HELPERS
############################################################
ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

phase_done() { [[ -f "${STATE_DIR}/$1.done" ]]; }
mark_done() { touch "${STATE_DIR}/$1.done"; }

run_phase() {
    local name="$1"; shift
    if phase_done "${name}"; then
        log "SKIP  ${name} (marker exists at ${STATE_DIR}/${name}.done)"
        return 0
    fi
    log "START ${name}"
    "$@"
    mark_done "${name}"
    log "DONE  ${name}"
}

# activate conda in this shell
# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${REPO}"

############################################################
# SMOKE CHECK — validate inputs before any waiting/training
############################################################
smoke_check() {
    local missing=0
    local required_files=(
        "${REPO}/configs/vit_beast.yaml"
        "${REPO}/configs/vit_beast_ae_patchconv.yaml"
        "${REPO}/configs/vit_beast_ae_warmstart.yaml"
        "${REPO}/scripts/extract_latents.py"
        "${REPO}/scripts/train_cls_ae.py"
        "${REPO}/encoding_decoding_code/test.py"
        "${NEURAL_DIR}/${EID}/${EID}_aligned.npz"
    )
    local required_dirs=(
        "${LEFT_DATA}"
        "${RIGHT_DATA}"
        "${EVAL_FRAMES_DIR}/_iblrig_leftCamera.raw.${EID}/train"
        "${EVAL_FRAMES_DIR}/_iblrig_leftCamera.raw.${EID}/val"
        "${EVAL_FRAMES_DIR}/_iblrig_leftCamera.raw.${EID}/test"
        "${EVAL_FRAMES_DIR}/_iblrig_rightCamera.raw.${EID}/train"
        "${EVAL_FRAMES_DIR}/_iblrig_rightCamera.raw.${EID}/val"
        "${EVAL_FRAMES_DIR}/_iblrig_rightCamera.raw.${EID}/test"
    )

    log "Smoke check: verifying inputs."
    for f in "${required_files[@]}"; do
        if [[ ! -f "${f}" ]]; then log "  MISSING file: ${f}"; missing=1; fi
    done
    for d in "${required_dirs[@]}"; do
        if [[ ! -d "${d}" ]]; then log "  MISSING dir:  ${d}"; missing=1; fi
    done

    # beast CLI must be importable/executable
    if ! command -v beast >/dev/null 2>&1; then
        log "  MISSING: 'beast' CLI not on PATH (conda env '${CONDA_ENV}' not activated?)"
        missing=1
    fi
    # GPU presence
    if ! nvidia-smi -L | grep -q 'GPU 0'; then log "  MISSING: nvidia-smi sees no GPUs"; missing=1; fi

    # At least one stage-1 process should currently be running — otherwise
    # phase_0 will loop forever. If user wants to run post-hoc, they can touch
    # the stage1_wait.done marker before launching.
    if ! phase_done stage1_wait; then
        local any_running
        any_running=$(pgrep -fa 'beast train --config .*vit_beast\.yaml' | grep -E "beast_(left|right)_${EID:0:8}" | wc -l || true)
        if (( any_running == 0 )); then
            log "  WARN: no stage-1 training processes found. Phase 0 would wait forever."
            log "  If stage-1 already finished, run: touch ${STATE_DIR}/stage1_wait.done"
            missing=1
        fi
    fi

    if (( missing != 0 )); then
        log "Smoke check FAILED. Aborting without any side effects."
        exit 1
    fi
    log "Smoke check passed."
}

smoke_check

############################################################
# PHASE 0 — wait for stage-1 L+R to finish
############################################################
wait_for_stage1() {
    log "Waiting for stage-1 BEAST L+R to finish (polling every ${POLL_SECS}s)."
    while true; do
        local l_running r_running
        l_running=0; r_running=0
        pgrep -fa 'beast train --config .*vit_beast\.yaml' | grep -q "beast_left_${EID:0:8}"  && l_running=1 || true
        pgrep -fa 'beast train --config .*vit_beast\.yaml' | grep -q "beast_right_${EID:0:8}" && r_running=1 || true
        log "  stage-1: L=$( ((l_running)) && echo running || echo done ), R=$( ((r_running)) && echo running || echo done )"
        if (( l_running == 0 && r_running == 0 )); then break; fi
        sleep "${POLL_SECS}"
    done

    # sanity-check the checkpoints landed
    [[ -f "${STAGE1_L_CKPT}" ]] || { log "FATAL: missing ${STAGE1_L_CKPT}"; exit 2; }
    [[ -f "${STAGE1_R_CKPT}" ]] || { log "FATAL: missing ${STAGE1_R_CKPT}"; exit 2; }
    log "Stage-1 complete. Checkpoints present."
}

############################################################
# PHASE 1 — CLS extract (1 GPU, fast)
############################################################
phase_cls_extract() {
    CUDA_VISIBLE_DEVICES=0 python scripts/extract_latents.py \
        --eid "${EID}" \
        --eval_frames_dir "${EVAL_FRAMES_DIR}" \
        --left_ckpt_dir   "${STAGE1_L_CKPT_DIR}" \
        --right_ckpt_dir  "${STAGE1_R_CKPT_DIR}" \
        --output_dir      "${CLS_LATENT_DIR}" \
        --mode cls \
        2>&1 | tee -a "${LOG_DIR}/pipeline_cls_extract.log"
}

############################################################
# PHASE 2 — CLS-AE train (1 GPU, fast)
############################################################
phase_clsae_train() {
    CUDA_VISIBLE_DEVICES=0 python scripts/train_cls_ae.py \
        --eid "${EID}" \
        --cls_latent_dir "${CLS_LATENT_DIR}" \
        --output_dir     "${CLSAE_DIR}" \
        --d_latent 100 \
        2>&1 | tee -a "${LOG_DIR}/pipeline_clsae_train.log"
    [[ -f "${CLSAE_CAM0_CKPT}" ]] || { log "FATAL: missing ${CLSAE_CAM0_CKPT}"; exit 3; }
    [[ -f "${CLSAE_CAM1_CKPT}" ]] || { log "FATAL: missing ${CLSAE_CAM1_CKPT}"; exit 3; }
}

############################################################
# PHASE 3 — BEAST-AE patchconv L + R in parallel (GPU0=L, GPU1=R)
############################################################
phase_patchconv_train() {
    local l_log="${LOG_DIR}/pipeline_patchconv_left.log"
    local r_log="${LOG_DIR}/pipeline_patchconv_right.log"

    CUDA_VISIBLE_DEVICES=0 beast train \
        --config configs/vit_beast_ae_patchconv.yaml \
        --data "${LEFT_DATA}" \
        --output "${PATCHCONV_L_OUT}" \
        --overrides "beast_pretrained_ckpt=${STAGE1_L_CKPT}" \
        > "${l_log}" 2>&1 &
    local pid_l=$!

    CUDA_VISIBLE_DEVICES=1 beast train \
        --config configs/vit_beast_ae_patchconv.yaml \
        --data "${RIGHT_DATA}" \
        --output "${PATCHCONV_R_OUT}" \
        --overrides "beast_pretrained_ckpt=${STAGE1_R_CKPT}" \
        > "${r_log}" 2>&1 &
    local pid_r=$!

    log "  patchconv-L pid=${pid_l}  log=${l_log}"
    log "  patchconv-R pid=${pid_r}  log=${r_log}"

    local fail=0
    wait "${pid_l}" || { log "patchconv-L failed"; fail=1; }
    wait "${pid_r}" || { log "patchconv-R failed"; fail=1; }
    (( fail == 0 )) || exit 4
}

############################################################
# PHASE 4 — extract patchconv bottleneck latents
############################################################
phase_patchconv_extract() {
    CUDA_VISIBLE_DEVICES=0 python scripts/extract_latents.py \
        --eid "${EID}" \
        --eval_frames_dir "${EVAL_FRAMES_DIR}" \
        --left_ckpt_dir   "${PATCHCONV_L_OUT}/tb_logs/version_0" \
        --right_ckpt_dir  "${PATCHCONV_R_OUT}/tb_logs/version_0" \
        --output_dir      "${LATENTS_PATCHCONV_DIR}" \
        --mode bottleneck \
        2>&1 | tee -a "${LOG_DIR}/pipeline_patchconv_extract.log"
}

############################################################
# PHASE 5 — evaluate patchconv (encoding + decoding)
############################################################
phase_patchconv_eval() {
    CUDA_VISIBLE_DEVICES=0 python encoding_decoding_code/test.py \
        --eid "${EID}" \
        --neural_input_dir "${NEURAL_DIR}" \
        --latent_input_dir "${LATENTS_PATCHCONV_DIR}" \
        --eval_task encoding \
        2>&1 | tee -a "${LOG_DIR}/pipeline_patchconv_encoding.log"

    CUDA_VISIBLE_DEVICES=0 python encoding_decoding_code/test.py \
        --eid "${EID}" \
        --neural_input_dir "${NEURAL_DIR}" \
        --latent_input_dir "${LATENTS_PATCHCONV_DIR}" \
        --eval_task decoding \
        2>&1 | tee -a "${LOG_DIR}/pipeline_patchconv_decoding.log"
}

############################################################
# PHASE 6 — BEAST-AE warmstart L + R in parallel (GPU0=L, GPU1=R)
############################################################
phase_warmstart_train() {
    local l_log="${LOG_DIR}/pipeline_warmstart_left.log"
    local r_log="${LOG_DIR}/pipeline_warmstart_right.log"

    CUDA_VISIBLE_DEVICES=0 beast train \
        --config configs/vit_beast_ae_warmstart.yaml \
        --data "${LEFT_DATA}" \
        --output "${WARMSTART_L_OUT}" \
        --overrides "beast_pretrained_ckpt=${STAGE1_L_CKPT}" "clsae_init_ckpt=${CLSAE_CAM0_CKPT}" \
        > "${l_log}" 2>&1 &
    local pid_l=$!

    CUDA_VISIBLE_DEVICES=1 beast train \
        --config configs/vit_beast_ae_warmstart.yaml \
        --data "${RIGHT_DATA}" \
        --output "${WARMSTART_R_OUT}" \
        --overrides "beast_pretrained_ckpt=${STAGE1_R_CKPT}" "clsae_init_ckpt=${CLSAE_CAM1_CKPT}" \
        > "${r_log}" 2>&1 &
    local pid_r=$!

    log "  warmstart-L pid=${pid_l}  log=${l_log}"
    log "  warmstart-R pid=${pid_r}  log=${r_log}"

    local fail=0
    wait "${pid_l}" || { log "warmstart-L failed"; fail=1; }
    wait "${pid_r}" || { log "warmstart-R failed"; fail=1; }
    (( fail == 0 )) || exit 5
}

############################################################
# PHASE 7 — extract warmstart bottleneck latents
############################################################
phase_warmstart_extract() {
    CUDA_VISIBLE_DEVICES=0 python scripts/extract_latents.py \
        --eid "${EID}" \
        --eval_frames_dir "${EVAL_FRAMES_DIR}" \
        --left_ckpt_dir   "${WARMSTART_L_OUT}/tb_logs/version_0" \
        --right_ckpt_dir  "${WARMSTART_R_OUT}/tb_logs/version_0" \
        --output_dir      "${LATENTS_WARMSTART_DIR}" \
        --mode bottleneck \
        2>&1 | tee -a "${LOG_DIR}/pipeline_warmstart_extract.log"
}

############################################################
# PHASE 8 — evaluate warmstart (encoding + decoding)
############################################################
phase_warmstart_eval() {
    CUDA_VISIBLE_DEVICES=0 python encoding_decoding_code/test.py \
        --eid "${EID}" \
        --neural_input_dir "${NEURAL_DIR}" \
        --latent_input_dir "${LATENTS_WARMSTART_DIR}" \
        --eval_task encoding \
        2>&1 | tee -a "${LOG_DIR}/pipeline_warmstart_encoding.log"

    CUDA_VISIBLE_DEVICES=0 python encoding_decoding_code/test.py \
        --eid "${EID}" \
        --neural_input_dir "${NEURAL_DIR}" \
        --latent_input_dir "${LATENTS_WARMSTART_DIR}" \
        --eval_task decoding \
        2>&1 | tee -a "${LOG_DIR}/pipeline_warmstart_decoding.log"
}

############################################################
# MAIN
############################################################
log "=========================================="
log "Pipeline start. EID=${EID}"
log "State dir: ${STATE_DIR} (remove a .done file to re-run a phase)"
log "=========================================="

run_phase stage1_wait        wait_for_stage1
run_phase cls_extract        phase_cls_extract
run_phase clsae_train        phase_clsae_train
run_phase patchconv_train    phase_patchconv_train
run_phase patchconv_extract  phase_patchconv_extract
run_phase patchconv_eval     phase_patchconv_eval
run_phase warmstart_train    phase_warmstart_train
run_phase warmstart_extract  phase_warmstart_extract
run_phase warmstart_eval     phase_warmstart_eval

log "Pipeline complete."
