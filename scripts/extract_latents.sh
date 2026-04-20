#!/bin/bash
#SBATCH --account=bezq-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --job-name="latents"
#SBATCH --output="latents.%j.out"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000
#SBATCH --time 0-01:00:00
#SBATCH --export=ALL

eid=${1}
eval_frames_dir=${2}
left_ckpt_dir=${3}
right_ckpt_dir=${4}
output_dir=${5}
mode=${6:-bottleneck}  # bottleneck | cls

# Load environment
. ~/.bashrc

module load ffmpeg
module load pytorch-conda/2.8
source /sw/rh9.4/user/python/conda-env/pytorch-2.8-cu128/etc/profile.d/conda.sh

echo "EID: $eid"
echo "Eval frames dir: $eval_frames_dir"
echo "Left ckpt: $left_ckpt_dir"
echo "Right ckpt: $right_ckpt_dir"
echo "Output dir: $output_dir"

# Change to repo root
cd ..

# Activate environment
conda activate beast

python scripts/extract_latents.py \
  --eid "$eid" \
  --eval_frames_dir "$eval_frames_dir" \
  --left_ckpt_dir "$left_ckpt_dir" \
  --right_ckpt_dir "$right_ckpt_dir" \
  --output_dir "$output_dir" \
  --mode "$mode"

# Deactivate environment
conda deactivate

# Return to scripts directory
cd scripts
