#!/bin/bash
#SBATCH --account=bezq-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --job-name="clsae"
#SBATCH --output="clsae.%j.out"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --time 0-01:00:00
#SBATCH --export=ALL

eid=${1}
cls_latent_dir=${2}
output_dir=${3}
d_latent=${4:-100}

# Load environment
. ~/.bashrc

module load ffmpeg
module load pytorch-conda/2.8
source /sw/rh9.4/user/python/conda-env/pytorch-2.8-cu128/etc/profile.d/conda.sh

echo "EID: $eid"
echo "CLS latent dir: $cls_latent_dir"
echo "Output dir: $output_dir"
echo "d_latent: $d_latent"

# Change to repo root
cd ..

# Activate environment
conda activate beast

python scripts/train_cls_ae.py \
  --eid "$eid" \
  --cls_latent_dir "$cls_latent_dir" \
  --output_dir "$output_dir" \
  --d_latent "$d_latent"

# Deactivate environment
conda deactivate

# Return to scripts directory
cd scripts
