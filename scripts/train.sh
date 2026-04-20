#!/bin/bash
#SBATCH --account=bezq-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --job-name="train"
#SBATCH --output="train.%j.out"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100000
#SBATCH --time 1-00:00:00
#SBATCH --export=ALL

data_path=${1}
checkpoint_path=${2}
config_path=${3}
overrides=${4:-}  # optional: space-separated KEY=VALUE overrides

# Load environment
. ~/.bashrc

module load ffmpeg
module load pytorch-conda/2.8
source /sw/rh9.4/user/python/conda-env/pytorch-2.8-cu128/etc/profile.d/conda.sh

echo "Output will be saved to: $checkpoint_path"

# Change to repo root
cd ..

# Activate environment
conda activate beast

if [ -n "$overrides" ]; then
  echo "Overrides: $overrides"
  beast train --config "$config_path" \
    --data "$data_path" \
    --output "$checkpoint_path" \
    --overrides $overrides
else
  beast train --config "$config_path" \
    --data "$data_path" \
    --output "$checkpoint_path"
fi

# Deactivate environment
conda deactivate

# Return to scripts directory
cd scripts
