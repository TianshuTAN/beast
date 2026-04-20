#!/bin/bash
#SBATCH -A bezq-delta-cpu 
#SBATCH --job-name="extract"
#SBATCH --output="extract.%j.out"
#SBATCH --partition=cpu
#SBATCH -c 1
#SBATCH --mem 200000
#SBATCH -t 0-03:00:00
#SBATCH --export=ALL

# Load environment
. ~/.bashrc

module load ffmpeg
module load pytorch-conda/2.8
source /sw/rh9.4/user/python/conda-env/pytorch-2.8-cu128/etc/profile.d/conda.sh

input_path=${1}
output_path=${2}
method=${3}
frames_per_video=${4}
timestamp_dir=${5:-}
neural_data_dir=${6:-}

echo "Output will be saved to: $output_path"

# Change to repo root
cd ..

# Activate environment
conda activate beast

extra_args=()
if [ -n "$timestamp_dir" ]; then
  extra_args+=(--timestamp_dir "$timestamp_dir")
fi
if [ -n "$neural_data_dir" ]; then
  extra_args+=(--neural_data_dir "$neural_data_dir")
fi

beast extract --input "$input_path" \
  --output "$output_path" \
  --method "$method" \
  --frames-per-video "$frames_per_video" \
  "${extra_args[@]}"

# Deactivate environment
conda deactivate

# Return to scripts directory
cd scripts
