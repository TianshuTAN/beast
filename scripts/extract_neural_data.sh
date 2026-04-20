#!/bin/bash
#SBATCH -A bezq-delta-cpu 
#SBATCH --job-name="extract"
#SBATCH --output="extract.%j.out"
#SBATCH --partition=cpu
#SBATCH -c 1
#SBATCH --mem 200000
#SBATCH -t 0-00:30:00
#SBATCH --export=ALL

# Load environment
. ~/.bashrc

module load ffmpeg
module load pytorch-conda/2.8
source /sw/rh9.4/user/python/conda-env/pytorch-2.8-cu128/etc/profile.d/conda.sh

eid=${1}
one_cache_path=${2}
video_timestamps=${3}
output_path=${4}
num_trials=${5}
# CPUs assigned to this task
n_workers="${SLURM_CPUS_PER_TASK:-1}"

echo "Output will be saved to: $output_path"

# Change to repo root
cd ..

# Activate environment
conda activate beast

python beast/extract_neural_data.py --eid "$eid" \
  --one_cache_path "$one_cache_path" \
  --video_timestamps "$video_timestamps" \
  --output_path "$output_path" \
  --num_trials "$num_trials" \
  --n_workers "$n_workers"

# Deactivate environment
conda deactivate

# Return to scripts directory
cd scripts
