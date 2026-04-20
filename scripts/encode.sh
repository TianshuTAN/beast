#!/bin/bash
#SBATCH --account=bezq-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --job-name="encode"
#SBATCH --output="encode.%j.out"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64000
#SBATCH --time 0-04:00:00
#SBATCH --export=ALL

eid=${1}
neural_input_dir=${2}
latent_input_dir=${3}
eval_task=${4:-encoding}

# Load environment
. ~/.bashrc

module load ffmpeg
module load pytorch-conda/2.8
source /sw/rh9.4/user/python/conda-env/pytorch-2.8-cu128/etc/profile.d/conda.sh

echo "EID: $eid"
echo "Neural input: $neural_input_dir"
echo "Latent input: $latent_input_dir"
echo "Eval task: $eval_task"

# Change to repo root
cd ..

# Activate environment
conda activate beast

python encoding_decoding_code/test.py \
  --eid "$eid" \
  --neural_input_dir "$neural_input_dir" \
  --latent_input_dir "$latent_input_dir" \
  --eval_task "$eval_task"

# Deactivate environment
conda deactivate

# Return to scripts directory
cd scripts
