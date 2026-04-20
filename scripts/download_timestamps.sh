#!/bin/bash
#SBATCH --account=bezq-delta-cpu
#SBATCH --partition=cpu
#SBATCH --job-name="dl_ts"
#SBATCH --output="dl_ts.%j.out"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000
#SBATCH --time 0-00:30:00
#SBATCH --export=ALL

eid=${1}
one_cache_path=${2}
output_path=${3}

. ~/.bashrc
module load pytorch-conda/2.8
source /sw/rh9.4/user/python/conda-env/pytorch-2.8-cu128/etc/profile.d/conda.sh

echo "EID: $eid"
echo "ONE cache: $one_cache_path"
echo "Output: $output_path"

cd ..
conda activate beast

python scripts/download_timestamps.py \
  --eid "$eid" \
  --one_cache_path "$one_cache_path" \
  --output_path "$output_path"

conda deactivate
cd scripts
