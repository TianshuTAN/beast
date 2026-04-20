#!/bin/bash
#SBATCH --account=bezq-delta-cpu
#SBATCH --partition=cpu
#SBATCH --job-name="dl_video"
#SBATCH --output="dl_video.%j.out"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16000
#SBATCH --time 0-02:00:00
#SBATCH --export=ALL

eid=${1}
one_cache_path=${2}
output_path=${3}

# Load environment
. ~/.bashrc

module load pytorch-conda/2.8
source /sw/rh9.4/user/python/conda-env/pytorch-2.8-cu128/etc/profile.d/conda.sh

echo "EID: $eid"
echo "ONE cache: $one_cache_path"
echo "Output: $output_path"

cd ..
conda activate beast

python scripts/download_videos.py \
  --eid "$eid" \
  --one_cache_path "$one_cache_path" \
  --output_path "$output_path"

# Create per-camera subdirs with symlinks so extract.sh can target one camera at a time.
# extract.py globs *.mp4 in the input dir, so we need per-camera dirs.
left_mp4="$output_path/_iblrig_leftCamera.raw.${eid}.mp4"
right_mp4="$output_path/_iblrig_rightCamera.raw.${eid}.mp4"
mkdir -p "$output_path/left_${eid}" "$output_path/right_${eid}"
ln -sf "$left_mp4" "$output_path/left_${eid}/_iblrig_leftCamera.raw.${eid}.mp4"
ln -sf "$right_mp4" "$output_path/right_${eid}/_iblrig_rightCamera.raw.${eid}.mp4"
echo "Created symlinks:"
ls -la "$output_path/left_${eid}" "$output_path/right_${eid}"

conda deactivate
cd scripts
