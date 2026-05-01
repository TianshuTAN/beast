#!/bin/bash

ONE_CACHE_PATH=${1}
VIDEO_TIMESTAMPS=${2}
OUTPUT_PATH=${3}

# List EIDs here and run this script with no arguments
EIDS=(
  "4b00df29-3769-43be-bb40-128b1cba6d35"
  "72cb5550-43b4-4ef0-add5-e4adfdfb5e02"
  "781b35fd-e1f0-4d14-b2bb-95b7263082bb"
  "f312aaec-3b6f-44b3-86b4-3a0c119c0438"
  "ecb5520d-1358-434c-95ec-93687ecd1396"
)

for eid in "${EIDS[@]}"; do
  echo "Submitting job for eid=${eid}"
  sbatch extract_neural_data.sh "$eid" "$ONE_CACHE_PATH" "$VIDEO_TIMESTAMPS" "$OUTPUT_PATH"
done
