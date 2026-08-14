#!/usr/bin/env bash
set -euo pipefail

set -o allexport
source .env
set +o allexport

gpu_ids=("NVIDIA RTX 2000 Ada Generation" "NVIDIA RTX 4000 Ada Generation" "NVIDIA RTX A5000")

for gpu_id in "${gpu_ids[@]}"; do
  if runpodctl pod create \
    --name "microstructure-training" \
    --image "michaelschott00/microstructures:main" \
    --compute-type gpu \
    --cloud-type SECURE \
    --volume-in-gb 0 \
    --container-disk-in-gb 25 \
    --env "{\"JUPYTER_PASSWORD\":\"\",\"PUBLIC_KEY\":\"${SSH_PUBLIC_KEY}\"}" \
    --ports "22/tcp" \
    --network-volume-id "${RUNPOD_VOLUME_ID}" \
    --data-center-ids "EUR-IS-1" \
    --gpu-id "$gpu_id" \
    --gpu-count 1 \
    --docker-args "$@" \
    --wait; then
    exit 0
  fi
done

exit 1