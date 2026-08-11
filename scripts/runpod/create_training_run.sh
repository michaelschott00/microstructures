#!/usr/bin/env bash
set -euo pipefail

set -o allexport
source .env
set +o allexport

runpodctl pod create \
  --name "microstructure-classification_1-training" \
  --image "michaelschott00/microstructures:main" \
  --compute-type gpu \
  --cloud-type SECURE \
  --volume-in-gb 10 \
  --container-disk-in-gb 10 \
  --env "{\"JUPYTER_PASSWORD\":\"\",\"PUBLIC_KEY\":\"${SSH_PUBLIC_KEY}\"}" \
  --ports "22/tcp" \
  --network-volume-id "${RUNPOD_VOLUME_ID}" \
  --data-center-ids "EUR-IS-1" \
  --gpu-id "NVIDIA RTX 4000 Ada Generation" \
  --gpu-count 1 \
  --docker-args "$@" \
  --wait
