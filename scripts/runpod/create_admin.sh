#!/usr/bin/env bash
set -euo pipefail

set -o allexport
source .env
set +o allexport

runpodctl pod create \
  --name "microstructure-admin" \
  --image "michaelschott00/microstructures-admin:main" \
  --compute-type cpu \
  --cloud-type SECURE \
  --volume-in-gb 0 \
  --container-disk-in-gb 20 \
  --env "{\"JUPYTER_PASSWORD\":\"\",\"PUBLIC_KEY\":\"${SSH_PUBLIC_KEY}\"}" \
  --ports "22/tcp" \
  --network-volume-id "${RUNPOD_VOLUME_ID}" \
  --data-center-ids "EUR-IS-1" \
  --wait
