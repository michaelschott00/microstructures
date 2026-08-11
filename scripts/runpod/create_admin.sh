#!/usr/bin/env bash
set -euo pipefail

runpodctl pod create \
  --name "microstructure-admin" \
  --image "runpod/base:1.0.2-ubuntu2404" \
  --compute-type cpu \
  --cloud-type SECURE \
  --volume-in-gb 10 \
  --container-disk-in-gb 10 \
  --env "{\"JUPYTER_PASSWORD\":\"\",\"PUBLIC_KEY\":\"${SSH_PUBLIC_KEY}\"}" \
  --ports "22/tcp" \
  --network-volume-id "${RUNPOD_VOLUME_ID}" \
  --data-center-ids "EUR-IS-1" \
  --wait
