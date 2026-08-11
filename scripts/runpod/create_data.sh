#!/usr/bin/env bash
set -euo pipefail

set -o allexport
source .env
set +o allexport

runpodctl network-volume create \
  --name "microstructure-data-volume" \
  --size 25 \
  --data-center-id "EUR-IS-1" \
  --wait
