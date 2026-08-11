#!/usr/bin/env bash
set -euo pipefail

runpodctl network-volume create \
  --name "microstructure-data-volume" \
  --size 25 \
  --data-center-id "EUR-IS-1" \
  --wait
