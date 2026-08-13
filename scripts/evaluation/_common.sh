#!/bin/bash

# Parse args.
#   COMPUTE: "local" or "runpod" (where to run)
#   MODE:    "regular" or "dry-run"
# Everything else accumulates in ARGS.
parse_args() {
  COMPUTE="runpod"
  MODE="regular"
  ARGS=()
  EXTRA_ARGS=()
  for arg in "$@"; do
    case "$arg" in
      --dry-run-local) COMPUTE="local"; MODE="dry-run" ;;
      --dry-run-runpod) COMPUTE="runpod"; MODE="dry-run" ;;
      *) ARGS+=("$arg") ;;
    esac
  done
}

# Start base image services (Jupyter/SSH) in background, or set up a local dry run
setup_env() {
  if [ "$MODE" = "dry-run" ]; then
    EXTRA_ARGS=(--trainer.fast_dev_run=True --data.init_args.batch_size=3)
    export MLDB_DATA_ROOT=$(mktemp -d /tmp/mldb.XXXXXX)
  fi
  if [ "$COMPUTE" = "runpod" ]; then
    /start.sh &

    # Wait for services to start
    sleep 2
  fi
}

# Stop pod
teardown() {
  if [ "$COMPUTE" = "runpod" ]; then
    runpodctl stop pod $RUNPOD_POD_ID
  fi
}