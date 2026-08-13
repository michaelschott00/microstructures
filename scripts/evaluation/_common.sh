#!/bin/bash

# Parse args: --dry-run sets DRY_RUN, everything else accumulates in ARGS
parse_args() {
  DRY_RUN=0
  ARGS=()
  for arg in "$@"; do
    if [ "$arg" = "--dry-run" ]; then
      DRY_RUN=1
    else
      ARGS+=("$arg")
    fi
  done
}

# Start base image services (Jupyter/SSH) in background, or set up a dry run
setup_env() {
  if [ "$DRY_RUN" = "0" ]; then
    /start.sh &

    # Wait for services to start
    sleep 2
  else
    export MLDB_DATA_ROOT=$(mktemp -d /tmp/mldb.XXXXXX)
    EXTRA_ARGS=(--trainer.fast_dev_run=True --data.init_args.batch_size=3)
  fi
}

# Stop pod
teardown() {
  if [ "$DRY_RUN" = "0" ]; then
    runpodctl stop pod $RUNPOD_POD_ID
  fi
}