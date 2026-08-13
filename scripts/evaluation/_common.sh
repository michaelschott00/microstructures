#!/bin/bash

# Parse args.
#   COMPUTE: "local" or "runpod" (where to run)
#   MODE:    "regular" or "dry-run"
# Everything else accumulates in ARGS.
parse_args() {
  COMPUTE="runpod"
  MODE="regular"
  CACHE_DIR=""
  ARGS=()
  EXTRA_ARGS=()
  while [ $# -gt 0 ]; do
    case "$1" in
      --dry-run-local) COMPUTE="local"; MODE="dry-run"; shift ;;
      --dry-run-runpod) COMPUTE="runpod"; MODE="dry-run"; shift ;;
      --cache-dir) CACHE_DIR="$2"; shift 2 ;;
      *) ARGS+=("$1"); shift ;;
    esac
  done
}

cache_key() {
  local out="" p v
  for p in "$@"; do
    v="$p"
    case "$p" in
      */*|*.yaml) v="$(basename "$p")"; v="${v%.*}" ;;
    esac
    if [ -z "$out" ]; then
      out="$v"
    else
      out="$out, $v"
    fi
  done
  printf '%s' "$out"
}

run_if_not_cached() {
  local key="$1"
  shift
  if [ -z "$CACHE_DIR" ]; then
    "$@"
    return $?
  fi
  local script_name="$(basename "$0")"
  script_name="${script_name%.sh}"
  local cache_file="$CACHE_DIR/$script_name/$key"
  if [ -f "$cache_file" ]; then
    echo "$(date) Skipping cached run: $key"
    return 0
  fi
  "$@"
  local status=$?
  if [ $status -eq 0 ]; then
    mkdir -p "$CACHE_DIR/$script_name"
    touch "$cache_file"
  fi
  return $status
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