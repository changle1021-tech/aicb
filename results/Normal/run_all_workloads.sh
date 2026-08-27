#!/bin/sh

# Run every SimAI inference workload in this Normal scenario.
# Usage: ./results/Normal/run_all_workloads.sh [TOPOLOGY_FILE]

set -u

SCENARIO_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCENARIO_DIR/../.." && pwd)
RUNNER=${SIMAI_RUNNER:-"$REPO_ROOT/results/run_simai_inference_with_topology.sh"}
TOPOLOGY=${1:-"$SCENARIO_DIR/DCN+SingleToR_8g_4gps_400Gbps_H100"}

if [ "$#" -gt 1 ]; then
  echo "Usage: $0 [TOPOLOGY_FILE]" >&2
  exit 2
fi

case $TOPOLOGY in
  /*) ;;
  *) TOPOLOGY="$REPO_ROOT/$TOPOLOGY" ;;
esac

if [ ! -f "$RUNNER" ]; then
  echo "SimAI runner not found: $RUNNER" >&2
  echo "Set SIMAI_RUNNER to override its location." >&2
  exit 2
fi

if [ ! -f "$TOPOLOGY" ]; then
  echo "Topology file not found: $TOPOLOGY" >&2
  exit 2
fi

set -- "$SCENARIO_DIR"/*.txt
if [ ! -e "$1" ]; then
  echo "No workload .txt files found in: $SCENARIO_DIR" >&2
  exit 2
fi

total=$#
passed=0
failed=0
index=0

cd "$REPO_ROOT" || exit 2

for workload do
  index=$((index + 1))
  echo "[$index/$total] Running: $(basename -- "$workload")"
  if "$RUNNER" -n "$TOPOLOGY" -w "$workload"; then
    passed=$((passed + 1))
  else
    status=$?
    failed=$((failed + 1))
    echo "[$index/$total] FAILED (exit $status): $(basename -- "$workload")" >&2
  fi
done

echo "Completed: total=$total passed=$passed failed=$failed"

if [ "$failed" -ne 0 ]; then
  exit 1
fi
