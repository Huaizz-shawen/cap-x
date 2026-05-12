#!/usr/bin/env bash
set -euo pipefail

# Check a collection run status using trial-latest-attempt semantics.
#
# Usage:
#   scripts/check_collect_run_status.sh <RUN_ID> [REPO_ROOT]
#
# Example:
#   scripts/check_collect_run_status.sh libero_long_10trials_tuneB_20260506_060905

RUN_ID="${1:-}"
REPO_ROOT="${2:-$(cd "$(dirname "$0")/.." && pwd)}"

if [ -z "$RUN_ID" ]; then
  echo "Usage: $0 <RUN_ID> [REPO_ROOT]" >&2
  exit 1
fi

OUT_DIR="$REPO_ROOT/outputs/deepseek-v4-pro/deepseek-v4-pro/$RUN_ID"

if [ ! -d "$OUT_DIR" ]; then
  echo "RUN_NOT_FOUND: $OUT_DIR" >&2
  exit 2
fi

total_trials=0
final_true=0
final_false=0
final_unknown=0
all_true=0
all_false=0
all_summaries=0

classify_summary_status() {
  local summary_path="$1"

  if [ ! -f "$summary_path" ]; then
    printf '%s\n' "UNKNOWN"
    return 0
  fi

  if grep -Eq "Task Completed:[[:space:]]*(True|true|1)" "$summary_path"; then
    printf '%s\n' "True"
    return 0
  fi

  if grep -Eq "Task Completed:[[:space:]]*(False|false|0)" "$summary_path"; then
    printf '%s\n' "False"
    return 0
  fi

  # Conservative fallback:
  # when trial summary exists but lacks explicit Task Completed marker
  # (e.g. abnormal teardown / timeout/partial write), treat it as failed.
  if grep -Eq "(Sandbox failed:|Num Code Blocks:|pre-codegen timeout|TimeoutError|Task timeout)" "$summary_path"; then
    printf '%s\n' "False"
    return 0
  fi

  printf '%s\n' "False"
}

echo "RUN_ID=$RUN_ID"
echo "OUT_DIR=$OUT_DIR"
echo
echo "Per-trial final status (latest attempt):"

while IFS= read -r trial_dir; do
  total_trials=$((total_trials + 1))
  trial_name="$(basename "$trial_dir")"
  latest_attempt="$(find "$trial_dir" -maxdepth 1 -type d -name 'attempt_*' | sort -V | tail -n1 || true)"

  if [ -z "$latest_attempt" ]; then
    echo "  $trial_name final_attempt=NONE status=UNKNOWN"
    final_unknown=$((final_unknown + 1))
    continue
  fi

  summary_path="$latest_attempt/summary.txt"
  status="$(classify_summary_status "$summary_path")"
  case "$status" in
    True) final_true=$((final_true + 1)) ;;
    False) final_false=$((final_false + 1)) ;;
    *) final_unknown=$((final_unknown + 1)) ;;
  esac

  echo "  $trial_name final_attempt=$(basename "$latest_attempt") status=$status"
done < <(find "$OUT_DIR" -maxdepth 1 -type d -name 'trial_*' | sort -V)

all_summaries="$(find "$OUT_DIR" -type f -name 'summary.txt' | wc -l | tr -d ' ')"
all_true="$(grep -R "Task Completed:[[:space:]]*True" "$OUT_DIR"/trial_*/attempt_*/summary.txt 2>/dev/null | wc -l | tr -d ' ')"
all_false="$(grep -R "Task Completed:[[:space:]]*False" "$OUT_DIR"/trial_*/attempt_*/summary.txt 2>/dev/null | wc -l | tr -d ' ')"

echo
echo "Final-by-trial summary:"
echo "  total_trials=$total_trials"
echo "  final_true=$final_true"
echo "  final_false=$final_false"
echo "  final_unknown=$final_unknown"

echo
echo "All-attempt diagnostics:"
echo "  all_summaries=$all_summaries"
echo "  all_true=$all_true"
echo "  all_false=$all_false"

if [ "$all_true" -ne "$final_true" ] || [ "$all_false" -lt "$final_false" ]; then
  echo
  echo "NOTE: all-attempt counts can differ from final-by-trial counts because retries may flip status."
fi
