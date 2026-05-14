#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BASELINE_DIR="${BASELINE_DIR:-local_data/buckets/500f_100t_100r_v1}"
SWEEP_DIR="${SWEEP_DIR:-local_data/bucket_sweeps/500f_100t_100r_nut_v1}"
AUDIT_BOARDS="${AUDIT_BOARDS:-200}"
AUDIT_TOP="${AUDIT_TOP:-15}"

mkdir -p "$SWEEP_DIR/logs"

run_logged() {
  local log_path="$1"
  shift
  echo "==> $*" | tee "$log_path"
  "$@" 2>&1 | tee -a "$log_path"
}

scorecard() {
  local dir="$1"
  local log="$2"
  run_logged "$log" cargo run -p poker-solver-trainer --release -- diag-clusters \
    -d "$dir" \
    --hand-class-audit \
    --hand-class-audit-boards "$AUDIT_BOARDS" \
    --hand-class-audit-top "$AUDIT_TOP" \
    --scorecard-json "$dir/scorecard.json"
}

diff_against_baseline() {
  local name="$1"
  local dir="$2"
  run_logged "$SWEEP_DIR/logs/${name}_diff.log" cargo run -p poker-solver-trainer --release -- diff-clusters \
    --dir-a "$BASELINE_DIR" \
    --dir-b "$dir" \
    --sample-boards "$AUDIT_BOARDS"
}

declare -a NAMES=(low med high river_heavy)
declare -a CONFIGS=(
  sample_configurations/blueprint_v2_500f_100t_100r_nut_low.yaml
  sample_configurations/blueprint_v2_500f_100t_100r_nut_med.yaml
  sample_configurations/blueprint_v2_500f_100t_100r_nut_high.yaml
  sample_configurations/blueprint_v2_500f_100t_100r_nut_river_heavy.yaml
)
declare -a OUTPUTS=(
  local_data/buckets/500f_100t_100r_nut_low_v1
  local_data/buckets/500f_100t_100r_nut_med_v1
  local_data/buckets/500f_100t_100r_nut_high_v1
  local_data/buckets/500f_100t_100r_nut_river_heavy_v1
)

if [[ ! -f "$BASELINE_DIR/scorecard.json" ]]; then
  scorecard "$BASELINE_DIR" "$SWEEP_DIR/logs/baseline_diag.log"
fi

for idx in "${!NAMES[@]}"; do
  name="${NAMES[$idx]}"
  config="${CONFIGS[$idx]}"
  output="${OUTPUTS[$idx]}"

  run_logged "$SWEEP_DIR/logs/${name}_cluster.log" cargo run -p poker-solver-trainer --release -- cluster \
    -c "$config" \
    -o "$output"
  scorecard "$output" "$SWEEP_DIR/logs/${name}_diag.log"
  diff_against_baseline "$name" "$output"
done

python3 scripts/analyze_bucket_sweep.py \
  --baseline "$BASELINE_DIR" \
  --candidate "low=${OUTPUTS[0]}" \
  --candidate "med=${OUTPUTS[1]}" \
  --candidate "high=${OUTPUTS[2]}" \
  --candidate "river_heavy=${OUTPUTS[3]}" \
  --output "$SWEEP_DIR/analysis.md"

echo "Sweep complete: $SWEEP_DIR/analysis.md"
