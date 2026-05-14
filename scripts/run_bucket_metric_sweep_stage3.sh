#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BASELINE_DIR="${BASELINE_DIR:-local_data/buckets/500f_100t_100r_v1}"
REFERENCE_DIR="${REFERENCE_DIR:-local_data/buckets/500f_100t_100r_nut_high_v1}"
SWEEP_DIR="${SWEEP_DIR:-local_data/bucket_sweeps/500f_100t_100r_nut_stage3_v1}"
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

diff_dirs() {
  local dir_a="$1"
  local dir_b="$2"
  local log="$3"
  run_logged "$log" cargo run -p poker-solver-trainer --release -- diff-clusters \
    --dir-a "$dir_a" \
    --dir-b "$dir_b" \
    --sample-boards "$AUDIT_BOARDS"
}

declare -a NAMES=(
  high_cap_1p0
  high_cap_0p5
  high_sqrt_cap_1p0
  high_log1p_cap_2p0
)
declare -a CONFIGS=(
  sample_configurations/blueprint_v2_500f_100t_100r_nut_high_cap_1p0.yaml
  sample_configurations/blueprint_v2_500f_100t_100r_nut_high_cap_0p5.yaml
  sample_configurations/blueprint_v2_500f_100t_100r_nut_high_sqrt_cap_1p0.yaml
  sample_configurations/blueprint_v2_500f_100t_100r_nut_high_log1p_cap_2p0.yaml
)
declare -a OUTPUTS=(
  local_data/buckets/500f_100t_100r_nut_high_cap_1p0_v1
  local_data/buckets/500f_100t_100r_nut_high_cap_0p5_v1
  local_data/buckets/500f_100t_100r_nut_high_sqrt_cap_1p0_v1
  local_data/buckets/500f_100t_100r_nut_high_log1p_cap_2p0_v1
)

if [[ ! -f "$BASELINE_DIR/scorecard.json" ]]; then
  scorecard "$BASELINE_DIR" "$SWEEP_DIR/logs/baseline_diag.log"
fi

if [[ -d "$REFERENCE_DIR" && ! -f "$REFERENCE_DIR/scorecard.json" ]]; then
  scorecard "$REFERENCE_DIR" "$SWEEP_DIR/logs/reference_high_diag.log"
fi

for idx in "${!NAMES[@]}"; do
  name="${NAMES[$idx]}"
  config="${CONFIGS[$idx]}"
  output="${OUTPUTS[$idx]}"

  run_logged "$SWEEP_DIR/logs/${name}_cluster.log" cargo run -p poker-solver-trainer --release -- cluster \
    -c "$config" \
    -o "$output"
  scorecard "$output" "$SWEEP_DIR/logs/${name}_diag.log"
  diff_dirs "$BASELINE_DIR" "$output" "$SWEEP_DIR/logs/${name}_vs_baseline_diff.log"
  if [[ -d "$REFERENCE_DIR" ]]; then
    diff_dirs "$REFERENCE_DIR" "$output" "$SWEEP_DIR/logs/${name}_vs_high_diff.log"
  fi
done

python3 scripts/analyze_bucket_sweep.py \
  --baseline "$BASELINE_DIR" \
  --candidate "first_high=${REFERENCE_DIR}" \
  --candidate "high_cap_1p0=${OUTPUTS[0]}" \
  --candidate "high_cap_0p5=${OUTPUTS[1]}" \
  --candidate "high_sqrt_cap_1p0=${OUTPUTS[2]}" \
  --candidate "high_log1p_cap_2p0=${OUTPUTS[3]}" \
  --output "$SWEEP_DIR/analysis.md"

echo "Sweep complete: $SWEEP_DIR/analysis.md"
