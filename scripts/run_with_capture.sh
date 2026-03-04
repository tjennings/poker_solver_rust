#!/bin/bash
# Run the postflop solver with full panic/output capture.
#
# Usage: ./scripts/run_with_capture.sh -c sample_configurations/full.yaml -o ./local_data/full_postflop_spr
#
# All stdout, stderr, and panic output is captured to:
#   logs/run_YYYYMMDD_HHMMSS.log
#
# The TUI is disabled (piped mode) so all solver output is visible.
# For TUI mode, the built-in panic hook now writes to the log file in ./logs/.

set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/run_${TIMESTAMP}.log"
mkdir -p logs

echo "Starting postflop solve at $(date)" | tee "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Args: $*" | tee -a "$LOG_FILE"
echo "---" | tee -a "$LOG_FILE"

RUST_BACKTRACE=1 cargo run -p poker-solver-trainer --release -- solve-postflop "$@" 2>&1 | tee -a "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}

echo "---" | tee -a "$LOG_FILE"
echo "Finished at $(date) with exit code $EXIT_CODE" | tee -a "$LOG_FILE"
exit $EXIT_CODE
