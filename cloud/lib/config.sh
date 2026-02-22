#!/usr/bin/env bash
# cloud/lib/config.sh — shared configuration and utility functions

set -euo pipefail

# ── Defaults (override via environment or cloud/config.local.sh) ────────────
SOLVER_AWS_REGION="${SOLVER_AWS_REGION:-us-east-1}"
SOLVER_S3_BUCKET="${SOLVER_S3_BUCKET:-poker-solver-models}"
SOLVER_ECR_REPO="${SOLVER_ECR_REPO:-poker-solver-trainer}"
SOLVER_KEY_NAME="${SOLVER_KEY_NAME:-solver-cloud}"
SOLVER_SG_NAME="${SOLVER_SG_NAME:-solver-cloud-sg}"
SOLVER_SSH_KEY="${SOLVER_SSH_KEY:-$HOME/.ssh/solver-cloud.pem}"
SOLVER_TAG_PREFIX="solver"

# Load local overrides if present
CLOUD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$CLOUD_DIR/config.local.sh" ]]; then
  # shellcheck source=/dev/null
  source "$CLOUD_DIR/config.local.sh"
fi

# ── AWS profile helper ──────────────────────────────────────────────────────
AWS_ARGS=()
if [[ -n "${SOLVER_AWS_PROFILE:-}" ]]; then
  AWS_ARGS+=(--profile "$SOLVER_AWS_PROFILE")
fi
AWS_ARGS+=(--region "$SOLVER_AWS_REGION")

aws_cmd() {
  aws "${AWS_ARGS[@]}" "$@"
}

# ── Logging ─────────────────────────────────────────────────────────────────
info()  { echo "[solver-cloud] $*"; }
error() { echo "[solver-cloud] ERROR: $*" >&2; }
die()   { error "$@"; exit 1; }

# ── Tag helpers ─────────────────────────────────────────────────────────────
get_tag() {
  local instance_id="$1" key="$2"
  aws_cmd ec2 describe-tags \
    --filters "Name=resource-id,Values=$instance_id" "Name=key,Values=${SOLVER_TAG_PREFIX}:${key}" \
    --query 'Tags[0].Value' --output text 2>/dev/null || echo ""
}

set_tag() {
  local instance_id="$1" key="$2" value="$3"
  aws_cmd ec2 create-tags \
    --resources "$instance_id" \
    --tags "Key=${SOLVER_TAG_PREFIX}:${key},Value=${value}"
}

get_instance_ip() {
  local instance_id="$1"
  aws_cmd ec2 describe-instances \
    --instance-ids "$instance_id" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
}

get_instance_state() {
  local instance_id="$1"
  aws_cmd ec2 describe-instances \
    --instance-ids "$instance_id" \
    --query 'Reservations[0].Instances[0].State.Name' --output text
}
