#!/usr/bin/env bash
# cloud/lib/s3.sh — S3 upload/download for model bundles

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

# Upload a local config file to S3 for the job
upload_config() {
  local config_path="$1" instance_id="$2"
  local s3_path="s3://${SOLVER_S3_BUCKET}/jobs/${instance_id}/config/$(basename "$config_path")"
  info "Uploading config to $s3_path"
  aws_cmd s3 cp "$config_path" "$s3_path"
  echo "$s3_path"
}

# Download the finished model bundle from S3
download_bundle() {
  local instance_id="$1" local_dir="$2"
  local s3_path="s3://${SOLVER_S3_BUCKET}/jobs/${instance_id}/bundle/"

  info "Syncing model from $s3_path to $local_dir"
  mkdir -p "$local_dir"
  aws_cmd s3 sync "$s3_path" "$local_dir"
  info "Download complete: $local_dir"
}

# Upload a config file before launch (used by launch command)
upload_config_for_launch() {
  local config_path="$1"
  local s3_path="s3://${SOLVER_S3_BUCKET}/configs/$(basename "$config_path")"
  aws_cmd s3 cp "$config_path" "$s3_path"
  echo "$s3_path"
}
