#!/usr/bin/env bash
# cloud/lib/docker.sh — build Docker image and push to ECR

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Compute a content hash for cache-busting: hash of Cargo.lock + all Rust source
content_hash() {
  (cd "$REPO_ROOT" && find crates/ Cargo.toml Cargo.lock \( -name '*.rs' -o -name 'Cargo.toml' -o -name 'Cargo.lock' \) | sort | xargs sha256sum | sha256sum | cut -c1-12)
}

# Get the ECR registry URI
ecr_registry() {
  local account_id
  account_id="$(aws_cmd sts get-caller-identity --query Account --output text)"
  echo "${account_id}.dkr.ecr.${SOLVER_AWS_REGION}.amazonaws.com"
}

ecr_image_uri() {
  local tag="${1:-latest}"
  echo "$(ecr_registry)/${SOLVER_ECR_REPO}:${tag}"
}

# Login to ECR
ecr_login() {
  local registry
  registry="$(ecr_registry)"
  info "Logging into ECR: $registry"
  aws_cmd ecr get-login-password | docker login --username AWS --password-stdin "$registry"
}

# Build and push. Returns the full image URI.
docker_build_and_push() {
  local tag
  tag="$(content_hash)"
  local image_uri
  image_uri="$(ecr_image_uri "$tag")"

  # Check if image already exists in ECR
  local existing
  existing="$(aws_cmd ecr describe-images \
    --repository-name "$SOLVER_ECR_REPO" \
    --image-ids "imageTag=$tag" \
    --query 'imageDetails[0].imageTags[0]' --output text 2>/dev/null || echo "None")"

  if [[ "$existing" != "None" ]]; then
    info "Image already up-to-date in ECR: $image_uri"
    echo "$image_uri"
    return 0
  fi

  info "Building Docker image (tag: $tag)..."
  docker build -f "$REPO_ROOT/cloud/Dockerfile" -t "$image_uri" "$REPO_ROOT"

  ecr_login
  info "Pushing to ECR..."
  docker push "$image_uri"

  # Also tag as latest
  docker tag "$image_uri" "$(ecr_image_uri latest)"
  docker push "$(ecr_image_uri latest)"

  info "Pushed: $image_uri"
  echo "$image_uri"
}
