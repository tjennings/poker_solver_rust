# solver-cloud Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Bash CLI scripts to launch poker solver training on AWS EC2 with Docker, S3 model transfer, and SSH-based monitoring.

**Architecture:** Shell scripts in `cloud/` wrap AWS CLI, Docker, and SSH. EC2 instances run a Docker container with the solver, training inside tmux. On completion, the bundle uploads to S3 and the instance self-terminates. All job state lives in EC2 tags.

**Tech Stack:** Bash, AWS CLI v2, Docker, tmux, SSH, S3

---

### Task 1: Config and Shared Utilities

**Files:**
- Create: `cloud/lib/config.sh`

**Step 1: Create the config file**

```bash
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
```

**Step 2: Add cloud/config.local.sh to .gitignore**

Append to `.gitignore`:
```
# Cloud local config
cloud/config.local.sh
```

**Step 3: Commit**

```bash
git add cloud/lib/config.sh .gitignore
git commit -m "feat(cloud): add config.sh with AWS defaults and tag helpers"
```

---

### Task 2: Dockerfile

**Files:**
- Create: `cloud/Dockerfile`
- Create: `cloud/.dockerignore`

**Step 1: Create the Dockerfile**

```dockerfile
# cloud/Dockerfile — multi-stage build for poker-solver-trainer
#
# Build:  docker build -f cloud/Dockerfile -t poker-solver-trainer .
# Run:    docker run --rm poker-solver-trainer train -c /config/training.yaml

# ── Stage 1: Build ──────────────────────────────────────────────────────────
FROM rust:1.85-bookworm AS builder

WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/

# Build only the trainer binary in release mode
RUN cargo build -p poker-solver-trainer --release

# ── Stage 2: Runtime ────────────────────────────────────────────────────────
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/target/release/poker-solver-trainer /usr/local/bin/

ENTRYPOINT ["poker-solver-trainer"]
```

**Step 2: Create .dockerignore**

```
# cloud/.dockerignore
target/
.git/
node_modules/
dist/
local_data/
.idea/
.vscode/
frontend/
cloud/
.beads/
.claude/
.dolt/
```

Note: the `.dockerignore` goes in `cloud/` but since we build with `-f cloud/Dockerfile` from the repo root, Docker uses the repo root's `.dockerignore`. We'll place it at the repo root instead.

Actually — Docker uses `.dockerignore` from the build context root. Since our build context is the repo root, create it there.

**Revised Step 2: Create `.dockerignore` at repo root**

```
target/
.git/
node_modules/
dist/
local_data/
.idea/
.vscode/
frontend/
cloud/
.beads/
.claude/
.dolt/
*.md
```

**Step 3: Test the Docker build locally**

```bash
docker build -f cloud/Dockerfile -t poker-solver-trainer .
docker run --rm poker-solver-trainer --help
```

Expected: prints the trainer's help text.

**Step 4: Commit**

```bash
git add cloud/Dockerfile .dockerignore
git commit -m "feat(cloud): add multi-stage Dockerfile for trainer binary"
```

---

### Task 3: Docker Build & ECR Push Script

**Files:**
- Create: `cloud/lib/docker.sh`

**Step 1: Create docker.sh**

```bash
#!/usr/bin/env bash
# cloud/lib/docker.sh — build Docker image and push to ECR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Compute a content hash for cache-busting: hash of Cargo.lock + all Rust source
content_hash() {
  (cd "$REPO_ROOT" && find crates/ Cargo.toml Cargo.lock -name '*.rs' -o -name 'Cargo.toml' -o -name 'Cargo.lock' | sort | xargs sha256sum | sha256sum | cut -c1-12)
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
```

**Step 2: Commit**

```bash
git add cloud/lib/docker.sh
git commit -m "feat(cloud): add docker.sh for ECR build and push with content-hash caching"
```

---

### Task 4: EC2 Launch & Management Script

**Files:**
- Create: `cloud/lib/ec2.sh`

**Step 1: Create ec2.sh**

```bash
#!/usr/bin/env bash
# cloud/lib/ec2.sh — EC2 instance launch, list, terminate

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

# Get the security group ID by name
get_sg_id() {
  aws_cmd ec2 describe-security-groups \
    --group-names "$SOLVER_SG_NAME" \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo ""
}

# Get the latest Amazon Linux 2023 AMI
get_ami() {
  aws_cmd ec2 describe-images \
    --owners amazon \
    --filters \
      "Name=name,Values=al2023-ami-2023.*-x86_64" \
      "Name=state,Values=available" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' --output text
}

# Launch an EC2 instance for training.
# Args: instance_type image_uri config_s3_path local_output_dir [--on-demand]
launch_instance() {
  local instance_type="$1"
  local image_uri="$2"
  local config_s3_path="$3"
  local local_output="$4"
  local spot="${5:-spot}"

  local ami sg_id user_data s3_job_prefix

  ami="$(get_ami)"
  sg_id="$(get_sg_id)"
  [[ -z "$sg_id" ]] && die "Security group '$SOLVER_SG_NAME' not found. Run 'solver-cloud setup' first."

  info "AMI: $ami"
  info "Instance type: $instance_type"
  info "Image: $image_uri"

  # Generate user-data from template
  local template="$CLOUD_DIR/user-data.sh.tpl"
  [[ -f "$template" ]] || die "user-data template not found: $template"

  user_data="$(
    SOLVER_IMAGE_URI="$image_uri" \
    SOLVER_CONFIG_S3="$config_s3_path" \
    SOLVER_S3_BUCKET="$SOLVER_S3_BUCKET" \
    SOLVER_AWS_REGION="$SOLVER_AWS_REGION" \
    SOLVER_TAG_PREFIX="$SOLVER_TAG_PREFIX" \
    envsubst < "$template"
  )"

  # Build run-instances args
  local run_args=(
    ec2 run-instances
    --image-id "$ami"
    --instance-type "$instance_type"
    --key-name "$SOLVER_KEY_NAME"
    --security-group-ids "$sg_id"
    --iam-instance-profile "Name=solver-cloud-instance"
    --user-data "$user_data"
    --tag-specifications "ResourceType=instance,Tags=[
      {Key=Name,Value=solver-training},
      {Key=${SOLVER_TAG_PREFIX}:status,Value=starting},
      {Key=${SOLVER_TAG_PREFIX}:config,Value=${config_s3_path}},
      {Key=${SOLVER_TAG_PREFIX}:local-output,Value=${local_output}},
      {Key=${SOLVER_TAG_PREFIX}:image,Value=${image_uri}}
    ]"
    --query 'Instances[0].InstanceId' --output text
  )

  if [[ "$spot" == "spot" ]]; then
    run_args+=(--instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time}')
  fi

  local instance_id
  instance_id="$(aws_cmd "${run_args[@]}")"

  # Set the S3 job path tag (needs instance ID)
  s3_job_prefix="s3://${SOLVER_S3_BUCKET}/jobs/${instance_id}"
  set_tag "$instance_id" "s3-path" "$s3_job_prefix"

  info "Launched: $instance_id"
  info "S3 path: $s3_job_prefix"
  echo "$instance_id"
}

# List all solver instances (running or recently stopped)
list_instances() {
  aws_cmd ec2 describe-instances \
    --filters "Name=tag-key,Values=${SOLVER_TAG_PREFIX}:status" \
              "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query 'Reservations[].Instances[].{
      ID:InstanceId,
      Type:InstanceType,
      State:State.Name,
      IP:PublicIpAddress,
      Launch:LaunchTime,
      Status:Tags[?Key==`'"${SOLVER_TAG_PREFIX}"':status`]|[0].Value
    }' --output table
}

# Terminate an instance
terminate_instance() {
  local instance_id="$1"
  info "Terminating $instance_id..."
  aws_cmd ec2 terminate-instances --instance-ids "$instance_id" --output text
}
```

**Step 2: Commit**

```bash
git add cloud/lib/ec2.sh
git commit -m "feat(cloud): add ec2.sh for instance launch, list, and terminate"
```

---

### Task 5: S3 Helpers

**Files:**
- Create: `cloud/lib/s3.sh`

**Step 1: Create s3.sh**

```bash
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
```

**Step 2: Commit**

```bash
git add cloud/lib/s3.sh
git commit -m "feat(cloud): add s3.sh for config upload and model download"
```

---

### Task 6: EC2 User-Data Template

**Files:**
- Create: `cloud/user-data.sh.tpl`

**Step 1: Create the user-data template**

This is the bootstrap script that runs on the EC2 instance at launch. It pulls the Docker image, downloads the config, runs training inside tmux, uploads the result to S3, and self-terminates.

```bash
#!/bin/bash
# user-data.sh — EC2 bootstrap for solver training
# Substituted at launch time by envsubst.

set -euo pipefail
exec > /var/log/solver-cloud.log 2>&1

INSTANCE_ID="$(ec2-metadata -i | awk '{print $2}')"
REGION="${SOLVER_AWS_REGION}"
IMAGE_URI="${SOLVER_IMAGE_URI}"
CONFIG_S3="${SOLVER_CONFIG_S3}"
S3_BUCKET="${SOLVER_S3_BUCKET}"
TAG_PREFIX="${SOLVER_TAG_PREFIX}"
S3_JOB="s3://${S3_BUCKET}/jobs/${INSTANCE_ID}"

tag() { aws ec2 create-tags --region "$REGION" --resources "$INSTANCE_ID" --tags "Key=${TAG_PREFIX}:status,Value=$1"; }

tag "initializing"

# ── Install dependencies ────────────────────────────────────────────────────
yum install -y docker tmux aws-cli
systemctl start docker
systemctl enable docker

# ── Pull Docker image ───────────────────────────────────────────────────────
tag "pulling-image"
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$(echo "$IMAGE_URI" | cut -d/ -f1)"
docker pull "$IMAGE_URI"

# ── Download config ─────────────────────────────────────────────────────────
mkdir -p /opt/solver/config /opt/solver/output
aws s3 cp "$CONFIG_S3" /opt/solver/config/training.yaml --region "$REGION"

# ── Run training in tmux ────────────────────────────────────────────────────
tag "training"
tmux new-session -d -s training "
  docker run --rm \
    -v /opt/solver/config:/config \
    -v /opt/solver/output:/output \
    $IMAGE_URI \
    train -c /config/training.yaml --output-dir /output/bundle \
  && echo '=== TRAINING COMPLETE ===' \
  || echo '=== TRAINING FAILED ==='
"

# ── Wait for training to finish ─────────────────────────────────────────────
while tmux has-session -t training 2>/dev/null; do
  sleep 30
done

# ── Check result and upload ─────────────────────────────────────────────────
if [[ -d /opt/solver/output/bundle ]]; then
  tag "uploading"
  aws s3 sync /opt/solver/output/bundle/ "${S3_JOB}/bundle/" --region "$REGION"
  tag "complete"
else
  tag "failed"
fi

# ── Self-terminate ──────────────────────────────────────────────────────────
sleep 10
aws ec2 terminate-instances --region "$REGION" --instance-ids "$INSTANCE_ID"
```

**Step 2: Commit**

```bash
git add cloud/user-data.sh.tpl
git commit -m "feat(cloud): add user-data template for EC2 training lifecycle"
```

---

### Task 7: Setup Command

**Files:**
- Create: `cloud/lib/setup.sh`

**Step 1: Create setup.sh**

```bash
#!/usr/bin/env bash
# cloud/lib/setup.sh — one-time AWS resource provisioning

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

run_setup() {
  info "Setting up AWS resources for solver-cloud..."

  # ── S3 bucket ───────────────────────────────────────────────────────────
  if aws_cmd s3api head-bucket --bucket "$SOLVER_S3_BUCKET" 2>/dev/null; then
    info "S3 bucket already exists: $SOLVER_S3_BUCKET"
  else
    info "Creating S3 bucket: $SOLVER_S3_BUCKET"
    if [[ "$SOLVER_AWS_REGION" == "us-east-1" ]]; then
      aws_cmd s3api create-bucket --bucket "$SOLVER_S3_BUCKET"
    else
      aws_cmd s3api create-bucket --bucket "$SOLVER_S3_BUCKET" \
        --create-bucket-configuration "LocationConstraint=$SOLVER_AWS_REGION"
    fi
  fi

  # ── ECR repository ─────────────────────────────────────────────────────
  if aws_cmd ecr describe-repositories --repository-names "$SOLVER_ECR_REPO" >/dev/null 2>&1; then
    info "ECR repo already exists: $SOLVER_ECR_REPO"
  else
    info "Creating ECR repo: $SOLVER_ECR_REPO"
    aws_cmd ecr create-repository --repository-name "$SOLVER_ECR_REPO"
  fi

  # ── Key pair ────────────────────────────────────────────────────────────
  if aws_cmd ec2 describe-key-pairs --key-names "$SOLVER_KEY_NAME" >/dev/null 2>&1; then
    info "Key pair already exists: $SOLVER_KEY_NAME"
  else
    info "Creating key pair: $SOLVER_KEY_NAME -> $SOLVER_SSH_KEY"
    mkdir -p "$(dirname "$SOLVER_SSH_KEY")"
    aws_cmd ec2 create-key-pair --key-name "$SOLVER_KEY_NAME" \
      --query 'KeyMaterial' --output text > "$SOLVER_SSH_KEY"
    chmod 600 "$SOLVER_SSH_KEY"
  fi

  # ── Security group ─────────────────────────────────────────────────────
  local sg_id
  sg_id="$(aws_cmd ec2 describe-security-groups \
    --group-names "$SOLVER_SG_NAME" \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo "")"

  if [[ -n "$sg_id" && "$sg_id" != "None" ]]; then
    info "Security group already exists: $SOLVER_SG_NAME ($sg_id)"
  else
    info "Creating security group: $SOLVER_SG_NAME"
    sg_id="$(aws_cmd ec2 create-security-group \
      --group-name "$SOLVER_SG_NAME" \
      --description "solver-cloud SSH access" \
      --query 'GroupId' --output text)"

    # Allow SSH from anywhere (you may want to restrict this)
    aws_cmd ec2 authorize-security-group-ingress \
      --group-id "$sg_id" \
      --protocol tcp --port 22 --cidr 0.0.0.0/0
    info "Security group created: $sg_id (SSH open)"
  fi

  # ── IAM instance profile ───────────────────────────────────────────────
  if aws_cmd iam get-instance-profile --instance-profile-name solver-cloud-instance >/dev/null 2>&1; then
    info "IAM instance profile already exists: solver-cloud-instance"
  else
    info "Creating IAM role and instance profile..."

    # Create role
    aws_cmd iam create-role \
      --role-name solver-cloud-instance \
      --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
          "Effect": "Allow",
          "Principal": {"Service": "ec2.amazonaws.com"},
          "Action": "sts:AssumeRole"
        }]
      }'

    # Attach policies: S3 access, ECR pull, EC2 self-terminate, EC2 tagging
    aws_cmd iam put-role-policy \
      --role-name solver-cloud-instance \
      --policy-name solver-cloud-permissions \
      --policy-document '{
        "Version": "2012-10-17",
        "Statement": [
          {
            "Effect": "Allow",
            "Action": ["s3:PutObject", "s3:GetObject", "s3:ListBucket"],
            "Resource": ["arn:aws:s3:::'"$SOLVER_S3_BUCKET"'", "arn:aws:s3:::'"$SOLVER_S3_BUCKET"'/*"]
          },
          {
            "Effect": "Allow",
            "Action": ["ecr:GetAuthorizationToken", "ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"],
            "Resource": "*"
          },
          {
            "Effect": "Allow",
            "Action": ["ec2:TerminateInstances", "ec2:CreateTags"],
            "Resource": "*",
            "Condition": {"StringEquals": {"ec2:ResourceTag/Name": "solver-training"}}
          }
        ]
      }'

    # Create instance profile and attach role
    aws_cmd iam create-instance-profile --instance-profile-name solver-cloud-instance
    aws_cmd iam add-role-to-instance-profile \
      --instance-profile-name solver-cloud-instance \
      --role-name solver-cloud-instance

    info "IAM instance profile created. Waiting for propagation..."
    sleep 10
  fi

  info "Setup complete!"
}
```

**Step 2: Commit**

```bash
git add cloud/lib/setup.sh
git commit -m "feat(cloud): add setup.sh for one-time AWS provisioning"
```

---

### Task 8: Main Entry Point (solver-cloud dispatcher)

**Files:**
- Create: `cloud/solver-cloud`

**Step 1: Create the main dispatcher**

```bash
#!/usr/bin/env bash
# cloud/solver-cloud — CLI entry point for AWS compute management
#
# Usage:
#   solver-cloud setup
#   solver-cloud launch --instance <type> --config <path> --output <dir> [--on-demand] [--profile <name>]
#   solver-cloud attach <job-id>
#   solver-cloud status [job-id]
#   solver-cloud download <job-id>
#   solver-cloud terminate <job-id>

set -euo pipefail

CLOUD_DIR="$(cd "$(dirname "$0")" && pwd)"

usage() {
  cat <<'EOF'
solver-cloud — Run poker solver training on AWS

Commands:
  setup                      One-time AWS resource provisioning
  launch [options]           Build, push, and launch a training job
  attach <job-id>            SSH into instance and attach to training tmux
  status [job-id]            List jobs or show details of one
  download <job-id>          Sync finished model from S3 to local dir
  terminate <job-id>         Kill a job and terminate its instance

Launch options:
  --instance <type>          EC2 instance type (e.g. p3.2xlarge, c5.4xlarge)
  --config <path>            Path to training config YAML
  --output <dir>             Local directory for the finished model
  --on-demand                Use on-demand pricing (default: spot)
  --profile <name>           AWS CLI profile to use
EOF
  exit 1
}

[[ $# -eq 0 ]] && usage
COMMAND="$1"; shift

case "$COMMAND" in
  setup)
    source "$CLOUD_DIR/lib/setup.sh"
    run_setup
    ;;

  launch)
    # Parse args
    INSTANCE_TYPE="" CONFIG_PATH="" OUTPUT_DIR="" SPOT="spot"
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --instance)   INSTANCE_TYPE="$2"; shift 2 ;;
        --config)     CONFIG_PATH="$2"; shift 2 ;;
        --output)     OUTPUT_DIR="$2"; shift 2 ;;
        --on-demand)  SPOT="on-demand"; shift ;;
        --profile)    export SOLVER_AWS_PROFILE="$2"; shift 2 ;;
        *)            die "Unknown option: $1" ;;
      esac
    done

    [[ -z "$INSTANCE_TYPE" ]] && die "Missing --instance"
    [[ -z "$CONFIG_PATH" ]]   && die "Missing --config"
    [[ -z "$OUTPUT_DIR" ]]    && die "Missing --output"
    [[ -f "$CONFIG_PATH" ]]   || die "Config not found: $CONFIG_PATH"

    source "$CLOUD_DIR/lib/docker.sh"
    source "$CLOUD_DIR/lib/s3.sh"
    source "$CLOUD_DIR/lib/ec2.sh"

    # 1. Build and push Docker image
    IMAGE_URI="$(docker_build_and_push)"

    # 2. Upload config to S3
    CONFIG_S3="$(upload_config_for_launch "$CONFIG_PATH")"

    # 3. Launch instance
    INSTANCE_ID="$(launch_instance "$INSTANCE_TYPE" "$IMAGE_URI" "$CONFIG_S3" "$OUTPUT_DIR" "$SPOT")"

    # 4. Store local output mapping
    set_tag "$INSTANCE_ID" "local-output" "$OUTPUT_DIR"

    echo ""
    info "Job launched: $INSTANCE_ID"
    info "Attach:   ./cloud/solver-cloud attach $INSTANCE_ID"
    info "Status:   ./cloud/solver-cloud status $INSTANCE_ID"
    info "Download: ./cloud/solver-cloud download $INSTANCE_ID"
    ;;

  attach)
    [[ $# -lt 1 ]] && die "Usage: solver-cloud attach <job-id>"
    INSTANCE_ID="$1"
    source "$CLOUD_DIR/lib/config.sh"

    IP="$(get_instance_ip "$INSTANCE_ID")"
    [[ -z "$IP" || "$IP" == "None" ]] && die "Instance $INSTANCE_ID has no public IP (may be terminated)"

    info "Attaching to $INSTANCE_ID ($IP)..."
    ssh -t -i "$SOLVER_SSH_KEY" -o StrictHostKeyChecking=no \
      "ec2-user@${IP}" "tmux attach -t training 2>/dev/null || echo 'No training session found. Use: tmux ls'"
    ;;

  status)
    source "$CLOUD_DIR/lib/ec2.sh"
    if [[ $# -ge 1 ]]; then
      INSTANCE_ID="$1"
      echo "Instance:  $INSTANCE_ID"
      echo "State:     $(get_instance_state "$INSTANCE_ID")"
      echo "IP:        $(get_instance_ip "$INSTANCE_ID")"
      echo "Status:    $(get_tag "$INSTANCE_ID" "status")"
      echo "Config:    $(get_tag "$INSTANCE_ID" "config")"
      echo "S3 Path:   $(get_tag "$INSTANCE_ID" "s3-path")"
      echo "Output:    $(get_tag "$INSTANCE_ID" "local-output")"
    else
      list_instances
    fi
    ;;

  download)
    [[ $# -lt 1 ]] && die "Usage: solver-cloud download <job-id>"
    INSTANCE_ID="$1"
    source "$CLOUD_DIR/lib/s3.sh"

    LOCAL_OUTPUT="$(get_tag "$INSTANCE_ID" "local-output")"
    [[ -z "$LOCAL_OUTPUT" || "$LOCAL_OUTPUT" == "None" ]] && die "No output dir tagged for $INSTANCE_ID"

    download_bundle "$INSTANCE_ID" "$LOCAL_OUTPUT"
    ;;

  terminate)
    [[ $# -lt 1 ]] && die "Usage: solver-cloud terminate <job-id>"
    INSTANCE_ID="$1"
    source "$CLOUD_DIR/lib/ec2.sh"
    terminate_instance "$INSTANCE_ID"
    ;;

  *)
    error "Unknown command: $COMMAND"
    usage
    ;;
esac
```

**Step 2: Make executable**

```bash
chmod +x cloud/solver-cloud
```

**Step 3: Commit**

```bash
git add cloud/solver-cloud
git commit -m "feat(cloud): add solver-cloud main CLI dispatcher"
```

---

### Task 9: Test the CLI (Smoke Tests)

**Step 1: Verify help output**

```bash
./cloud/solver-cloud
```

Expected: prints usage text with all commands listed.

**Step 2: Verify Docker build locally**

```bash
docker build -f cloud/Dockerfile -t poker-solver-trainer:test .
docker run --rm poker-solver-trainer:test --help
```

Expected: prints trainer help text.

**Step 3: Verify scripts source without errors**

```bash
bash -n cloud/solver-cloud
bash -n cloud/lib/config.sh
bash -n cloud/lib/docker.sh
bash -n cloud/lib/ec2.sh
bash -n cloud/lib/s3.sh
bash -n cloud/lib/setup.sh
```

Expected: all pass with no output (no syntax errors).

**Step 4: Commit any fixes**

If any issues found, fix and commit.

---

### Task 10: Documentation

**Files:**
- Modify: `docs/training.md` — add cloud training section
- Create: `cloud/README.md`

**Step 1: Add cloud/README.md**

Quick reference for cloud usage with prerequisites, setup, and example workflows.

**Step 2: Add section to docs/training.md**

Add a "Cloud Training (AWS)" section documenting:
- Prerequisites (AWS CLI, Docker, AWS account)
- `solver-cloud setup`
- Launch, attach, status, download workflow
- Config overrides via `cloud/config.local.sh`

**Step 3: Commit**

```bash
git add cloud/README.md docs/training.md
git commit -m "docs: add cloud training documentation"
```
