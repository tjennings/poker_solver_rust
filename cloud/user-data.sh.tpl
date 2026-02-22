#!/bin/bash
# user-data.sh — EC2 bootstrap for solver training
# Substituted at launch time by envsubst.

set -euo pipefail
exec > >(tee /var/log/solver-cloud.log) 2>&1

# IMDSv2-compatible instance ID retrieval
TOKEN="$(curl -s -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 300')"
INSTANCE_ID="$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id)"
REGION="${SOLVER_AWS_REGION}"
IMAGE_URI="${SOLVER_IMAGE_URI}"
CONFIG_S3="${SOLVER_CONFIG_S3}"
S3_BUCKET="${SOLVER_S3_BUCKET}"
TAG_PREFIX="${SOLVER_TAG_PREFIX}"
S3_JOB="s3://${S3_BUCKET}/jobs/${INSTANCE_ID}"

tag() { aws ec2 create-tags --region "$REGION" --resources "$INSTANCE_ID" --tags "Key=${TAG_PREFIX}:status,Value=$1"; }

tag "initializing"

# ── Install dependencies ────────────────────────────────────────────────────
yum install -y docker tmux
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
    train -c /config/training.yaml --output-dir /output/bundle;
  echo \$? > /opt/solver/output/exit_code
"

# ── Wait for training to finish ─────────────────────────────────────────────
while tmux has-session -t training 2>/dev/null; do
  sleep 30
done

# ── Check result and upload ─────────────────────────────────────────────────
EXIT_CODE="$(cat /opt/solver/output/exit_code 2>/dev/null || echo "1")"

if [[ "$EXIT_CODE" == "0" && -d /opt/solver/output/bundle ]]; then
  tag "uploading"
  aws s3 sync /opt/solver/output/bundle/ "${S3_JOB}/bundle/" --region "$REGION"
  tag "complete"
else
  tag "failed"
fi

# ── Self-terminate ──────────────────────────────────────────────────────────
sleep 10
aws ec2 terminate-instances --region "$REGION" --instance-ids "$INSTANCE_ID"
