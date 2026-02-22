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

    aws_cmd iam create-instance-profile --instance-profile-name solver-cloud-instance
    aws_cmd iam add-role-to-instance-profile \
      --instance-profile-name solver-cloud-instance \
      --role-name solver-cloud-instance

    info "IAM instance profile created. Waiting for propagation..."
    sleep 10
  fi

  info "Setup complete!"
}
