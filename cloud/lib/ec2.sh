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
# Args: instance_type image_uri config_s3_path local_output_dir [spot|on-demand]
launch_instance() {
  local instance_type="$1"
  local image_uri="$2"
  local config_s3_path="$3"
  local local_output="$4"
  local spot="${5:-spot}"

  local ami sg_id s3_job_prefix

  ami="$(get_ami)"
  sg_id="$(get_sg_id)"
  [[ -z "$sg_id" ]] && die "Security group '$SOLVER_SG_NAME' not found. Run 'solver-cloud setup' first."

  info "AMI: $ami"
  info "Instance type: $instance_type"
  info "Image: $image_uri"

  # Generate user-data from template
  local template="$CLOUD_DIR/user-data.sh.tpl"
  [[ -f "$template" ]] || die "user-data template not found: $template"

  local user_data
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
