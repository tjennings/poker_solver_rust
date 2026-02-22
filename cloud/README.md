# solver-cloud

Run poker solver training on AWS EC2 with Docker.

## Prerequisites

- [AWS CLI v2](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) configured with a profile
- [Docker](https://docs.docker.com/get-docker/)
- An AWS account with permissions for EC2, S3, ECR, and IAM

## Quick Start

```bash
# 1. One-time AWS setup (creates S3 bucket, ECR repo, key pair, security group, IAM role)
./cloud/solver-cloud setup

# 2. Launch a training job
./cloud/solver-cloud launch \
  --instance c5.4xlarge \
  --config sample_configurations/fast_buckets.yaml \
  --output ~/models/run-001

# 3. Watch training progress
./cloud/solver-cloud attach <job-id>

# 4. Check job status
./cloud/solver-cloud status

# 5. Download finished model
./cloud/solver-cloud download <job-id>
```

## Commands

| Command | Description |
|-|-|
| `setup` | One-time AWS resource provisioning |
| `launch` | Build Docker image, push to ECR, launch EC2 spot instance |
| `attach <job-id>` | SSH into instance, attach to training tmux session |
| `status [job-id]` | List all jobs or show details of one |
| `download <job-id>` | Sync finished model from S3 to local directory |
| `terminate <job-id>` | Kill a running job and terminate its instance |

## Launch Options

```
--instance <type>    EC2 instance type (e.g. p3.2xlarge, c5.4xlarge)
--config <path>      Path to training config YAML
--output <dir>       Local directory for the finished model
--on-demand          Use on-demand pricing (default: spot)
--profile <name>     AWS CLI profile to use
```

## Configuration

Default settings are in `cloud/lib/config.sh`. Override per-machine by creating `cloud/config.local.sh`:

```bash
# cloud/config.local.sh (gitignored)
SOLVER_AWS_REGION="us-west-2"
SOLVER_S3_BUCKET="my-custom-bucket"
SOLVER_SSH_KEY="$HOME/.ssh/my-key.pem"
```

Or use environment variables:

```bash
SOLVER_AWS_REGION=eu-west-1 ./cloud/solver-cloud launch ...
```

## How It Works

1. **launch** builds a Docker image from the repo, pushes to ECR (cached by content hash), uploads the training config to S3, and launches an EC2 spot instance.
2. The instance's **user-data** script pulls the Docker image, downloads the config, and runs training inside a tmux session.
3. On completion, the training output is uploaded to S3 and the instance **self-terminates**.
4. **download** syncs the finished model from S3 to your local machine.

All job state is stored as EC2 instance tags — no external database needed.
