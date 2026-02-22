# Cloud Compute Reference

Run poker solver training on AWS EC2 via the `solver-cloud` CLI.

```bash
./cloud/solver-cloud <command> [options]
```

## Prerequisites

- [AWS CLI v2](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) with a configured profile
- [Docker](https://docs.docker.com/get-docker/)
- AWS account with EC2, S3, ECR, and IAM permissions

## Setup

One-time provisioning of AWS resources:

```bash
./cloud/solver-cloud setup
```

This creates:

| Resource | Name | Purpose |
|-|-|-|
| S3 bucket | `poker-solver-models` | Store training configs and finished models |
| ECR repository | `poker-solver-trainer` | Docker image registry |
| EC2 key pair | `solver-cloud` | SSH access to instances |
| Security group | `solver-cloud-sg` | SSH ingress (restricted to your IP) |
| IAM instance profile | `solver-cloud-instance` | Instance permissions (S3, ECR, self-terminate) |

All resources are created idempotently — safe to re-run.

## Commands

### launch

Build Docker image, push to ECR, and launch an EC2 instance for training.

```bash
./cloud/solver-cloud launch \
  --instance c5.4xlarge \
  --config sample_configurations/preflop_medium.yaml \
  --output ~/models/run-001
```

| Option | Description |
|-|-|
| `--instance <type>` | EC2 instance type (required) |
| `--config <path>` | Path to training config YAML (required) |
| `--output <dir>` | Local directory for the finished model (required) |
| `--on-demand` | Use on-demand pricing (default: spot) |
| `--profile <name>` | AWS CLI profile to use |

The Docker image is cached by content hash — only rebuilds when Rust source or Cargo files change.

### attach

SSH into a running instance and attach to the training tmux session.

```bash
./cloud/solver-cloud attach <job-id>
```

Detach with `Ctrl-B D` (standard tmux). The training continues after you detach.

### status

List all solver jobs or show details of a specific one.

```bash
# List all running/stopped jobs
./cloud/solver-cloud status

# Show details of a specific job
./cloud/solver-cloud status <job-id>
```

Job detail includes: instance state, public IP, training status, config path, S3 path, and local output directory.

### download

Sync the finished model bundle from S3 to the local output directory specified at launch.

```bash
./cloud/solver-cloud download <job-id>
```

### terminate

Kill a running job and terminate its EC2 instance.

```bash
./cloud/solver-cloud terminate <job-id>
```

## Job Lifecycle

```
launch
  ├── Docker build & push to ECR (cached by content hash)
  ├── Upload config to S3
  └── Launch EC2 spot instance
        ├── Pull Docker image
        ├── Download config from S3
        ├── Run training in tmux session
        ├── Upload finished bundle to S3
        └── Self-terminate
```

Training status is tracked via EC2 tags (`solver:status`): `starting` → `initializing` → `pulling-image` → `training` → `uploading` → `complete` (or `failed`).

## Configuration

Defaults are defined in `cloud/lib/config.sh`:

| Variable | Default | Description |
|-|-|-|
| `SOLVER_AWS_REGION` | `us-east-1` | AWS region |
| `SOLVER_S3_BUCKET` | `poker-solver-models` | S3 bucket for models and configs |
| `SOLVER_ECR_REPO` | `poker-solver-trainer` | ECR repository name |
| `SOLVER_KEY_NAME` | `solver-cloud` | EC2 key pair name |
| `SOLVER_SG_NAME` | `solver-cloud-sg` | Security group name |
| `SOLVER_SSH_KEY` | `~/.ssh/solver-cloud.pem` | Path to SSH private key |

### Overriding Defaults

**Per-machine** — create `cloud/config.local.sh` (gitignored):

```bash
SOLVER_AWS_REGION="us-west-2"
SOLVER_S3_BUCKET="my-custom-bucket"
SOLVER_SSH_KEY="$HOME/.ssh/my-key.pem"
```

**Per-invocation** — use environment variables:

```bash
SOLVER_AWS_REGION=eu-west-1 ./cloud/solver-cloud launch ...
```

**AWS profile** — use the `--profile` flag on launch:

```bash
./cloud/solver-cloud launch --profile my-profile ...
```

## Architecture

### File Structure

```
cloud/
├── solver-cloud          # CLI entry point (bash)
├── lib/
│   ├── config.sh         # Shared config, AWS helpers, logging
│   ├── docker.sh         # Docker build & ECR push
│   ├── ec2.sh            # EC2 launch, list, terminate
│   ├── s3.sh             # S3 upload/download
│   └── setup.sh          # One-time AWS provisioning
├── Dockerfile            # Multi-stage build (rust:1.85 → debian:bookworm-slim)
├── user-data.sh.tpl      # EC2 bootstrap template (envsubst at launch)
└── README.md             # Quick-start guide
```

### Docker Image

Multi-stage build: `rust:1.85-bookworm` compiles the trainer binary, `debian:bookworm-slim` runs it. Only the `poker-solver-trainer` binary is included in the final image.

### State Management

All job state lives in EC2 instance tags — no database or external state store. The job ID is the EC2 instance ID. Tags used:

| Tag | Content |
|-|-|
| `solver:status` | Lifecycle phase (starting, training, complete, failed, etc.) |
| `solver:config` | S3 path to the training config |
| `solver:s3-path` | S3 prefix for this job's output |
| `solver:local-output` | Local directory for model download |
| `solver:image` | Docker image URI used |

### S3 Layout

```
s3://<bucket>/
├── configs/              # Uploaded training configs
│   └── preflop_medium.yaml
└── jobs/
    └── <instance-id>/
        ├── config/       # Per-job config copy
        └── bundle/       # Finished model output
```

## Troubleshooting

**Instance fails to start**: Check `./cloud/solver-cloud status <job-id>` — if status is `failed`, the training exited non-zero. SSH in manually to inspect `/var/log/solver-cloud.log`.

**Can't attach**: Instance may have self-terminated. Check status — if `complete`, use `download` instead.

**SSH connection refused**: Your IP may have changed since setup. Add your current IP to the security group:

```bash
aws ec2 authorize-security-group-ingress \
  --group-name solver-cloud-sg \
  --protocol tcp --port 22 --cidr $(curl -s https://checkip.amazonaws.com)/32
```

**Docker build slow**: The image is cached by content hash. If source hasn't changed, launch reuses the existing ECR image.

**Spot instance terminated**: AWS can reclaim spot instances. Re-launch with `--on-demand` for critical long-running jobs.
