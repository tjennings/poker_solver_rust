# solver-cloud: AWS Compute CLI

**Date:** 2026-02-22
**Status:** Approved

## Overview

Bash scripts in `cloud/` that orchestrate Docker build → ECR push → EC2 launch → S3 upload → local sync for running poker solver training on AWS. No database — all state tracked via EC2 tags and S3 prefixes.

## Directory Structure

```
cloud/
├── solver-cloud          # Main entry point (dispatcher)
├── lib/
│   ├── config.sh         # AWS defaults (region, S3 bucket, ECR repo, key pair, SG)
│   ├── docker.sh         # Build & push Docker image to ECR
│   ├── ec2.sh            # Launch, list, terminate instances
│   └── s3.sh             # Upload/download model bundles
├── Dockerfile            # Multi-stage: build solver binary, slim runtime image
└── user-data.sh.tpl      # EC2 bootstrap template (pull image, run training, upload, terminate)
```

## Commands

| Command | What it does |
|-|-|
| `solver-cloud launch --instance <type> --config <path> --output <local-dir>` | Build/push Docker image, launch EC2 spot instance, start training |
| `solver-cloud attach <job-id>` | SSH into instance, attach to tmux session showing training output |
| `solver-cloud status [job-id]` | List running jobs or show details of one (instance state, uptime, training phase) |
| `solver-cloud download <job-id>` | Sync finished model from S3 to local destination |
| `solver-cloud terminate <job-id>` | Kill a running job and terminate instance |
| `solver-cloud setup` | One-time: create S3 bucket, ECR repo, security group, key pair |

## Job Lifecycle

```
launch → EC2 starts → user-data pulls Docker image → runs training in tmux
                                                    ↓
                                              training completes
                                                    ↓
                                         uploads bundle to S3
                                                    ↓
                                         self-terminates instance
```

Attach at any point while running. After completion, download syncs from S3.

## Design Decisions

1. **Job ID = EC2 instance ID** — no separate tracking. Tags store config, output path, status.
2. **EC2 tags**: `solver:job-config`, `solver:s3-path`, `solver:status` (running/uploading/complete/failed).
3. **Spot instances by default** — with `--on-demand` flag for long-running critical jobs.
4. **Docker image cached** — only rebuilds when source changes (content hash tag).
5. **tmux on instance** — training runs inside tmux so attach works and output survives SSH disconnects.
6. **user-data.sh** handles the full lifecycle autonomously: pull image → run → upload → tag complete → terminate.
7. **S3 path**: `s3://<bucket>/jobs/<instance-id>/bundle/`.
8. **AWS auth**: existing AWS CLI profiles via `--profile` flag.
9. **Auto-terminate on completion** — instance uploads to S3 then self-terminates.

## Dockerfile

```dockerfile
# Stage 1: Build
FROM rust:1.85 AS builder
COPY . /src
RUN cargo build -p poker-solver-trainer --release

# Stage 2: Runtime
FROM debian:bookworm-slim
COPY --from=builder /src/target/release/poker-solver-trainer /usr/local/bin/
ENTRYPOINT ["poker-solver-trainer"]
```

## Example Usage

```bash
# One-time setup
./cloud/solver-cloud setup

# Launch training on a big instance
./cloud/solver-cloud launch \
  --instance p3.2xlarge \
  --config sample_configurations/fast_buckets.yaml \
  --output ~/models/run-001

# Watch it run
./cloud/solver-cloud attach i-0abc123def

# Check status
./cloud/solver-cloud status

# Download when done
./cloud/solver-cloud download i-0abc123def
```
