# AWS Vertical Scaling: Large EC2 Instance Training

**Date:** 2026-02-14
**Status:** Draft
**Prerequisite:** None (zero code changes)

## Overview

Deploy the existing trainer binary on a large multi-core EC2 instance to get 10-20x speedup over a typical development machine. No code changes required — the existing Rayon-based parallelism scales automatically with available cores.

## Why this first

- Zero code changes, zero risk of introducing bugs
- Immediate 10-20x speedup for ~$8/hr
- Validates whether more parallelism actually helps before investing in distributed architecture
- Establishes AWS deployment workflow reused by Option 2

## Instance Selection

| Instance | vCPUs | RAM | Cost/hr | Use case |
|----------|-------|-----|---------|----------|
| `c7i.24xlarge` | 96 | 192 GB | ~$4.08 | Good default |
| `c7i.48xlarge` | 192 | 384 GB | ~$8.16 | Maximum single-machine |
| `c7i.metal-48xl` | 192 | 384 GB | ~$8.16 | Bare metal, no hypervisor |
| `c7g.16xlarge` | 64 | 128 GB | ~$2.18 | ARM/Graviton, cheaper |

**Recommendation:** Start with `c7i.24xlarge` (96 vCPUs). The MCCFR parallel loop uses Rayon work-stealing which scales well to ~64-128 threads for typical sample counts. Going beyond 96 cores only helps if `mccfr_samples` is also increased proportionally.

### Why compute-optimized (c7i)

- Training is CPU-bound: pure floating-point arithmetic in CFR traversals
- Memory footprint is modest: ~100-500 MB for regret/strategy tables
- No disk I/O during training (checkpoints are infrequent)
- c7i uses Intel Sapphire Rapids with AVX-512 — good for f64 arithmetic

### Graviton (ARM) option

The codebase is pure Rust with no x86-specific dependencies. Cross-compiling for `aarch64-unknown-linux-gnu` is straightforward and Graviton instances are ~30% cheaper per vCPU.

```bash
# Cross-compile for Graviton
rustup target add aarch64-unknown-linux-gnu
cargo build --release --target aarch64-unknown-linux-gnu -p poker-solver-trainer
```

## Deployment

### Option A: Direct EC2 (simplest)

```bash
# 1. Launch instance
aws ec2 run-instances \
  --image-id ami-0abcdef1234567890 \  # Amazon Linux 2023
  --instance-type c7i.24xlarge \
  --key-name my-key \
  --security-group-ids sg-xxx \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":50,"VolumeType":"gp3"}}]'

# 2. Install Rust toolchain on instance
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 3. Clone and build
git clone <repo> && cd poker_solver_rust
cargo build --release -p poker-solver-trainer

# 4. Run training
cargo run --release -p poker-solver-trainer -- train -c training_mccfr.yaml

# 5. Copy results back
aws s3 cp ./mccfr_100bb s3://my-bucket/training-runs/run-001/ --recursive
```

### Option B: Pre-built AMI (faster iteration)

Build a custom AMI with Rust toolchain and dependencies pre-installed. Launch instances from this AMI to skip the build step.

### Option C: Docker on ECS (most reproducible)

```dockerfile
FROM rust:1.85-slim AS builder
WORKDIR /app
COPY . .
RUN cargo build --release -p poker-solver-trainer

FROM debian:bookworm-slim
COPY --from=builder /app/target/release/poker-solver-trainer /usr/local/bin/
COPY training_mccfr.yaml /app/
WORKDIR /app
CMD ["poker-solver-trainer", "train", "-c", "training_mccfr.yaml"]
```

## Training Config Adjustments

For a 96-core machine, increase `mccfr_samples` to utilize all cores:

```yaml
# training_mccfr_aws.yaml
game:
  stack_depth: 100
  bet_sizes: [0.33, 0.5, 0.75, 1.0]

abstraction:
  flop_buckets: 200
  turn_buckets: 200
  river_buckets: 500
  samples_per_street: 5000

training:
  iterations: 5000        # More iterations with faster hardware
  seed: 42
  output_dir: "./mccfr_100bb"
  mccfr_samples: 2000     # 4x more samples to saturate 96 cores
  deal_count: 100000       # Larger deal pool for better coverage
  pruning: true
  pruning_warmup_fraction: 0.2
  pruning_probe_interval: 20
```

**Key insight:** Each Rayon work unit is one sample traversal. With 96 cores, you want at least 96 samples per iteration (ideally 2-4x more for good load balancing via work-stealing). The current default of 500 already works well for 96 cores.

## Monitoring

Use `htop` or CloudWatch to verify all cores are utilized during the parallel phase. If CPU utilization is low, the bottleneck is likely the single-threaded merge/discount phase — in that case, increasing samples won't help (but it's a small fraction of total time).

## Cost Estimates

| Config | Instance | Est. time | Cost |
|--------|----------|-----------|------|
| 1K iters, 500 samples, 50K deals | c7i.24xlarge | ~30 min | ~$2 |
| 5K iters, 2K samples, 100K deals | c7i.24xlarge | ~4 hr | ~$16 |
| 10K iters, 2K samples, 100K deals | c7i.48xlarge | ~4 hr | ~$32 |

These are rough estimates. Actual time depends heavily on abstraction granularity and game tree depth.

## S3 Integration for Checkpoints

Add a post-checkpoint hook to upload results to S3 so they survive instance termination:

```bash
# Run with periodic S3 sync
poker-solver-trainer train -c training_mccfr.yaml &
TRAINER_PID=$!

# Sync checkpoints every 5 minutes
while kill -0 $TRAINER_PID 2>/dev/null; do
  aws s3 sync ./mccfr_100bb s3://my-bucket/runs/$(date +%Y%m%d)/ --quiet
  sleep 300
done
```

## Spot Instances

For 60-90% cost savings, use spot instances with checkpoint-based recovery:

```bash
aws ec2 run-instances \
  --instance-market-options '{"MarketType":"spot","SpotOptions":{"MaxPrice":"3.00","SpotInstanceType":"persistent"}}' \
  --instance-type c7i.24xlarge \
  ...
```

The existing checkpoint system saves progress every N iterations. If the spot instance is terminated, restart from the last checkpoint. This requires adding checkpoint-resume support to the trainer (not yet implemented but straightforward — deserialize `regret_sum` and `strategy_sum` from the last checkpoint file).

## When to move to Option 2

Move to distributed training (next plan) when:

1. Single-machine training still takes too long even on 192 vCPUs
2. You need to scale beyond ~2000 samples/iteration (diminishing returns from adding threads)
3. The merge/discount phase becomes the bottleneck (>20% of iteration time)
4. You want to run multiple training configurations concurrently with shared infrastructure
