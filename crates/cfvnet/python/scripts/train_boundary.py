#!/usr/bin/env python3
"""Train a BoundaryNet model.

Usage:
    python scripts/train_boundary.py \
        --config ../../sample_configurations/boundary_net_river.yaml \
        --data /path/to/training/data \
        --output /path/to/output/dir
"""

import argparse
from pathlib import Path

import torch
import yaml

from cfvnet.config import load_config
from cfvnet.export import export_onnx
from cfvnet.train import train_boundary


def main() -> None:
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(description="Train BoundaryNet")
    parser.add_argument("--config", "-c", type=Path, required=True, help="YAML config file")
    parser.add_argument(
        "--data", "-d", type=Path, required=True, help="Training data (file or dir)",
    )
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output directory")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, or cuda")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    config = load_config(args.config)

    _save_config_copy(args.config, args.output)

    print(f"Device: {device}")
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")

    result = train_boundary(
        data_path=args.data,
        config=config,
        output_dir=args.output,
        device=device,
    )

    print(f"\nTraining complete. Final loss: {result.final_train_loss:.6f}")

    _export_latest_checkpoint(config, args.output)


def _save_config_copy(config_path: Path, output_dir: Path) -> None:
    """Save a copy of the training config to the output directory.

    Args:
        config_path: Path to the original YAML config file.
        output_dir: Output directory to save the copy.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(raw, f)


def _export_latest_checkpoint(config, output_dir: Path) -> None:
    """Load the latest checkpoint and export to ONNX.

    Args:
        config: TrainConfig with model architecture parameters.
        output_dir: Directory containing checkpoints.
    """
    from cfvnet.model import BoundaryNet

    checkpoints = sorted(output_dir.glob("checkpoint_*.pt"))
    if not checkpoints:
        print("Warning: no checkpoints found, skipping ONNX export")
        return

    model = BoundaryNet(config.hidden_layers, config.hidden_size)
    ckpt = torch.load(checkpoints[-1], weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    onnx_path = output_dir / "model.onnx"
    export_onnx(model, onnx_path)
    print(f"ONNX model saved to {onnx_path}")


def _resolve_device(device_str: str) -> torch.device:
    """Resolve device string to torch.device.

    Args:
        device_str: One of "auto", "cpu", or "cuda".

    Returns:
        Resolved torch.device.
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


if __name__ == "__main__":
    main()
