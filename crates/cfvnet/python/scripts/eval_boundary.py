#!/usr/bin/env python3
"""Evaluate a BoundaryNet model on held-out data.

Accepts ONNX models (.onnx), PyTorch checkpoints (.pt), or a model
directory (auto-finds latest checkpoint, exports to ONNX if needed).

Usage:
    python scripts/eval_boundary.py --model /path/to/model.onnx --data /path/to/data.bin
    python scripts/eval_boundary.py --model /path/to/checkpoint_epoch100.pt --data /path/to/data.bin
    python scripts/eval_boundary.py --model /path/to/model_dir --data /path/to/data.bin
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort

from cfvnet.constants import INPUT_SIZE
from cfvnet.data import encode_boundary_record, read_records_from_path


def main() -> None:
    """Parse arguments and run evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate BoundaryNet")
    parser.add_argument("--model", "-m", type=Path, required=True,
                        help="ONNX model, PyTorch checkpoint (.pt), or model directory")
    parser.add_argument("--data", "-d", type=Path, required=True,
                        help="Eval data (file or dir)")
    args = parser.parse_args()

    session = _load_model(args.model)
    records = read_records_from_path(args.data)
    print(f"Evaluating {len(records)} records...")

    maes, spr_maes = _compute_maes(session, records)

    _print_stats("Overall", maes)
    print("\nMAE by SPR bucket:")
    for label in ["<1", "1-3", "3-10", "10+"]:
        _print_stats(f"  SPR {label:<5}", spr_maes[label])


def _load_model(model_path: Path) -> ort.InferenceSession:
    """Load model from ONNX, PyTorch checkpoint, or directory.

    If given a .pt file or directory, exports to ONNX first.

    Args:
        model_path: Path to .onnx, .pt, or model directory.

    Returns:
        ONNX InferenceSession ready for inference.
    """
    if model_path.suffix == ".onnx":
        print(f"Loading ONNX model: {model_path}")
        return ort.InferenceSession(str(model_path))

    if model_path.suffix == ".pt":
        return _export_and_load(model_path)

    if model_path.is_dir():
        # Check for existing ONNX export.
        onnx_path = model_path / "model.onnx"
        if onnx_path.exists():
            print(f"Loading ONNX model: {onnx_path}")
            return ort.InferenceSession(str(onnx_path))
        # Find latest checkpoint.
        checkpoints = sorted(
            model_path.glob("checkpoint_epoch*.pt"),
            key=lambda p: int(p.stem.replace("checkpoint_epoch", "")),
        )
        if checkpoints:
            return _export_and_load(checkpoints[-1])
        raise FileNotFoundError(f"No .onnx or .pt files found in {model_path}")

    raise ValueError(f"Unsupported model format: {model_path}")


def _export_and_load(checkpoint_path: Path) -> ort.InferenceSession:
    """Load PyTorch checkpoint, export to ONNX, return session.

    Reads hidden_layers and hidden_size from config.yaml in the same
    directory, or uses defaults.

    Args:
        checkpoint_path: Path to .pt checkpoint file.

    Returns:
        ONNX InferenceSession.
    """
    import torch

    from cfvnet.config import load_config
    from cfvnet.export import export_onnx
    from cfvnet.model import BoundaryNet

    # Try to read config from model directory.
    config_path = checkpoint_path.parent / "config.yaml"
    if config_path.exists():
        cfg = load_config(config_path)
        hidden_layers = cfg.hidden_layers
        hidden_size = cfg.hidden_size
        print(f"Config: {hidden_layers} layers x {hidden_size} hidden")
    else:
        hidden_layers = 7
        hidden_size = 768
        print(f"No config.yaml found, using defaults: {hidden_layers}x{hidden_size}")

    print(f"Loading checkpoint: {checkpoint_path}")
    model = BoundaryNet(hidden_layers, hidden_size)
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])

    # Export to temp ONNX file.
    onnx_path = checkpoint_path.with_suffix(".onnx")
    print(f"Exporting to ONNX: {onnx_path}")
    export_onnx(model, onnx_path)

    return ort.InferenceSession(str(onnx_path))


def _compute_maes(
    session: ort.InferenceSession,
    records: list,
) -> tuple[list[float], dict[str, list[float]]]:
    """Compute per-record MAE values and bucket by SPR.

    Args:
        session: ONNX inference session.
        records: List of TrainingRecord instances.

    Returns:
        Tuple of (overall MAE list, dict mapping SPR bucket to MAE list).
    """
    maes: list[float] = []
    spr_maes: dict[str, list[float]] = {"<1": [], "1-3": [], "3-10": [], "10+": []}

    for rec in records:
        item = encode_boundary_record(rec)
        inp = item.input.reshape(1, INPUT_SIZE).astype(np.float32)
        pred = session.run(None, {"input": inp})[0][0]

        valid = item.mask > 0.5
        if valid.sum() == 0:
            continue
        mae = float(np.abs(pred[valid] - item.target[valid]).mean())
        maes.append(mae)

        spr = rec.effective_stack / rec.pot if rec.pot > 0 else 0.0
        bucket = _spr_bucket(spr)
        spr_maes[bucket].append(mae)

    return maes, spr_maes


def _spr_bucket(spr: float) -> str:
    """Map SPR value to bucket label."""
    if spr < 1:
        return "<1"
    if spr < 3:
        return "1-3"
    if spr < 10:
        return "3-10"
    return "10+"


def _print_stats(label: str, values: list[float]) -> None:
    """Print mean, std, and percentiles for a list of values."""
    if not values:
        print(f"{label}: N/A (0 records)")
        return
    arr = np.array(sorted(values))
    n = len(arr)

    def percentile(frac: float) -> float:
        return float(arr[int(frac * (n - 1))])

    print(
        f"{label}: mean={arr.mean():.6f} std={arr.std():.4f} "
        f"p50={percentile(0.5):.4f} p90={percentile(0.9):.4f} p95={percentile(0.95):.4f} "
        f"p99={percentile(0.99):.4f} max={arr.max():.4f}  ({n} records)"
    )


if __name__ == "__main__":
    main()
