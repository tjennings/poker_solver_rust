#!/usr/bin/env python3
"""Evaluate a BoundaryNet model on held-out data.

Usage:
    python scripts/eval_boundary.py \
        --model /path/to/model.onnx \
        --data /path/to/eval/data.bin
"""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort

from cfvnet.constants import INPUT_SIZE
from cfvnet.data import encode_boundary_record, read_records_from_path


def main() -> None:
    """Parse arguments and run evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate BoundaryNet")
    parser.add_argument("--model", "-m", type=Path, required=True, help="ONNX model path")
    parser.add_argument("--data", "-d", type=Path, required=True, help="Eval data (file or dir)")
    args = parser.parse_args()

    session = ort.InferenceSession(str(args.model))
    records = read_records_from_path(args.data)
    print(f"Evaluating {len(records)} records...")

    maes, spr_maes = _compute_maes(session, records)

    _print_stats("Overall", maes)
    print("\nMAE by SPR bucket:")
    for label in ["<1", "1-3", "3-10", "10+"]:
        _print_stats(f"  SPR {label:<5}", spr_maes[label])


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
    """Map SPR value to bucket label.

    Args:
        spr: Stack-to-pot ratio.

    Returns:
        Bucket label string.
    """
    if spr < 1:
        return "<1"
    if spr < 3:
        return "1-3"
    if spr < 10:
        return "3-10"
    return "10+"


def _print_stats(label: str, values: list[float]) -> None:
    """Print mean, std, and percentiles for a list of values.

    Args:
        label: Display label for the row.
        values: List of MAE values.
    """
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
