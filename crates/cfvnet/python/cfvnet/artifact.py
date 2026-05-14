"""Training artifact manifest helpers."""

from __future__ import annotations

import csv
import hashlib
import subprocess
from pathlib import Path
from typing import Any

import yaml

from cfvnet.config import TrainConfig
from cfvnet.manifest import read_manifest, read_validation_split


def write_model_artifact(
    output_dir: Path,
    data_path: Path,
    config_path: Path,
    config: TrainConfig,
    checkpoint_path: Path,
    onnx_path: Path,
) -> Path:
    """Write a reproducibility manifest for a trained/exported model."""
    artifact = {
        "schema_version": 1,
        "model": {
            "checkpoint": _file_entry(output_dir, checkpoint_path),
            "onnx": _file_entry(output_dir, onnx_path),
            "onnx_external_data": [
                _file_entry(output_dir, path)
                for path in sorted(output_dir.glob(f"{onnx_path.name}.*"))
            ],
            # Current Python BoundaryNet training stores targets as
            # bcfv * pot / (pot + effective_stack). Rust must evaluate these
            # checkpoints with direct_normalized_legacy, not direct.
            "output_unit": "bcfv_scaled_by_pot_over_total_stake",
            "recommended_model_kind": "direct_normalized_legacy",
        },
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path) if config_path.is_file() else None,
            "training": config.__dict__.copy(),
        },
        "dataset": _dataset_entry(data_path),
        "training_log": _training_log_entry(output_dir / "training_log.csv"),
        "git": {"commit": _git_commit()},
    }

    path = output_dir / "model_artifact.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(artifact, f, sort_keys=False)
    return path


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_entry(base_dir: Path, path: Path) -> dict[str, Any]:
    rel = path.relative_to(base_dir) if path.is_relative_to(base_dir) else path
    return {
        "path": str(rel),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _dataset_entry(data_path: Path) -> dict[str, Any]:
    entry: dict[str, Any] = {"path": str(data_path)}
    manifest_path = _find_neighbor(data_path, "manifest.yaml", "manifest.yml")
    validation_path = _find_neighbor(data_path, "validation_split.yaml", "validation_split.yml")

    if manifest_path is not None:
        manifest = read_manifest(manifest_path)
        entry["manifest"] = {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "schema_version": manifest.schema_version,
            "street": manifest.street,
            "target_source": manifest.target_source,
            "total_records": manifest.coverage.get("total_records"),
            "shards": len(manifest.shards),
        }
    elif data_path.is_file():
        entry["sha256"] = sha256_file(data_path)
        entry["bytes"] = data_path.stat().st_size

    if validation_path is not None:
        split = read_validation_split(validation_path)
        entry["validation_split"] = {
            "path": str(validation_path),
            "sha256": sha256_file(validation_path),
            "total_records": split.total_records,
            "train_records": split.train_records,
            "validation_records": split.validation_records,
            "validation_fraction": split.validation_fraction,
            "strata": len(split.strata),
        }

    return entry


def _training_log_entry(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "final_row": rows[-1],
    }


def _find_neighbor(path: Path, *names: str) -> Path | None:
    base = path if path.is_dir() else path.parent
    for name in names:
        candidate = base / name
        if candidate.is_file():
            return candidate
    return None


def _git_commit() -> str | None:
    repo = Path(__file__).resolve().parents[4]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()
