"""Tests for training artifact manifests."""

from pathlib import Path

import yaml

from cfvnet.artifact import sha256_file, write_model_artifact
from cfvnet.config import TrainConfig
from cfvnet.manifest import DatasetManifest, ShardMetadata, write_manifest


def test_write_model_artifact_records_checksums_and_dataset_metadata(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    shard = data_dir / "turn_000001.bin"
    shard.write_bytes(b"turn-boundary-records")

    manifest = DatasetManifest.turn_boundary("river_net")
    manifest.coverage["total_records"] = 2
    manifest.shards = [
        ShardMetadata(
            path=shard.name,
            records=2,
            board_size=4,
            record_size_bytes=17257,
        )
    ]
    write_manifest(data_dir / "manifest.yaml", manifest)
    (data_dir / "validation_split.yaml").write_text(
        """
schema_version: 1
seed: 123
total_records: 2
train_records: 1
validation_records: 1
validation_fraction: 0.5
strata:
  raise=single|boundary=first|spr_1_5_4:
    total_records: 2
    validation_records: 1
validation_indices:
  - 1
"""
    )

    output_dir = tmp_path / "model"
    output_dir.mkdir()
    checkpoint = output_dir / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    onnx = output_dir / "model.onnx"
    onnx.write_bytes(b"onnx")
    (output_dir / "model.onnx.data").write_bytes(b"external-data")
    (output_dir / "training_log.csv").write_text("epoch,train_loss,val_huber\n1,0.2,0.3\n")

    config_path = tmp_path / "config.yaml"
    config_path.write_text("training:\n  epochs: 1\n")
    config = TrainConfig(street="turn_boundary", board_size=4, epochs=1)

    artifact_path = write_model_artifact(
        output_dir=output_dir,
        data_path=data_dir,
        config_path=config_path,
        config=config,
        checkpoint_path=checkpoint,
        onnx_path=onnx,
    )

    raw = yaml.safe_load(artifact_path.read_text())
    assert raw["schema_version"] == 1
    assert raw["model"]["onnx"]["sha256"] == sha256_file(onnx)
    assert raw["model"]["onnx_external_data"][0]["path"] == "model.onnx.data"
    assert raw["model"]["output_unit"] == "bcfv_scaled_by_pot_over_total_stake"
    assert raw["model"]["recommended_model_kind"] == "direct_normalized_legacy"
    assert raw["config"]["training"]["street"] == "turn_boundary"
    assert raw["dataset"]["manifest"]["street"] == "turn_boundary"
    assert raw["dataset"]["manifest"]["total_records"] == 2
    assert raw["dataset"]["validation_split"]["validation_records"] == 1
    assert raw["training_log"]["final_row"]["val_huber"] == "0.3"
