from pathlib import Path

import pytest

from cfvnet.constants import INPUT_SIZE, NUM_COMBOS, record_size
from cfvnet.data import _resolve_bin_files
from cfvnet.manifest import (
    TURN_BOUNDARY_BOARD_SIZE,
    TURN_BOUNDARY_RECORD_SIZE,
    DatasetManifest,
    ShardMetadata,
    read_manifest,
    write_manifest,
)


def test_turn_boundary_manifest_contract_matches_record_layout(tmp_path: Path):
    manifest = DatasetManifest.turn_boundary("river_net")
    schema = manifest.record_schema

    assert manifest.schema_version == 1
    assert manifest.street == "turn_boundary"
    assert schema.board_size == TURN_BOUNDARY_BOARD_SIZE
    assert schema.record_size_bytes == record_size(4)
    assert schema.record_size_bytes == TURN_BOUNDARY_RECORD_SIZE
    assert schema.input_size == INPUT_SIZE
    assert schema.output_size == NUM_COMBOS
    assert schema.normalization == "chip_cfv_over_pot_plus_stack"

    manifest.validate_turn_boundary()


def test_manifest_round_trips_yaml(tmp_path: Path):
    manifest = DatasetManifest.turn_boundary("river_net")
    manifest.source["generator_commit"] = "abc123"
    manifest.coverage["total_records"] = 256
    manifest.coverage["by_raise_depth"] = {"4bet_plus": 12}
    manifest.shards.append(
        ShardMetadata(
            path="turn_000001.bin",
            records=256,
            board_size=4,
            record_size_bytes=record_size(4),
        )
    )

    path = tmp_path / "manifest.yaml"
    write_manifest(path, manifest)
    loaded = read_manifest(path)

    assert loaded == manifest
    loaded.validate_turn_boundary()


def test_manifest_validation_rejects_river_schema():
    manifest = DatasetManifest.turn_boundary("river_net")
    manifest.record_schema.board_size = 5

    with pytest.raises(ValueError, match="record_schema does not match"):
        manifest.validate_turn_boundary()


def test_dataset_directory_uses_manifest_shard_order(tmp_path: Path):
    first = tmp_path / "turn_000002.bin"
    second = tmp_path / "turn_000001.bin"
    ignored = tmp_path / "ignored.bin"
    for path in (first, second, ignored):
        path.write_bytes(b"")

    manifest = DatasetManifest.turn_boundary("river_net")
    manifest.shards = [
        ShardMetadata(
            path=first.name,
            records=0,
            board_size=4,
            record_size_bytes=record_size(4),
        ),
        ShardMetadata(
            path=second.name,
            records=0,
            board_size=4,
            record_size_bytes=record_size(4),
        ),
    ]
    write_manifest(tmp_path / "manifest.yaml", manifest)

    assert _resolve_bin_files(tmp_path) == [first, second]


def test_dataset_directory_manifest_rejects_missing_shard(tmp_path: Path):
    manifest = DatasetManifest.turn_boundary("river_net")
    manifest.shards = [
        ShardMetadata(
            path="missing.bin",
            records=10,
            board_size=4,
            record_size_bytes=record_size(4),
        )
    ]
    write_manifest(tmp_path / "manifest.yaml", manifest)

    with pytest.raises(ValueError, match="missing shard"):
        _resolve_bin_files(tmp_path)
