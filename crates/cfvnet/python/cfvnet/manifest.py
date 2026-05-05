"""Turn-boundary dataset manifest schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from cfvnet.constants import INPUT_SIZE, NUM_COMBOS, record_size

TURN_BOUNDARY_SCHEMA_VERSION = 1
TURN_BOUNDARY_BOARD_SIZE = 4
TURN_BOUNDARY_RECORD_SIZE = record_size(TURN_BOUNDARY_BOARD_SIZE)
TURN_BOUNDARY_NORMALIZATION = "chip_cfv_over_pot_plus_stack"


@dataclass
class RecordSchema:
    """Physical TrainingRecord layout for a dataset shard."""

    format: str
    board_size: int
    record_size_bytes: int
    input_size: int
    output_size: int
    normalization: str

    @classmethod
    def turn_boundary(cls) -> RecordSchema:
        """Return the canonical turn-boundary record schema."""
        return cls(
            format="cfvnet_training_record_v1",
            board_size=TURN_BOUNDARY_BOARD_SIZE,
            record_size_bytes=TURN_BOUNDARY_RECORD_SIZE,
            input_size=INPUT_SIZE,
            output_size=NUM_COMBOS,
            normalization=TURN_BOUNDARY_NORMALIZATION,
        )


@dataclass
class ShardMetadata:
    """Metadata for a single binary shard."""

    path: str
    records: int
    board_size: int
    record_size_bytes: int
    target_source: str | None = None


@dataclass
class DatasetManifest:
    """Dataset-level metadata shared by Rust generators and Python training."""

    schema_version: int
    street: str
    record_schema: RecordSchema
    target_source: str
    source: dict[str, Any] = field(default_factory=dict)
    coverage: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    shards: list[ShardMetadata] = field(default_factory=list)

    @classmethod
    def turn_boundary(cls, target_source: str) -> DatasetManifest:
        """Create an empty turn-boundary manifest."""
        return cls(
            schema_version=TURN_BOUNDARY_SCHEMA_VERSION,
            street="turn_boundary",
            record_schema=RecordSchema.turn_boundary(),
            target_source=target_source,
            source={},
            coverage={"total_records": 0},
            validation={},
            shards=[],
        )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DatasetManifest:
        """Parse a manifest dictionary loaded from YAML."""
        schema_raw = raw["record_schema"]
        shards_raw = raw.get("shards", [])
        return cls(
            schema_version=raw["schema_version"],
            street=raw["street"],
            record_schema=RecordSchema(**schema_raw),
            target_source=raw["target_source"],
            source=raw.get("source") or {},
            coverage=raw.get("coverage") or {},
            validation=raw.get("validation") or {},
            shards=[ShardMetadata(**shard) for shard in shards_raw],
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a YAML-friendly dictionary."""
        return {
            "schema_version": self.schema_version,
            "street": self.street,
            "record_schema": self.record_schema.__dict__.copy(),
            "target_source": self.target_source,
            "source": self.source,
            "coverage": self.coverage,
            "validation": self.validation,
            "shards": [shard.__dict__.copy() for shard in self.shards],
        }

    def validate_turn_boundary(self) -> None:
        """Raise ValueError if this is not a compatible turn-boundary manifest."""
        if self.schema_version != TURN_BOUNDARY_SCHEMA_VERSION:
            raise ValueError(
                f"expected schema_version={TURN_BOUNDARY_SCHEMA_VERSION}, "
                f"got {self.schema_version}"
            )
        if self.street != "turn_boundary":
            raise ValueError(f"expected street='turn_boundary', got {self.street!r}")

        schema = self.record_schema
        expected_schema = RecordSchema.turn_boundary()
        if schema != expected_schema:
            raise ValueError(
                "record_schema does not match turn-boundary contract: "
                f"expected {expected_schema}, got {schema}"
            )

        for shard in self.shards:
            if shard.board_size != TURN_BOUNDARY_BOARD_SIZE:
                raise ValueError(
                    f"shard {shard.path} expected board_size={TURN_BOUNDARY_BOARD_SIZE}, "
                    f"got {shard.board_size}"
                )
            if shard.record_size_bytes != TURN_BOUNDARY_RECORD_SIZE:
                raise ValueError(
                    f"shard {shard.path} expected record_size_bytes="
                    f"{TURN_BOUNDARY_RECORD_SIZE}, got {shard.record_size_bytes}"
                )


def read_manifest(path: Path) -> DatasetManifest:
    """Read a dataset manifest from YAML."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return DatasetManifest.from_dict(raw)


def write_manifest(path: Path, manifest: DatasetManifest) -> None:
    """Write a dataset manifest as YAML."""
    with open(path, "w") as f:
        yaml.safe_dump(manifest.to_dict(), f, sort_keys=False)
