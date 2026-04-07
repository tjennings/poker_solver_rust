"""Binary TrainingRecord reader and PyTorch dataset."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cfvnet.constants import NUM_COMBOS

# Byte sizes for the fixed portion of a record (after board).
_F32_BYTES = 4
_RANGE_BYTES = NUM_COMBOS * _F32_BYTES  # 5304
_MASK_BYTES = NUM_COMBOS  # 1326


@dataclass
class TrainingRecord:
    """Single training record matching Rust's TrainingRecord."""

    board: list[int]
    pot: float
    effective_stack: float
    player: int
    game_value: float
    oop_range: np.ndarray  # (1326,) float32
    ip_range: np.ndarray   # (1326,) float32
    cfvs: np.ndarray       # (1326,) float32
    valid_mask: np.ndarray  # (1326,) uint8


def _read_one(f) -> TrainingRecord | None:
    """Read a single record from an open binary file.

    Returns None on EOF.
    """
    header = f.read(1)
    if len(header) == 0:
        return None
    board_size = header[0]

    board_bytes = f.read(board_size)
    if len(board_bytes) < board_size:
        return None
    board = list(board_bytes)

    # pot(f32) + stack(f32) + player(u8) + game_value(f32) = 13 bytes
    fixed = f.read(13)
    if len(fixed) < 13:
        return None
    pot = struct.unpack_from("<f", fixed, 0)[0]
    effective_stack = struct.unpack_from("<f", fixed, 4)[0]
    player = fixed[8]
    game_value = struct.unpack_from("<f", fixed, 9)[0]

    oop_raw = f.read(_RANGE_BYTES)
    ip_raw = f.read(_RANGE_BYTES)
    cfvs_raw = f.read(_RANGE_BYTES)
    mask_raw = f.read(_MASK_BYTES)

    if len(mask_raw) < _MASK_BYTES:
        return None

    oop_range = np.frombuffer(oop_raw, dtype=np.float32).copy()
    ip_range = np.frombuffer(ip_raw, dtype=np.float32).copy()
    cfvs = np.frombuffer(cfvs_raw, dtype=np.float32).copy()
    valid_mask = np.frombuffer(mask_raw, dtype=np.uint8).copy()

    return TrainingRecord(
        board=board, pot=pot, effective_stack=effective_stack,
        player=player, game_value=game_value,
        oop_range=oop_range, ip_range=ip_range,
        cfvs=cfvs, valid_mask=valid_mask,
    )


def read_records(path: Path) -> list[TrainingRecord]:
    """Read all training records from a binary file.

    Args:
        path: Path to binary training data file.

    Returns:
        List of TrainingRecord instances.
    """
    records: list[TrainingRecord] = []
    with open(path, "rb") as f:
        while True:
            rec = _read_one(f)
            if rec is None:
                break
            records.append(rec)
    return records
