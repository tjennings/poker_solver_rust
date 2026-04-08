"""Binary TrainingRecord reader and PyTorch dataset."""

from __future__ import annotations

import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cfvnet.constants import DECK_SIZE, INPUT_SIZE, NUM_COMBOS, NUM_RANKS

# Byte sizes for the fixed portion of a record (after board).
_F32_BYTES = 4
_RANGE_BYTES = NUM_COMBOS * _F32_BYTES  # 5304
_MASK_BYTES = NUM_COMBOS  # 1326

# Fixed record size for 5-card boards (river).
# layout: board_size(1) + board(5) + pot(4) + stack(4) + player(1) + game_value(4)
#         + oop_range(5304) + ip_range(5304) + cfvs(5304) + valid_mask(1326) = 17257
RECORD_SIZE_RIVER = 1 + 5 + 4 + 4 + 1 + 4 + _RANGE_BYTES * 3 + _MASK_BYTES


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


@dataclass
class BoundaryItem:
    """Encoded training item for BoundaryNet."""

    input: np.ndarray     # (INPUT_SIZE,) float32
    target: np.ndarray    # (OUTPUT_SIZE,) float32
    mask: np.ndarray      # (OUTPUT_SIZE,) float32
    range: np.ndarray     # (OUTPUT_SIZE,) float32
    game_value: float
    sample_weight: float


def encode_boundary_record(rec: TrainingRecord) -> BoundaryItem:
    """Encode a TrainingRecord with normalized pot/stack and EV targets.

    Matches Rust's encode_boundary_record() exactly.

    Args:
        rec: Raw training record from binary file.

    Returns:
        BoundaryItem with normalized inputs, targets, and SPR-based weight.
    """
    total_stake = rec.pot + rec.effective_stack
    norm = total_stake if total_stake > 0.0 else 1.0

    inp = _build_input_vector(rec, norm)
    target = _build_normalized_target(rec.cfvs, rec.pot, norm)
    mask = (rec.valid_mask != 0).astype(np.float32)
    player_range = rec.oop_range if rec.player == 0 else rec.ip_range
    game_value = float(np.sum(player_range * target))
    sample_weight = _compute_spr_weight(rec.pot, rec.effective_stack)

    return BoundaryItem(
        input=inp, target=target, mask=mask,
        range=player_range.copy(), game_value=game_value,
        sample_weight=sample_weight,
    )


def _build_input_vector(rec: TrainingRecord, norm: float) -> np.ndarray:
    """Build the 2720-element input feature vector.

    Layout: oop_range(1326) + ip_range(1326) + board_onehot(52)
            + rank_presence(13) + pot/norm + stack/norm + player.
    """
    board_onehot = np.zeros(DECK_SIZE, dtype=np.float32)
    for card in rec.board:
        board_onehot[card] = 1.0

    rank_presence = np.zeros(NUM_RANKS, dtype=np.float32)
    for card in rec.board:
        rank_presence[card // 4] = 1.0

    inp = np.concatenate([
        rec.oop_range,
        rec.ip_range,
        board_onehot,
        rank_presence,
        np.array([rec.pot / norm, rec.effective_stack / norm, float(rec.player)],
                 dtype=np.float32),
    ])
    return inp


def _build_normalized_target(cfvs: np.ndarray, pot: float, norm: float) -> np.ndarray:
    """Convert pot-relative CFVs to normalized EVs: cfv * pot / total_stake."""
    return cfvs * (pot / norm)


def _compute_spr_weight(pot: float, stack: float) -> float:
    """Compute SPR-based sample weight: 1/max(SPR, 0.1), capped at 10."""
    spr = stack / pot if pot > 0.0 else 1.0
    return min(1.0 / max(spr, 0.1), 10.0)


def _resolve_bin_files(path: Path) -> list[Path]:
    """Get sorted list of .bin files from a path."""
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix == ".bin")
        if not files:
            raise ValueError(f"No .bin files found in {path}")
        return files
    return [path]


def _count_records_in_file(path: Path) -> int:
    """Count records in a binary file by dividing file size by record size."""
    size = path.stat().st_size
    if size % RECORD_SIZE_RIVER != 0:
        # Variable board size or partial records — fall back to sequential read
        return -1
    return size // RECORD_SIZE_RIVER


def _load_bulk(files: list[Path]) -> BoundaryDataset:
    """Load binary training data directly into pre-allocated numpy arrays.

    Skips Python object creation for each record. Reads raw bytes and
    slices them into the appropriate fields using numpy views.
    """
    # Count total records across all files.
    total = 0
    file_counts: list[int] = []
    for f in files:
        n = _count_records_in_file(f)
        if n < 0:
            # Fall back to slow path for non-standard files
            print(f"  Warning: {f.name} has non-standard size, using slow reader")
            records = read_records_from_path(f)
            items = [encode_boundary_record(r) for r in records]
            return BoundaryDataset(items)
        file_counts.append(n)
        total += n

    print(f"  Loading {total:,} records from {len(files)} files...")

    # Pre-allocate output arrays.
    inputs = np.zeros((total, INPUT_SIZE), dtype=np.float32)
    targets = np.zeros((total, NUM_COMBOS), dtype=np.float32)
    masks = np.zeros((total, NUM_COMBOS), dtype=np.float32)
    ranges = np.zeros((total, NUM_COMBOS), dtype=np.float32)
    game_values = np.zeros(total, dtype=np.float32)
    sample_weights = np.zeros(total, dtype=np.float32)

    offset = 0
    for file_idx, (path, n_records) in enumerate(zip(files, file_counts)):
        if n_records == 0:
            continue
        raw = np.fromfile(path, dtype=np.uint8)
        _decode_file_records(
            raw, n_records, inputs, targets, masks, ranges,
            game_values, sample_weights, offset,
        )
        offset += n_records

        if (file_idx + 1) % 100 == 0 or file_idx + 1 == len(files):
            pct = 100.0 * (file_idx + 1) / len(files)
            print(f"  [{pct:5.1f}%] {offset:,} records loaded from {file_idx + 1}/{len(files)} files",
                  flush=True)

    print(f"  Done: {offset:,} records, "
          f"{inputs.nbytes / 1024**3:.1f} GB inputs, "
          f"{targets.nbytes / 1024**3:.1f} GB targets")

    ds = BoundaryDataset.__new__(BoundaryDataset)
    ds._inputs = inputs
    ds._targets = targets
    ds._masks = masks
    ds._ranges = ranges
    ds._game_values = game_values
    ds._sample_weights = sample_weights
    return ds


def _decode_file_records(
    raw: np.ndarray,
    n_records: int,
    inputs: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    ranges: np.ndarray,
    game_values: np.ndarray,
    sample_weights: np.ndarray,
    offset: int,
) -> None:
    """Decode raw bytes from one file into pre-allocated arrays.

    Parses the fixed-layout binary format directly with numpy slicing.
    """
    rec_size = RECORD_SIZE_RIVER
    for i in range(n_records):
        base = i * rec_size
        row = offset + i
        _decode_single_record(
            raw, base, inputs[row], targets[row], masks[row],
            ranges[row], game_values, sample_weights, row,
        )


def _decode_single_record(
    raw: np.ndarray,
    base: int,
    inp_out: np.ndarray,
    target_out: np.ndarray,
    mask_out: np.ndarray,
    range_out: np.ndarray,
    game_values: np.ndarray,
    sample_weights: np.ndarray,
    row: int,
) -> None:
    """Decode one record from raw bytes into output arrays."""
    # Skip board_size byte, read 5 board cards
    board = raw[base + 1: base + 6].astype(np.int32)
    pos = base + 6

    # pot(f32), stack(f32), player(u8), game_value(f32)
    pot = np.frombuffer(raw[pos:pos + 4].tobytes(), dtype=np.float32)[0]
    stack = np.frombuffer(raw[pos + 4:pos + 8].tobytes(), dtype=np.float32)[0]
    player = raw[pos + 8]
    # game_value not used (we recompute from normalized targets)
    pos += 13

    # Ranges: oop(1326xf32), ip(1326xf32), cfvs(1326xf32)
    oop = np.frombuffer(raw[pos:pos + _RANGE_BYTES].tobytes(), dtype=np.float32)
    pos += _RANGE_BYTES
    ip = np.frombuffer(raw[pos:pos + _RANGE_BYTES].tobytes(), dtype=np.float32)
    pos += _RANGE_BYTES
    cfvs = np.frombuffer(raw[pos:pos + _RANGE_BYTES].tobytes(), dtype=np.float32)
    pos += _RANGE_BYTES
    valid_mask = raw[pos:pos + _MASK_BYTES]

    # Encode into BoundaryNet format
    total_stake = pot + stack
    norm = total_stake if total_stake > 0.0 else 1.0

    # Build input vector in-place
    inp_out[:NUM_COMBOS] = oop
    inp_out[NUM_COMBOS:2 * NUM_COMBOS] = ip
    board_start = 2 * NUM_COMBOS
    inp_out[board_start:board_start + DECK_SIZE] = 0.0
    for c in board:
        inp_out[board_start + c] = 1.0
    rank_start = board_start + DECK_SIZE
    inp_out[rank_start:rank_start + NUM_RANKS] = 0.0
    for c in board:
        inp_out[rank_start + c // 4] = 1.0
    inp_out[-3] = pot / norm
    inp_out[-2] = stack / norm
    inp_out[-1] = float(player)

    # Normalized target
    pot_over_norm = pot / norm
    np.multiply(cfvs, pot_over_norm, out=target_out)

    # Mask
    mask_out[:] = (valid_mask != 0).astype(np.float32)

    # Player range
    range_out[:] = oop if player == 0 else ip

    # Game value = sum(range * target)
    game_values[row] = float(np.dot(range_out, target_out))

    # SPR weight
    spr = stack / pot if pot > 0.0 else 1.0
    sample_weights[row] = min(1.0 / max(spr, 0.1), 10.0)


def read_records_from_path(path: Path) -> list[TrainingRecord]:
    """Read records from a file or directory.

    Args:
        path: Single .bin file or directory containing .bin files.

    Returns:
        List of all TrainingRecords.
    """
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix == ".bin")
        records: list[TrainingRecord] = []
        for f in files:
            records.extend(read_records(f))
        return records
    return read_records(path)


class BoundaryDataset:
    """PyTorch dataset of pre-encoded BoundaryItems.

    Loads all records into memory at init for fast random access.
    Supports loading from a single file or a directory of files.
    """

    def __init__(self, items: list[BoundaryItem]) -> None:
        self._inputs = np.stack([it.input for it in items])
        self._targets = np.stack([it.target for it in items])
        self._masks = np.stack([it.mask for it in items])
        self._ranges = np.stack([it.range for it in items])
        self._game_values = np.array([it.game_value for it in items], dtype=np.float32)
        self._sample_weights = np.array([it.sample_weight for it in items], dtype=np.float32)

    @classmethod
    def from_path(cls, path: Path) -> BoundaryDataset:
        """Load from a file or directory of .bin files.

        Uses fast bulk loading for large datasets (reads directly into numpy
        arrays without Python object intermediary).

        Args:
            path: Single .bin file or directory containing .bin files.

        Returns:
            BoundaryDataset with all records encoded.
        """
        files = _resolve_bin_files(path)
        return _load_bulk(files)

    @classmethod
    def from_records(cls, records: list[TrainingRecord]) -> BoundaryDataset:
        """Create from a list of TrainingRecords (for tests / small data).

        Args:
            records: List of training records.

        Returns:
            BoundaryDataset with all records encoded.
        """
        items = [encode_boundary_record(r) for r in records]
        return cls(items)

    def __len__(self) -> int:
        return len(self._game_values)

    def __getitem__(self, idx: int):
        return (
            self._inputs[idx],
            self._targets[idx],
            self._masks[idx],
            self._ranges[idx],
            self._game_values[idx],
            self._sample_weights[idx],
        )
