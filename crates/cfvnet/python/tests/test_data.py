# tests/test_data.py
import struct
import tempfile
from pathlib import Path

import numpy as np

from cfvnet.constants import (
    NUM_COMBOS,
    RECORD_SIZE_RIVER,
    record_size,
)
from cfvnet.data import (
    _count_records_in_file,
    _encode_raw_to_tensors,
    read_records,
)


def _write_test_record(f, board: list[int], pot: float, stack: float,
                       player: int, game_value: float) -> None:
    """Write one TrainingRecord in Rust binary format."""
    f.write(struct.pack("B", len(board)))
    f.write(bytes(board))
    f.write(struct.pack("<f", pot))
    f.write(struct.pack("<f", stack))
    f.write(struct.pack("B", player))
    f.write(struct.pack("<f", game_value))
    # oop_range, ip_range, cfvs: 1326 x f32 each
    oop = np.zeros(NUM_COMBOS, dtype=np.float32)
    oop[0] = 0.5
    oop[1] = 0.5
    f.write(oop.tobytes())
    ip = np.zeros(NUM_COMBOS, dtype=np.float32)
    ip[100] = 1.0
    f.write(ip.tobytes())
    cfvs = np.zeros(NUM_COMBOS, dtype=np.float32)
    cfvs[0] = 0.3
    f.write(cfvs.tobytes())
    mask = np.zeros(NUM_COMBOS, dtype=np.uint8)
    mask[0] = 1
    mask[1] = 1
    mask[100] = 1
    f.write(mask.tobytes())


def test_read_records_parses_single_record():
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        _write_test_record(f, [0, 4, 8, 12, 16], 100.0, 150.0, 0, 0.05)
        f.flush()
        path = Path(f.name)

    records = read_records(path)
    assert len(records) == 1
    rec = records[0]
    assert rec.board == [0, 4, 8, 12, 16]
    assert abs(rec.pot - 100.0) < 1e-6
    assert abs(rec.effective_stack - 150.0) < 1e-6
    assert rec.player == 0
    assert abs(rec.oop_range[0] - 0.5) < 1e-6
    assert abs(rec.cfvs[0] - 0.3) < 1e-6
    assert rec.valid_mask[0] == 1
    assert rec.valid_mask[2] == 0


def test_read_records_handles_multiple():
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        _write_test_record(f, [0, 4, 8, 12, 16], 100.0, 150.0, 0, 0.05)
        _write_test_record(f, [1, 5, 9, 13, 17], 200.0, 50.0, 1, -0.1)
        f.flush()
        path = Path(f.name)

    records = read_records(path)
    assert len(records) == 2
    assert abs(records[1].pot - 200.0) < 1e-6
    assert records[1].player == 1


def test_record_size_matches_rust_layout():
    # board_size(1) + board(N) + pot(4) + stack(4) + player(1) + game_value(4)
    # + oop(5304) + ip(5304) + cfvs(5304) + mask(1326) = 17253 + N
    base = 1 + 4 + 4 + 1 + 4 + NUM_COMBOS * 4 * 3 + NUM_COMBOS
    assert record_size(3) == base + 3
    assert record_size(4) == base + 4
    assert record_size(5) == base + 5
    assert record_size(5) == RECORD_SIZE_RIVER


def _encode_record_bytes(board: list[int], pot: float, stack: float,
                         player: int, game_value: float,
                         oop: np.ndarray, ip: np.ndarray,
                         cfvs: np.ndarray, mask: np.ndarray) -> bytes:
    """Serialize a TrainingRecord to bytes matching the Rust layout."""
    parts = [
        struct.pack("B", len(board)),
        bytes(board),
        struct.pack("<f", pot),
        struct.pack("<f", stack),
        struct.pack("B", player),
        struct.pack("<f", game_value),
        oop.astype(np.float32).tobytes(),
        ip.astype(np.float32).tobytes(),
        cfvs.astype(np.float32).tobytes(),
        mask.astype(np.uint8).tobytes(),
    ]
    return b"".join(parts)


def _decode_and_assert(board: list[int], pot: float, stack: float,
                      player: int, oop: np.ndarray, ip: np.ndarray,
                      cfvs: np.ndarray, mask: np.ndarray) -> None:
    """Encode one record, run it through _decode_single_record, check fields."""
    raw_bytes = _encode_record_bytes(
        board, pot, stack, player, 0.123, oop, ip, cfvs, mask,
    )
    raw = np.frombuffer(raw_bytes, dtype=np.uint8).copy()
    assert len(raw) == record_size(len(board))

    inp, target, mask_out, player_range, gv, sw = _encode_raw_to_tensors(raw)

    # Input layout: oop(1326) + ip(1326) + board_onehot(52) + rank(13) + pot/s/player
    np.testing.assert_allclose(inp[:NUM_COMBOS], oop, atol=1e-6)
    np.testing.assert_allclose(inp[NUM_COMBOS:2 * NUM_COMBOS], ip, atol=1e-6)

    board_oh = np.zeros(52, dtype=np.float32)
    for c in board:
        board_oh[c] = 1.0
    np.testing.assert_allclose(
        inp[2 * NUM_COMBOS:2 * NUM_COMBOS + 52], board_oh, atol=1e-6,
    )

    rank_oh = np.zeros(13, dtype=np.float32)
    for c in board:
        rank_oh[c // 4] = 1.0
    np.testing.assert_allclose(
        inp[2 * NUM_COMBOS + 52:2 * NUM_COMBOS + 52 + 13], rank_oh, atol=1e-6,
    )

    norm = pot + stack
    assert abs(inp[-3] - pot / norm) < 1e-6
    assert abs(inp[-2] - stack / norm) < 1e-6
    assert abs(inp[-1] - float(player)) < 1e-6

    np.testing.assert_allclose(target, cfvs * (pot / norm), atol=1e-6)
    np.testing.assert_allclose(
        mask_out, (mask != 0).astype(np.float32), atol=1e-6,
    )
    expected_range = oop if player == 0 else ip
    np.testing.assert_allclose(player_range, expected_range, atol=1e-6)

    expected_gv = float(np.sum(expected_range * cfvs * (pot / norm)))
    assert abs(gv - expected_gv) < 1e-4

    spr = stack / pot if pot > 0 else 1.0
    expected_sw = min(1.0 / max(spr, 0.1), 10.0)
    assert abs(sw - expected_sw) < 1e-4


def test_decode_single_record_river_5_card_board():
    oop = np.zeros(NUM_COMBOS, dtype=np.float32)
    oop[0] = 0.4
    oop[7] = 0.6
    ip = np.zeros(NUM_COMBOS, dtype=np.float32)
    ip[200] = 1.0
    cfvs = np.zeros(NUM_COMBOS, dtype=np.float32)
    cfvs[0] = 0.25
    cfvs[7] = -0.3
    mask = np.zeros(NUM_COMBOS, dtype=np.uint8)
    mask[0] = 1
    mask[7] = 1
    _decode_and_assert(
        board=[0, 4, 8, 12, 16], pot=100.0, stack=200.0, player=0,
        oop=oop, ip=ip, cfvs=cfvs, mask=mask,
    )


def test_decode_single_record_turn_4_card_board():
    oop = np.zeros(NUM_COMBOS, dtype=np.float32)
    oop[10] = 0.5
    oop[11] = 0.5
    ip = np.zeros(NUM_COMBOS, dtype=np.float32)
    ip[300] = 1.0
    cfvs = np.zeros(NUM_COMBOS, dtype=np.float32)
    cfvs[10] = 0.1
    cfvs[11] = -0.2
    mask = np.zeros(NUM_COMBOS, dtype=np.uint8)
    mask[10] = 1
    mask[11] = 1
    _decode_and_assert(
        board=[5, 11, 22, 33], pot=50.0, stack=50.0, player=1,
        oop=oop, ip=ip, cfvs=cfvs, mask=mask,
    )


def test_decode_single_record_flop_3_card_board():
    oop = np.ones(NUM_COMBOS, dtype=np.float32) * (1.0 / NUM_COMBOS)
    ip = np.ones(NUM_COMBOS, dtype=np.float32) * (1.0 / NUM_COMBOS)
    cfvs = np.zeros(NUM_COMBOS, dtype=np.float32)
    cfvs[50] = 0.5
    mask = np.ones(NUM_COMBOS, dtype=np.uint8)
    _decode_and_assert(
        board=[1, 17, 42], pot=20.0, stack=180.0, player=0,
        oop=oop, ip=ip, cfvs=cfvs, mask=mask,
    )


def test_count_records_in_file_river():
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        for _ in range(3):
            _write_test_record(f, [0, 4, 8, 12, 16], 100.0, 50.0, 0, 0.0)
        f.flush()
        path = Path(f.name)
    assert _count_records_in_file(path) == 3


def test_count_records_in_file_turn():
    oop = np.zeros(NUM_COMBOS, dtype=np.float32)
    ip = np.zeros(NUM_COMBOS, dtype=np.float32)
    cfvs = np.zeros(NUM_COMBOS, dtype=np.float32)
    mask = np.zeros(NUM_COMBOS, dtype=np.uint8)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        for _ in range(5):
            f.write(_encode_record_bytes(
                [0, 4, 8, 12], 100.0, 50.0, 0, 0.0, oop, ip, cfvs, mask,
            ))
        f.flush()
        path = Path(f.name)
    assert _count_records_in_file(path) == 5


def test_count_records_in_file_flop():
    oop = np.zeros(NUM_COMBOS, dtype=np.float32)
    ip = np.zeros(NUM_COMBOS, dtype=np.float32)
    cfvs = np.zeros(NUM_COMBOS, dtype=np.float32)
    mask = np.zeros(NUM_COMBOS, dtype=np.uint8)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        for _ in range(2):
            f.write(_encode_record_bytes(
                [0, 4, 8], 100.0, 50.0, 0, 0.0, oop, ip, cfvs, mask,
            ))
        f.flush()
        path = Path(f.name)
    assert _count_records_in_file(path) == 2


def test_count_records_in_file_empty():
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        path = Path(f.name)
    assert _count_records_in_file(path) == 0


def test_count_records_in_file_corrupt_returns_negative():
    # A file with a valid board_size prefix but a byte count that is not
    # a multiple of record_size(board_size) must return -1 (corrupt).
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        # First byte is a valid board size (5), but write only 10 bytes total
        # so size = 10, which is not divisible by record_size(5) = 17258.
        f.write(struct.pack("B", 5))
        f.write(b"\x00" * 9)
        f.flush()
        path = Path(f.name)
    assert _count_records_in_file(path) == -1
