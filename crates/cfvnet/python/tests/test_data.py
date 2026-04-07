# tests/test_data.py
import struct
import tempfile
from pathlib import Path

import numpy as np

from cfvnet.constants import NUM_COMBOS
from cfvnet.data import read_records


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
