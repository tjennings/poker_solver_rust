# tests/test_encoding.py
import numpy as np

from cfvnet.constants import INPUT_SIZE, NUM_COMBOS, POT_INDEX
from cfvnet.data import TrainingRecord, encode_boundary_record


def _sample_record() -> TrainingRecord:
    """Same test record as Rust's sample_record()."""
    oop = np.zeros(NUM_COMBOS, dtype=np.float32)
    oop[0] = 0.5
    oop[1] = 0.5
    ip = np.zeros(NUM_COMBOS, dtype=np.float32)
    ip[100] = 1.0
    cfvs = np.zeros(NUM_COMBOS, dtype=np.float32)
    cfvs[0] = 0.3
    mask = np.zeros(NUM_COMBOS, dtype=np.uint8)
    mask[0] = 1
    mask[1] = 1
    mask[100] = 1
    return TrainingRecord(
        board=[0, 4, 8, 12, 16], pot=100.0, effective_stack=150.0,
        player=0, game_value=0.05,
        oop_range=oop, ip_range=ip, cfvs=cfvs, valid_mask=mask,
    )


def test_encode_produces_correct_input_size():
    item = encode_boundary_record(_sample_record())
    assert item.input.shape == (INPUT_SIZE,)


def test_encode_normalizes_pot_and_stack():
    item = encode_boundary_record(_sample_record())
    # pot=100, stack=150, total=250
    assert abs(item.input[POT_INDEX] - 0.4) < 1e-6
    assert abs(item.input[POT_INDEX + 1] - 0.6) < 1e-6


def test_encode_normalizes_target():
    item = encode_boundary_record(_sample_record())
    # cfv[0]=0.3 pot-relative, target = 0.3 * 100/250 = 0.12
    assert abs(item.target[0] - 0.12) < 1e-6


def test_encode_game_value_is_range_weighted_target_sum():
    item = encode_boundary_record(_sample_record())
    # range=[0.5, 0.5, 0...], target[0]=0.12, target[1]=0
    # game_value = 0.5*0.12 + 0.5*0 = 0.06
    assert abs(item.game_value - 0.06) < 1e-6


def test_encode_sample_weight_from_spr():
    item = encode_boundary_record(_sample_record())
    # SPR = 150/100 = 1.5, weight = 1/1.5 = 0.6667
    assert abs(item.sample_weight - 1.0 / 1.5) < 1e-4


def test_encode_selects_ip_range_for_player_1():
    rec = _sample_record()
    rec.player = 1
    item = encode_boundary_record(rec)
    assert abs(item.range[100] - 1.0) < 1e-6
