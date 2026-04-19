"""Constants matching the Rust BoundaryNet model."""

NUM_COMBOS: int = 1326
OUTPUT_SIZE: int = NUM_COMBOS
DECK_SIZE: int = 52
NUM_RANKS: int = 13
INPUT_SIZE: int = NUM_COMBOS + NUM_COMBOS + DECK_SIZE + NUM_RANKS + 1 + 1 + 1  # 2720
POT_INDEX: int = NUM_COMBOS + NUM_COMBOS + DECK_SIZE + NUM_RANKS  # 2717

_F32_BYTES: int = 4
_RANGE_BYTES: int = NUM_COMBOS * _F32_BYTES  # 5304
_MASK_BYTES: int = NUM_COMBOS  # 1326


def record_size(board_size: int) -> int:
    """Byte size of one TrainingRecord for a given board size.

    Mirrors Rust's cfvnet::datagen::storage::record_size().
    Layout: board_size(1) + board(board_size) + pot(4) + stack(4) + player(1)
            + game_value(4) + oop_range(1326*4) + ip_range(1326*4)
            + cfvs(1326*4) + valid_mask(1326)
    """
    return (
        1
        + board_size
        + 4
        + 4
        + 1
        + 4
        + _RANGE_BYTES * 3
        + _MASK_BYTES
    )


RECORD_SIZE_RIVER: int = record_size(5)
