"""Constants matching the Rust BoundaryNet model."""

NUM_COMBOS: int = 1326
OUTPUT_SIZE: int = NUM_COMBOS
DECK_SIZE: int = 52
NUM_RANKS: int = 13
INPUT_SIZE: int = NUM_COMBOS + NUM_COMBOS + DECK_SIZE + NUM_RANKS + 1 + 1 + 1  # 2720
POT_INDEX: int = NUM_COMBOS + NUM_COMBOS + DECK_SIZE + NUM_RANKS  # 2717
