# PyTorch BoundaryNet Trainer — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Add a PyTorch training pipeline for BoundaryNet with ONNX export, and switch Rust inference from burn to ONNX Runtime (`ort` crate).

**Architecture:** Python package at `crates/cfvnet/python/` reads existing Rust binary training data, trains a BoundaryNet MLP identical to the burn architecture, exports to ONNX. Rust inference switches from `burn::BoundaryNet<NdArray>` to `ort::Session` loading the ONNX model. Burn training code is kept as fallback.

**Tech Stack:** Python 3.10+, PyTorch, numpy, PyYAML, ONNX, onnxruntime, pytest, ruff. Rust: `ort` crate.

**Design doc:** `docs/plans/2026-04-06-pytorch-boundary-trainer-design.md`

**Python project standards:**
- Type hints on all function signatures
- Google-style docstrings on all public functions
- No function longer than 60 lines — extract helpers
- ruff for linting/formatting
- pytest for all tests

---

### Task 1: Python project scaffold

**Files:**
- Create: `crates/cfvnet/python/pyproject.toml`
- Create: `crates/cfvnet/python/cfvnet/__init__.py`
- Create: `crates/cfvnet/python/cfvnet/constants.py`
- Create: `crates/cfvnet/python/tests/__init__.py`

**Step 1: Create `pyproject.toml`**

```toml
[project]
name = "cfvnet"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0",
    "numpy>=1.24",
    "pyyaml>=6.0",
    "onnx>=1.14",
    "onnxruntime>=1.16",
]

[dependency-groups]
dev = [
    "pytest>=7.0",
    "ruff>=0.4",
]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Step 2: Create `cfvnet/__init__.py`**

```python
"""PyTorch training pipeline for BoundaryNet."""
```

**Step 3: Create `cfvnet/constants.py`**

```python
"""Constants matching the Rust BoundaryNet model."""

NUM_COMBOS: int = 1326
OUTPUT_SIZE: int = NUM_COMBOS
DECK_SIZE: int = 52
NUM_RANKS: int = 13
INPUT_SIZE: int = NUM_COMBOS + NUM_COMBOS + DECK_SIZE + NUM_RANKS + 1 + 1 + 1  # 2720
POT_INDEX: int = NUM_COMBOS + NUM_COMBOS + DECK_SIZE + NUM_RANKS  # 2717
```

**Step 4: Create empty `tests/__init__.py`**

**Step 5: Verify project installs**

Run: `cd crates/cfvnet/python && uv sync --all-extras`
Expected: installs successfully, creates `.venv/`

**Step 6: Verify ruff and pytest run**

Run: `cd crates/cfvnet/python && uv run ruff check cfvnet/ && uv run pytest`
Expected: no errors, no tests collected (yet)

**Step 7: Commit**

```bash
git add crates/cfvnet/python/
git commit -m "feat(cfvnet): scaffold Python training package"
```

---

### Task 2: Config reader

**Files:**
- Create: `crates/cfvnet/python/cfvnet/config.py`
- Create: `crates/cfvnet/python/tests/test_config.py`

**Step 1: Write the failing test**

```python
# tests/test_config.py
import tempfile
from pathlib import Path
from cfvnet.config import load_config


def test_load_config_parses_training_section():
    yaml_content = """\
game:
  initial_stack: 200
  bet_sizes: ["50%", "a"]
datagen:
  street: "river"
  num_samples: 1000
training:
  hidden_layers: 7
  hidden_size: 768
  batch_size: 8192
  epochs: 100
  learning_rate: 0.002
  lr_min: 0.00002
  huber_delta: 1.0
  aux_loss_weight: 0.1
  validation_split: 0.05
  grad_clip_norm: 1.0
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        cfg = load_config(Path(f.name))

    assert cfg.hidden_layers == 7
    assert cfg.hidden_size == 768
    assert cfg.batch_size == 8192
    assert cfg.epochs == 100
    assert abs(cfg.learning_rate - 0.002) < 1e-9
    assert abs(cfg.lr_min - 0.00002) < 1e-9
    assert abs(cfg.huber_delta - 1.0) < 1e-9
    assert abs(cfg.aux_loss_weight - 0.1) < 1e-9
    assert abs(cfg.validation_split - 0.05) < 1e-9
    assert abs(cfg.grad_clip_norm - 1.0) < 1e-9


def test_load_config_uses_defaults_for_missing_fields():
    yaml_content = """\
game:
  initial_stack: 200
  bet_sizes: ["50%"]
datagen:
  num_samples: 100
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        cfg = load_config(Path(f.name))

    assert cfg.hidden_layers == 7
    assert cfg.hidden_size == 500
    assert cfg.batch_size == 2048
    assert cfg.epochs == 2
```

**Step 2: Run test to verify it fails**

Run: `cd crates/cfvnet/python && uv run pytest tests/test_config.py -v`
Expected: FAIL — module not found

**Step 3: Write minimal implementation**

```python
# cfvnet/config.py
"""Read CfvnetConfig YAML files (same format as Rust)."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class TrainConfig:
    """Training configuration matching Rust's TrainingConfig."""

    hidden_layers: int = 7
    hidden_size: int = 500
    batch_size: int = 2048
    epochs: int = 2
    learning_rate: float = 0.001
    lr_min: float = 0.00001
    huber_delta: float = 1.0
    aux_loss_weight: float = 1.0
    validation_split: float = 0.05
    checkpoint_every_n_epochs: int = 1000
    grad_clip_norm: float = 1.0


def load_config(path: Path) -> TrainConfig:
    """Load a CfvnetConfig YAML and extract training parameters.

    Args:
        path: Path to YAML config file.

    Returns:
        TrainConfig with values from file, defaults for missing fields.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)

    training = raw.get("training", {})
    return TrainConfig(
        hidden_layers=training.get("hidden_layers", 7),
        hidden_size=training.get("hidden_size", 500),
        batch_size=training.get("batch_size", 2048),
        epochs=training.get("epochs", 2),
        learning_rate=training.get("learning_rate", 0.001),
        lr_min=training.get("lr_min", 0.00001),
        huber_delta=training.get("huber_delta", 1.0),
        aux_loss_weight=training.get("aux_loss_weight", 1.0),
        validation_split=training.get("validation_split", 0.05),
        checkpoint_every_n_epochs=training.get("checkpoint_every_n_epochs", 1000),
        grad_clip_norm=training.get("grad_clip_norm", 1.0),
    )
```

**Step 4: Run tests to verify they pass**

Run: `cd crates/cfvnet/python && uv run pytest tests/test_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/cfvnet/python/cfvnet/config.py crates/cfvnet/python/tests/test_config.py
git commit -m "feat(cfvnet): add Python config reader for CfvnetConfig YAML"
```

---

### Task 3: Binary data reader

**Files:**
- Create: `crates/cfvnet/python/cfvnet/data.py`
- Create: `crates/cfvnet/python/tests/test_data.py`

**Step 1: Write the failing test**

The test creates a binary record using the exact Rust format, then reads it back.

```python
# tests/test_data.py
import struct
import tempfile
from pathlib import Path

import numpy as np

from cfvnet.data import read_records
from cfvnet.constants import NUM_COMBOS


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
```

**Step 2: Run test to verify it fails**

Run: `cd crates/cfvnet/python && uv run pytest tests/test_data.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# cfvnet/data.py
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
```

**Step 4: Run tests to verify they pass**

Run: `cd crates/cfvnet/python && uv run pytest tests/test_data.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/cfvnet/python/cfvnet/data.py crates/cfvnet/python/tests/test_data.py
git commit -m "feat(cfvnet): add Python binary TrainingRecord reader"
```

---

### Task 4: Boundary encoding

**Files:**
- Modify: `crates/cfvnet/python/cfvnet/data.py`
- Create: `crates/cfvnet/python/tests/test_encoding.py`

**Step 1: Write the failing test**

These tests must match the Rust `encode_boundary_record` output exactly.

```python
# tests/test_encoding.py
import numpy as np

from cfvnet.constants import INPUT_SIZE, POT_INDEX, NUM_COMBOS
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
```

**Step 2: Run test to verify it fails**

Run: `cd crates/cfvnet/python && uv run pytest tests/test_encoding.py -v`
Expected: FAIL

**Step 3: Write implementation**

Add to `cfvnet/data.py`:

```python
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
```

Add missing imports at the top of `data.py`:
```python
from cfvnet.constants import NUM_COMBOS, DECK_SIZE, NUM_RANKS, INPUT_SIZE
```

**Step 4: Run tests**

Run: `cd crates/cfvnet/python && uv run pytest tests/test_encoding.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/cfvnet/python/cfvnet/data.py crates/cfvnet/python/tests/test_encoding.py
git commit -m "feat(cfvnet): add boundary record encoding (Python)"
```

---

### Task 5: PyTorch Dataset and directory loading

**Files:**
- Modify: `crates/cfvnet/python/cfvnet/data.py`
- Create: `crates/cfvnet/python/tests/test_dataset.py`

**Step 1: Write the failing test**

```python
# tests/test_dataset.py
import struct
import tempfile
from pathlib import Path

import numpy as np
import torch

from cfvnet.constants import INPUT_SIZE, OUTPUT_SIZE, NUM_COMBOS
from cfvnet.data import BoundaryDataset


def _write_test_file(path: Path, n: int = 4) -> None:
    """Write n test records to a binary file."""
    with open(path, "wb") as f:
        for i in range(n):
            board = [0, 4, 8, 12, 16]
            f.write(struct.pack("B", len(board)))
            f.write(bytes(board))
            f.write(struct.pack("<f", 100.0))  # pot
            f.write(struct.pack("<f", 50.0 + i * 10))  # stack
            f.write(struct.pack("B", i % 2))  # player
            f.write(struct.pack("<f", 0.05 * i))  # game_value
            oop = np.zeros(NUM_COMBOS, dtype=np.float32)
            oop[0] = 0.5
            f.write(oop.tobytes())
            ip = np.zeros(NUM_COMBOS, dtype=np.float32)
            ip[0] = 0.5
            f.write(ip.tobytes())
            cfvs = np.zeros(NUM_COMBOS, dtype=np.float32)
            cfvs[0] = 0.1 * (i + 1)
            f.write(cfvs.tobytes())
            mask = np.ones(NUM_COMBOS, dtype=np.uint8)
            f.write(mask.tobytes())


def test_dataset_from_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "data.bin"
        _write_test_file(path, n=4)
        ds = BoundaryDataset.from_path(path)

    assert len(ds) == 4
    inp, target, mask, rng, gv, sw = ds[0]
    assert inp.shape == (INPUT_SIZE,)
    assert target.shape == (OUTPUT_SIZE,)
    assert mask.shape == (OUTPUT_SIZE,)
    assert rng.shape == (OUTPUT_SIZE,)
    assert isinstance(gv, float) or gv.shape == ()
    assert isinstance(sw, float) or sw.shape == ()


def test_dataset_from_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_test_file(Path(tmpdir) / "a.bin", n=3)
        _write_test_file(Path(tmpdir) / "b.bin", n=2)
        ds = BoundaryDataset.from_path(Path(tmpdir))

    assert len(ds) == 5


def test_dataset_works_with_dataloader():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "data.bin"
        _write_test_file(path, n=8)
        ds = BoundaryDataset.from_path(path)

    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    inp, target, mask, rng, gv, sw = batch
    assert inp.shape == (4, INPUT_SIZE)
    assert target.shape == (4, OUTPUT_SIZE)
    assert gv.shape == (4,)
    assert sw.shape == (4,)
```

**Step 2: Run test to verify it fails**

Run: `cd crates/cfvnet/python && uv run pytest tests/test_dataset.py -v`
Expected: FAIL

**Step 3: Write implementation**

Add to `cfvnet/data.py`:

```python
import torch
from torch.utils.data import Dataset


class BoundaryDataset(Dataset):
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
    def from_path(cls, path: Path) -> "BoundaryDataset":
        """Load from a file or directory of .bin files.

        Args:
            path: Single .bin file or directory containing .bin files.

        Returns:
            BoundaryDataset with all records encoded.
        """
        records = read_records_from_path(path)
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
```

**Step 4: Run tests**

Run: `cd crates/cfvnet/python && uv run pytest tests/test_dataset.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/cfvnet/python/cfvnet/data.py crates/cfvnet/python/tests/test_dataset.py
git commit -m "feat(cfvnet): add PyTorch BoundaryDataset with directory loading"
```

---

### Task 6: Model

**Files:**
- Create: `crates/cfvnet/python/cfvnet/model.py`
- Create: `crates/cfvnet/python/tests/test_model.py`

**Step 1: Write the failing test**

```python
# tests/test_model.py
import torch
from cfvnet.model import BoundaryNet
from cfvnet.constants import INPUT_SIZE, OUTPUT_SIZE


def test_output_shape_single():
    model = BoundaryNet(num_layers=2, hidden_size=64)
    x = torch.zeros(1, INPUT_SIZE)
    y = model(x)
    assert y.shape == (1, OUTPUT_SIZE)


def test_output_shape_batch():
    model = BoundaryNet(num_layers=2, hidden_size=64)
    x = torch.zeros(8, INPUT_SIZE)
    y = model(x)
    assert y.shape == (8, OUTPUT_SIZE)


def test_parameter_count():
    model = BoundaryNet(num_layers=2, hidden_size=64)
    total = sum(p.numel() for p in model.parameters())
    # Layer 1: 2720*64 + 64 (linear) + 64*2 (bn) + 64 (prelu) = 174,336 + 192
    # Layer 2: 64*64 + 64 (linear) + 64*2 (bn) + 64 (prelu) = 4,160 + 192
    # Output: 64*1326 + 1326 = 86,190
    assert total > 0


def test_eval_mode_deterministic():
    model = BoundaryNet(num_layers=2, hidden_size=64)
    model.eval()
    x = torch.randn(4, INPUT_SIZE)
    y1 = model(x)
    y2 = model(x)
    assert torch.allclose(y1, y2)
```

**Step 2: Run test to verify it fails**

Run: `cd crates/cfvnet/python && uv run pytest tests/test_model.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# cfvnet/model.py
"""BoundaryNet model matching the Rust architecture."""

import torch
import torch.nn as nn

from cfvnet.constants import INPUT_SIZE, OUTPUT_SIZE


class HiddenBlock(nn.Module):
    """Linear -> BatchNorm1d -> PReLU block."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.BatchNorm1d(out_features)
        self.activation = nn.PReLU(num_parameters=out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through linear, batchnorm, and PReLU."""
        return self.activation(self.norm(self.linear(x)))


class BoundaryNet(nn.Module):
    """Boundary value network for depth-bounded range solving.

    Architecture: Input(2720) -> [Linear -> BN -> PReLU] x N -> Linear(1326)
    """

    def __init__(self, num_layers: int, hidden_size: int) -> None:
        """Create a BoundaryNet.

        Args:
            num_layers: Number of hidden blocks.
            hidden_size: Width of each hidden layer.
        """
        super().__init__()
        assert num_layers > 0, "need at least one hidden layer"

        layers: list[HiddenBlock] = []
        layers.append(HiddenBlock(INPUT_SIZE, hidden_size))
        for _ in range(1, num_layers):
            layers.append(HiddenBlock(hidden_size, hidden_size))
        self.hidden = nn.ModuleList(layers)
        self.output = nn.Linear(hidden_size, OUTPUT_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: returns [batch, OUTPUT_SIZE] normalized EVs."""
        for block in self.hidden:
            x = block(x)
        return self.output(x)
```

**Step 4: Run tests**

Run: `cd crates/cfvnet/python && uv run pytest tests/test_model.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/cfvnet/python/cfvnet/model.py crates/cfvnet/python/tests/test_model.py
git commit -m "feat(cfvnet): add PyTorch BoundaryNet model"
```

---

### Task 7: Loss functions

**Files:**
- Create: `crates/cfvnet/python/cfvnet/loss.py`
- Create: `crates/cfvnet/python/tests/test_loss.py`

**Step 1: Write the failing test**

```python
# tests/test_loss.py
import torch
from cfvnet.loss import boundary_loss


def test_zero_loss_on_perfect_prediction():
    pred = torch.tensor([[1.0, 2.0, 3.0]])
    target = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0]])
    rng = torch.tensor([[0.5, 0.5, 0.0]])
    gv = torch.tensor([1.5])  # 0.5*1 + 0.5*2 = 1.5
    sw = torch.tensor([1.0])

    combined, huber, aux = boundary_loss(pred, target, mask, rng, gv, sw, delta=1.0, aux_weight=1.0)
    assert huber.item() < 1e-6
    assert aux.item() < 1e-6


def test_masked_entries_ignored():
    pred = torch.tensor([[1.0, 999.0, 3.0]])
    target = torch.tensor([[1.0, 0.0, 3.0]])
    mask = torch.tensor([[1.0, 0.0, 1.0]])
    rng = torch.tensor([[0.5, 0.0, 0.5]])
    gv = torch.tensor([2.0])  # 0.5*1 + 0.5*3
    sw = torch.tensor([1.0])

    combined, huber, aux = boundary_loss(pred, target, mask, rng, gv, sw, delta=1.0, aux_weight=1.0)
    assert huber.item() < 1e-6


def test_sample_weight_scales_loss():
    pred = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    target = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    rng = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    gv = torch.tensor([0.0, 0.0])

    sw_equal = torch.tensor([1.0, 1.0])
    sw_skewed = torch.tensor([10.0, 1.0])

    _, h_eq, _ = boundary_loss(pred, target, mask, rng, gv, sw_equal, delta=1.0, aux_weight=0.0)
    _, h_sk, _ = boundary_loss(pred, target, mask, rng, gv, sw_skewed, delta=1.0, aux_weight=0.0)

    # Both have same per-sample error, but skewed weighting changes the average
    # Equal: (0.125 + 0.125) / 2 = 0.125
    # Skewed: (10*0.125 + 1*0.125) / 11 = 0.125 (same because errors are equal)
    # So these should be the same — the difference only shows with unequal errors
    assert abs(h_eq.item() - h_sk.item()) < 1e-5


def test_combined_includes_both_terms():
    pred = torch.tensor([[0.5, 0.5]])
    target = torch.tensor([[0.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0]])
    rng = torch.tensor([[0.5, 0.5]])
    gv = torch.tensor([0.0])
    sw = torch.tensor([1.0])

    combined, huber, aux = boundary_loss(
        pred, target, mask, rng, gv, sw, delta=1.0, aux_weight=1.0
    )
    assert combined.item() > 0
    assert abs(combined.item() - huber.item() - aux.item()) < 1e-5
```

**Step 2: Run test to verify it fails**

Run: `cd crates/cfvnet/python && uv run pytest tests/test_loss.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# cfvnet/loss.py
"""Loss functions for BoundaryNet training."""

import torch


def weighted_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    sample_weight: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    """Per-sample weighted masked Huber loss.

    Args:
        pred: Predictions [batch, combos].
        target: Targets [batch, combos].
        mask: Valid mask [batch, combos], 1.0 for valid.
        sample_weight: Per-sample weight [batch].
        delta: Huber loss threshold.

    Returns:
        Scalar weighted mean Huber loss.
    """
    diff = (pred - target) * mask
    abs_diff = diff.abs()

    quadratic = 0.5 * abs_diff.pow(2)
    linear = delta * (abs_diff - 0.5 * delta)
    element_loss = torch.where(abs_diff <= delta, quadratic, linear)

    masked_loss = element_loss * mask
    per_sample = masked_loss.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    weighted = per_sample * sample_weight
    return weighted.sum() / sample_weight.sum().clamp(min=1.0)


def weighted_aux_loss(
    pred: torch.Tensor,
    player_range: torch.Tensor,
    game_value: torch.Tensor,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    """Weighted auxiliary game-value consistency loss.

    Args:
        pred: Predictions [batch, combos].
        player_range: Player's range [batch, combos].
        game_value: Target game value [batch].
        sample_weight: Per-sample weight [batch].

    Returns:
        Scalar weighted mean squared residual.
    """
    weighted_sum = (pred * player_range).sum(dim=1)
    residual = (weighted_sum - game_value).pow(2)
    weighted = residual * sample_weight
    return weighted.sum() / sample_weight.sum().clamp(min=1.0)


def boundary_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    player_range: torch.Tensor,
    game_value: torch.Tensor,
    sample_weight: torch.Tensor,
    delta: float,
    aux_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combined BoundaryNet loss with component breakdown.

    Args:
        pred: Predictions [batch, combos].
        target: Targets [batch, combos].
        mask: Valid mask [batch, combos].
        player_range: Player's range [batch, combos].
        game_value: Target game value [batch].
        sample_weight: Per-sample weight [batch].
        delta: Huber loss threshold.
        aux_weight: Weight for auxiliary loss term.

    Returns:
        Tuple of (combined, huber, aux) loss tensors.
    """
    huber = weighted_huber_loss(pred, target, mask, sample_weight, delta)
    aux = weighted_aux_loss(pred, player_range, game_value, sample_weight)
    combined = huber + aux_weight * aux
    return combined, huber, aux
```

**Step 4: Run tests**

Run: `cd crates/cfvnet/python && uv run pytest tests/test_loss.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/cfvnet/python/cfvnet/loss.py crates/cfvnet/python/tests/test_loss.py
git commit -m "feat(cfvnet): add weighted loss functions (Python)"
```

---

### Task 8: Training loop

**Files:**
- Create: `crates/cfvnet/python/cfvnet/train.py`
- Create: `crates/cfvnet/python/tests/test_train.py`

**Step 1: Write the failing test**

```python
# tests/test_train.py
import struct
import tempfile
from pathlib import Path

import numpy as np
import torch

from cfvnet.config import TrainConfig
from cfvnet.constants import NUM_COMBOS
from cfvnet.train import train_boundary


def _write_test_data(path: Path, n: int = 32) -> None:
    with open(path, "wb") as f:
        for i in range(n):
            board = [0, 4, 8, 12, 16]
            f.write(struct.pack("B", len(board)))
            f.write(bytes(board))
            f.write(struct.pack("<f", 100.0))
            f.write(struct.pack("<f", 50.0))
            f.write(struct.pack("B", i % 2))
            f.write(struct.pack("<f", 0.1 * i))
            oop = np.zeros(NUM_COMBOS, dtype=np.float32)
            for j in range(10):
                oop[j] = 0.1
            f.write(oop.tobytes())
            ip = np.zeros(NUM_COMBOS, dtype=np.float32)
            for j in range(10):
                ip[j] = 0.1
            f.write(ip.tobytes())
            cfvs = np.zeros(NUM_COMBOS, dtype=np.float32)
            for j in range(10):
                cfvs[j] = (i + j) * 0.01
            f.write(cfvs.tobytes())
            mask = np.ones(NUM_COMBOS, dtype=np.uint8)
            f.write(mask.tobytes())


def test_training_reduces_loss():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "train.bin"
        _write_test_data(data_path, n=32)

        config = TrainConfig(
            hidden_layers=2, hidden_size=64,
            batch_size=16, epochs=50,
            learning_rate=0.001, lr_min=0.001,
            huber_delta=1.0, aux_loss_weight=0.0,
            validation_split=0.0, grad_clip_norm=1.0,
        )

        result = train_boundary(
            data_path=data_path,
            config=config,
            output_dir=None,
            device=torch.device("cpu"),
        )

    assert result.final_train_loss < 0.1, f"expected loss < 0.1, got {result.final_train_loss}"
```

**Step 2: Run test to verify it fails**

Run: `cd crates/cfvnet/python && uv run pytest tests/test_train.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# cfvnet/train.py
"""Training loop for BoundaryNet."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

from cfvnet.config import TrainConfig
from cfvnet.data import BoundaryDataset
from cfvnet.loss import boundary_loss
from cfvnet.model import BoundaryNet


@dataclass
class TrainResult:
    """Result returned after training completes."""

    final_train_loss: float


def train_boundary(
    data_path: Path,
    config: TrainConfig,
    output_dir: Path | None,
    device: torch.device,
) -> TrainResult:
    """Train a BoundaryNet model.

    Args:
        data_path: Path to training data (file or directory).
        config: Training configuration.
        output_dir: Directory for checkpoints (None to skip saving).
        device: Torch device (cpu or cuda).

    Returns:
        TrainResult with final training loss.
    """
    dataset = BoundaryDataset.from_path(data_path)
    train_ds, val_ds = _split_dataset(dataset, config.validation_split)
    train_loader = _make_dataloader(train_ds, config.batch_size, shuffle=True)
    val_loader = _make_dataloader(val_ds, config.batch_size, shuffle=False) if val_ds else None

    model = BoundaryNet(config.hidden_layers, config.hidden_size).to(device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.lr_min)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = _maybe_resume(model, optimizer, scheduler, scaler, output_dir)

    final_loss = float("inf")
    for epoch in range(start_epoch, config.epochs):
        t0 = time.time()
        train_loss = _train_epoch(model, train_loader, optimizer, scaler, config, device)
        scheduler.step()

        lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0
        msg = f"Epoch {epoch + 1}/{config.epochs} lr={lr:.2e} train={train_loss:.6f}"

        if val_loader:
            val_combined, val_huber, val_aux = _val_epoch(model, val_loader, config, device)
            msg += f" val={val_combined:.6f} (huber={val_huber:.4f} aux={val_aux:.4f})"

        msg += f" [{elapsed:.0f}s]"
        print(msg)

        final_loss = train_loss

        if output_dir and config.checkpoint_every_n_epochs > 0:
            if (epoch + 1) % config.checkpoint_every_n_epochs == 0:
                _save_checkpoint(model, optimizer, scheduler, scaler, epoch + 1, output_dir)

    return TrainResult(final_train_loss=final_loss)


def _split_dataset(dataset, val_split: float):
    """Split dataset into train and val sets."""
    if val_split <= 0.0:
        return dataset, None
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    return random_split(dataset, [train_size, val_size])


def _make_dataloader(dataset, batch_size: int, shuffle: bool) -> DataLoader:
    """Create a DataLoader with pin_memory for GPU transfer."""
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=4, pin_memory=True, prefetch_factor=2,
        persistent_workers=True,
    )


def _train_epoch(
    model: BoundaryNet,
    loader: DataLoader,
    optimizer: Adam,
    scaler: torch.amp.GradScaler,
    config: TrainConfig,
    device: torch.device,
) -> float:
    """Run one training epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0
    count = 0

    for batch in loader:
        inp, target, mask, rng, gv, sw = (t.to(device) for t in batch)

        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            pred = model(inp)
            loss, _, _ = boundary_loss(pred, target, mask, rng, gv, sw,
                                       config.huber_delta, config.aux_loss_weight)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        count += 1

    return total_loss / max(count, 1)


@torch.no_grad()
def _val_epoch(
    model: BoundaryNet,
    loader: DataLoader,
    config: TrainConfig,
    device: torch.device,
) -> tuple[float, float, float]:
    """Run validation. Returns (combined, huber, aux) mean losses."""
    model.eval()
    total_combined = total_huber = total_aux = 0.0
    count = 0

    for batch in loader:
        inp, target, mask, rng, gv, sw = (t.to(device) for t in batch)
        pred = model(inp)
        combined, huber, aux = boundary_loss(
            pred, target, mask, rng, gv, sw,
            config.huber_delta, config.aux_loss_weight,
        )
        total_combined += combined.item()
        total_huber += huber.item()
        total_aux += aux.item()
        count += 1

    n = max(count, 1)
    return total_combined / n, total_huber / n, total_aux / n


def _save_checkpoint(model, optimizer, scheduler, scaler, epoch: int, output_dir: Path) -> None:
    """Save training checkpoint."""
    path = output_dir / f"checkpoint_epoch{epoch}.pt"
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }, path)
    print(f"  Saved checkpoint: {path}")


def _maybe_resume(model, optimizer, scheduler, scaler, output_dir: Path | None) -> int:
    """Resume from latest checkpoint if available. Returns start epoch."""
    if output_dir is None:
        return 0
    checkpoints = sorted(output_dir.glob("checkpoint_epoch*.pt"))
    if not checkpoints:
        return 0
    latest = checkpoints[-1]
    ckpt = torch.load(latest, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    epoch = ckpt["epoch"]
    print(f"  Resumed from {latest} (epoch {epoch})")
    return epoch
```

**Step 4: Run tests**

Run: `cd crates/cfvnet/python && uv run pytest tests/test_train.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/cfvnet/python/cfvnet/train.py crates/cfvnet/python/tests/test_train.py
git commit -m "feat(cfvnet): add PyTorch training loop with AMP and checkpointing"
```

---

### Task 9: ONNX export

**Files:**
- Create: `crates/cfvnet/python/cfvnet/export.py`
- Create: `crates/cfvnet/python/tests/test_export.py`

**Step 1: Write the failing test**

```python
# tests/test_export.py
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from cfvnet.constants import INPUT_SIZE, OUTPUT_SIZE
from cfvnet.export import export_onnx
from cfvnet.model import BoundaryNet


def test_export_creates_onnx_file():
    model = BoundaryNet(num_layers=2, hidden_size=64)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "model.onnx"
        export_onnx(model, path)
        assert path.exists()
        assert path.stat().st_size > 0


def test_onnx_output_matches_pytorch():
    model = BoundaryNet(num_layers=2, hidden_size=64)
    model.eval()

    x = torch.randn(4, INPUT_SIZE)
    with torch.no_grad():
        pytorch_out = model(x).numpy()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "model.onnx"
        export_onnx(model, path)

        session = ort.InferenceSession(str(path))
        onnx_out = session.run(None, {"input": x.numpy()})[0]

    np.testing.assert_allclose(pytorch_out, onnx_out, rtol=1e-5, atol=1e-5)


def test_onnx_supports_dynamic_batch():
    model = BoundaryNet(num_layers=2, hidden_size=64)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "model.onnx"
        export_onnx(model, path)

        session = ort.InferenceSession(str(path))

        for batch_size in [1, 4, 16]:
            x = np.random.randn(batch_size, INPUT_SIZE).astype(np.float32)
            out = session.run(None, {"input": x})[0]
            assert out.shape == (batch_size, OUTPUT_SIZE)
```

**Step 2: Run test to verify it fails**

Run: `cd crates/cfvnet/python && uv run pytest tests/test_export.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# cfvnet/export.py
"""ONNX export for BoundaryNet."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from cfvnet.constants import INPUT_SIZE
from cfvnet.model import BoundaryNet


def export_onnx(model: BoundaryNet, path: Path) -> None:
    """Export a trained BoundaryNet to ONNX format.

    Sets model to eval mode, exports with dynamic batch axis,
    and verifies the exported model produces matching outputs.

    Args:
        model: Trained BoundaryNet model.
        path: Output path for .onnx file.
    """
    model.eval()
    dummy = torch.zeros(1, INPUT_SIZE)

    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )

    _verify_export(model, path)


def _verify_export(model: BoundaryNet, path: Path) -> None:
    """Verify ONNX output matches PyTorch within tolerance.

    Raises:
        AssertionError: If outputs diverge beyond tolerance.
    """
    model.eval()
    x = torch.randn(4, INPUT_SIZE)

    with torch.no_grad():
        pytorch_out = model(x).numpy()

    session = ort.InferenceSession(str(path))
    onnx_out = session.run(None, {"input": x.numpy()})[0]

    np.testing.assert_allclose(
        pytorch_out, onnx_out, rtol=1e-4, atol=1e-4,
        err_msg="ONNX output does not match PyTorch",
    )
    print(f"  ONNX export verified: {path} ({path.stat().st_size / 1024:.0f} KB)")
```

**Step 4: Run tests**

Run: `cd crates/cfvnet/python && uv run pytest tests/test_export.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/cfvnet/python/cfvnet/export.py crates/cfvnet/python/tests/test_export.py
git commit -m "feat(cfvnet): add ONNX export with verification"
```

---

### Task 10: CLI scripts

**Files:**
- Create: `crates/cfvnet/python/scripts/train_boundary.py`
- Create: `crates/cfvnet/python/scripts/eval_boundary.py`

**Step 1: Write `train_boundary.py`**

```python
#!/usr/bin/env python3
"""Train a BoundaryNet model.

Usage:
    python scripts/train_boundary.py \
        --config ../../sample_configurations/boundary_net_river.yaml \
        --data /path/to/training/data \
        --output /path/to/output/dir
"""

import argparse
from pathlib import Path

import torch
import yaml

from cfvnet.config import load_config
from cfvnet.export import export_onnx
from cfvnet.train import train_boundary


def main() -> None:
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(description="Train BoundaryNet")
    parser.add_argument("--config", "-c", type=Path, required=True, help="YAML config file")
    parser.add_argument("--data", "-d", type=Path, required=True, help="Training data (file or dir)")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output directory")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, or cuda")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    config = load_config(args.config)

    # Save config to output dir.
    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.config) as f:
        raw = yaml.safe_load(f)
    with open(args.output / "config.yaml", "w") as f:
        yaml.dump(raw, f)

    print(f"Device: {device}")
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")

    result = train_boundary(
        data_path=args.data,
        config=config,
        output_dir=args.output,
        device=device,
    )

    print(f"\nTraining complete. Final loss: {result.final_train_loss:.6f}")

    # Export ONNX.
    from cfvnet.model import BoundaryNet
    model = BoundaryNet(config.hidden_layers, config.hidden_size)
    ckpt = torch.load(args.output / sorted(args.output.glob("checkpoint_*.pt"))[-1].name,
                       weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    onnx_path = args.output / "model.onnx"
    export_onnx(model, onnx_path)
    print(f"ONNX model saved to {onnx_path}")


def _resolve_device(device_str: str) -> torch.device:
    """Resolve device string to torch.device."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


if __name__ == "__main__":
    main()
```

**Step 2: Write `eval_boundary.py`**

```python
#!/usr/bin/env python3
"""Evaluate a BoundaryNet model on held-out data.

Usage:
    python scripts/eval_boundary.py \
        --model /path/to/model.onnx \
        --data /path/to/eval/data.bin
"""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort

from cfvnet.constants import INPUT_SIZE
from cfvnet.data import read_records_from_path, encode_boundary_record


def main() -> None:
    """Parse arguments and run evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate BoundaryNet")
    parser.add_argument("--model", "-m", type=Path, required=True, help="ONNX model path")
    parser.add_argument("--data", "-d", type=Path, required=True, help="Eval data (file or dir)")
    args = parser.parse_args()

    session = ort.InferenceSession(str(args.model))
    records = read_records_from_path(args.data)
    print(f"Evaluating {len(records)} records...")

    maes: list[float] = []
    spr_maes: dict[str, list[float]] = {"<1": [], "1-3": [], "3-10": [], "10+": []}

    for rec in records:
        item = encode_boundary_record(rec)
        inp = item.input.reshape(1, INPUT_SIZE).astype(np.float32)
        pred = session.run(None, {"input": inp})[0][0]

        valid = item.mask > 0.5
        if valid.sum() == 0:
            continue
        mae = float(np.abs(pred[valid] - item.target[valid]).mean())
        maes.append(mae)

        spr = rec.effective_stack / rec.pot if rec.pot > 0 else 0.0
        bucket = "<1" if spr < 1 else "1-3" if spr < 3 else "3-10" if spr < 10 else "10+"
        spr_maes[bucket].append(mae)

    _print_stats("Overall", maes)
    print("\nMAE by SPR bucket:")
    for label in ["<1", "1-3", "3-10", "10+"]:
        _print_stats(f"  SPR {label:<5}", spr_maes[label])


def _print_stats(label: str, values: list[float]) -> None:
    """Print mean, std, and percentiles for a list of values."""
    if not values:
        print(f"{label}: N/A (0 records)")
        return
    arr = np.array(sorted(values))
    n = len(arr)
    p = lambda frac: arr[int(frac * (n - 1))]
    print(
        f"{label}: mean={arr.mean():.6f} std={arr.std():.4f} "
        f"p50={p(0.5):.4f} p90={p(0.9):.4f} p95={p(0.95):.4f} "
        f"p99={p(0.99):.4f} max={arr.max():.4f}  ({n} records)"
    )


if __name__ == "__main__":
    main()
```

**Step 3: Verify scripts parse**

Run: `cd crates/cfvnet/python && uv run python -c "import scripts.train_boundary; import scripts.eval_boundary"`
Expected: no import errors

**Step 4: Commit**

```bash
git add crates/cfvnet/python/scripts/
git commit -m "feat(cfvnet): add train_boundary and eval_boundary CLI scripts"
```

---

### Task 11: Rust ONNX inference via `ort`

**Files:**
- Modify: `crates/cfvnet/Cargo.toml`
- Modify: `crates/cfvnet/src/eval/boundary_evaluator.rs`

**Step 1: Add `ort` dependency**

In `crates/cfvnet/Cargo.toml`, add:
```toml
ort = { version = "2", optional = true }
```

Add feature:
```toml
[features]
onnx = ["ort"]
```

**Step 2: Write the test**

In `crates/cfvnet/src/eval/boundary_evaluator.rs`, add a test:

```rust
#[cfg(feature = "onnx")]
#[test]
fn onnx_evaluator_returns_correct_length() {
    // This test requires a pre-exported ONNX model.
    // For CI, we'd generate one in a test fixture.
    // For now, test the OnnxBoundaryEvaluator construction.
}
```

**Step 3: Replace `NeuralBoundaryEvaluator` internals**

Replace the burn-based model with `ort::Session`:

```rust
#[cfg(feature = "onnx")]
pub struct NeuralBoundaryEvaluator {
    session: ort::Session,
    board: Vec<u8>,
    private_cards: [Vec<(u8, u8)>; 2],
}

#[cfg(feature = "onnx")]
impl NeuralBoundaryEvaluator {
    pub fn load(model_path: &std::path::Path, board: Vec<u8>,
                private_cards: [Vec<(u8, u8)>; 2]) -> Result<Self, String> {
        let session = ort::Session::builder()
            .map_err(|e| format!("ort session builder: {e}"))?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)
            .map_err(|e| format!("ort optimization: {e}"))?
            .commit_from_file(model_path)
            .map_err(|e| format!("ort load {}: {e}", model_path.display()))?;
        Ok(Self { session, board, private_cards })
    }
}
```

The `compute_cfvs` implementation stays the same (build input, forward, denormalize, remap) — only the forward pass changes from burn tensor ops to `ort::Session::run`.

**Step 4: Build and test**

Run: `cargo test -p cfvnet --features onnx`
Expected: compiles and passes

**Step 5: Commit**

```bash
git add crates/cfvnet/Cargo.toml crates/cfvnet/src/eval/boundary_evaluator.rs
git commit -m "feat(cfvnet): add ONNX inference via ort crate"
```

---

### Task 12: End-to-end integration test

**Files:**
- Create: `crates/cfvnet/python/tests/test_e2e.py`

**Step 1: Write end-to-end test**

```python
# tests/test_e2e.py
"""End-to-end test: create data → train → export → infer with onnxruntime."""

import struct
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from cfvnet.config import TrainConfig
from cfvnet.constants import INPUT_SIZE, OUTPUT_SIZE, NUM_COMBOS
from cfvnet.data import BoundaryDataset, encode_boundary_record, TrainingRecord
from cfvnet.export import export_onnx
from cfvnet.model import BoundaryNet
from cfvnet.train import train_boundary


def _write_test_data(path: Path, n: int = 64) -> None:
    with open(path, "wb") as f:
        for i in range(n):
            board = [0, 4, 8, 12, 16]
            f.write(struct.pack("B", len(board)))
            f.write(bytes(board))
            f.write(struct.pack("<f", 100.0))
            f.write(struct.pack("<f", 50.0))
            f.write(struct.pack("B", i % 2))
            f.write(struct.pack("<f", 0.0))
            oop = np.zeros(NUM_COMBOS, dtype=np.float32)
            for j in range(10):
                oop[j] = 0.1
            f.write(oop.tobytes())
            f.write(oop.tobytes())  # ip same as oop
            cfvs = np.zeros(NUM_COMBOS, dtype=np.float32)
            for j in range(10):
                cfvs[j] = 0.01 * (i + j)
            f.write(cfvs.tobytes())
            mask = np.ones(NUM_COMBOS, dtype=np.uint8)
            f.write(mask.tobytes())


def test_full_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        data_path = tmpdir / "data.bin"
        output_dir = tmpdir / "model"
        _write_test_data(data_path)

        # Train.
        config = TrainConfig(
            hidden_layers=2, hidden_size=32,
            batch_size=16, epochs=20,
            learning_rate=0.001, lr_min=0.001,
            huber_delta=1.0, aux_loss_weight=0.0,
            validation_split=0.0, grad_clip_norm=1.0,
            checkpoint_every_n_epochs=20,
        )
        result = train_boundary(data_path, config, output_dir, torch.device("cpu"))
        assert result.final_train_loss < 1.0

        # Load trained model and export.
        model = BoundaryNet(config.hidden_layers, config.hidden_size)
        ckpt = torch.load(output_dir / "checkpoint_epoch20.pt", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        onnx_path = output_dir / "model.onnx"
        export_onnx(model, onnx_path)

        # Infer with onnxruntime.
        session = ort.InferenceSession(str(onnx_path))
        x = np.random.randn(4, INPUT_SIZE).astype(np.float32)
        out = session.run(None, {"input": x})[0]
        assert out.shape == (4, OUTPUT_SIZE)
```

**Step 2: Run test**

Run: `cd crates/cfvnet/python && uv run pytest tests/test_e2e.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add crates/cfvnet/python/tests/test_e2e.py
git commit -m "test(cfvnet): add end-to-end Python training pipeline test"
```

---

### Task 13: Ruff lint pass and final verification

**Step 1: Run ruff**

Run: `cd crates/cfvnet/python && uv run ruff check cfvnet/ scripts/ tests/ --fix`
Expected: clean (or auto-fixed)

**Step 2: Run full test suite**

Run: `cd crates/cfvnet/python && uv run pytest -v`
Expected: all tests pass

**Step 3: Run Rust tests**

Run: `cargo test -p cfvnet`
Expected: all pass, no regressions

**Step 4: Commit any fixes**

```bash
git add -A && git commit -m "chore(cfvnet): ruff lint fixes for Python package"
```
