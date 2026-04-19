"""Training loop for BoundaryNet."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

from cfvnet.config import TrainConfig
from cfvnet.data import LazyBoundaryDataset
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
    num_workers: int = 8,
    gpu_buffer_size: int = 1_000_000,
) -> TrainResult:
    """Train a BoundaryNet model.

    On CUDA, uses a GPU ring buffer for zero-transfer training. On CPU,
    falls back to DataLoader with lazy dataset.

    Args:
        data_path: Path to training data (file or directory).
        config: Training configuration.
        output_dir: Directory for checkpoints (None to skip saving).
        device: Torch device (cpu or cuda).
        num_workers: Workers for DataLoader (CPU) or refill threads (CUDA).
        gpu_buffer_size: Number of records in GPU ring buffer (CUDA only).

    Returns:
        TrainResult with final training loss.
    """
    model = BoundaryNet(config.hidden_layers, config.hidden_size).to(device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.lr_min)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = _maybe_resume(model, optimizer, scheduler, scaler, output_dir)

    if device.type == "cuda":
        final_loss = _train_with_gpu_buffer(
            model, optimizer, scheduler, scaler, config, device,
            data_path, output_dir, start_epoch, gpu_buffer_size, num_workers,
        )
    else:
        final_loss = _train_with_dataloader(
            model, optimizer, scheduler, scaler, config, device,
            data_path, output_dir, start_epoch, num_workers,
        )

    return TrainResult(final_train_loss=final_loss)


# ---------------------------------------------------------------------------
# GPU ring buffer training path
# ---------------------------------------------------------------------------


def _train_with_gpu_buffer(
    model: BoundaryNet,
    optimizer: Adam,
    scheduler: CosineAnnealingLR,
    scaler: torch.amp.GradScaler,
    config: TrainConfig,
    device: torch.device,
    data_path: Path,
    output_dir: Path | None,
    start_epoch: int,
    buffer_size: int,
    num_workers: int,
) -> float:
    """Train using GPU ring buffer — zero CPU-GPU transfer per batch."""
    import threading
    from cfvnet.gpu_buffer import GpuRingBuffer

    # Refresh 10% of the pool per epoch, overlapped with training.
    refresh_count = buffer_size // 10

    buf = GpuRingBuffer(data_path, capacity=buffer_size, device=device,
                        num_workers=num_workers)

    # Steps per epoch = total records / batch_size (approximate).
    steps_per_epoch = max(buf._total_records // config.batch_size, 1)

    final_loss = float("inf")
    best_val_huber = float("inf")
    prep_thread: threading.Thread | None = None
    pending_ctx: dict | None = None
    prep_lock = threading.Lock()
    last_prep_time = 0.0

    def _bg_prepare():
        """Background: read records from disk into CPU buffer."""
        nonlocal pending_ctx, last_prep_time
        t0 = time.time()
        ctx = buf.prepare_refresh(refresh_count)
        last_prep_time = time.time() - t0
        with prep_lock:
            pending_ctx = ctx

    # Start first prep immediately.
    prep_thread = threading.Thread(target=_bg_prepare, daemon=True)
    prep_thread.start()

    for epoch in range(start_epoch, config.epochs):
        t0 = time.time()
        train_combined, train_huber, train_aux = _train_epoch_buffer(
            model, buf, optimizer, scaler, config, device, steps_per_epoch,
        )
        scheduler.step()

        val_combined, val_huber, val_aux = _val_from_buffer(
            model, buf, config, device, num_val_batches=10,
        )

        train_elapsed = time.time() - t0

        # Apply any ready refresh (fast GPU copy) and start next prep.
        refresh_status = ""
        with prep_lock:
            ctx = pending_ctx
            pending_ctx = None
        if ctx is not None:
            buf.apply_refresh(ctx)
            pct = 100.0 * refresh_count / buf._capacity
            refresh_status = f" refresh={pct:.0f}% [prep {last_prep_time:.0f}s]"

        # Start next prep if not already running.
        if prep_thread is None or not prep_thread.is_alive():
            prep_thread = threading.Thread(target=_bg_prepare, daemon=True)
            prep_thread.start()

        lr = scheduler.get_last_lr()[0]
        msg = (
            f"Epoch {epoch + 1}/{config.epochs} lr={lr:.2e} "
            f"train={train_combined:.6f} (h={train_huber:.4f} a={train_aux:.4f}) "
            f"val={val_combined:.6f} (h={val_huber:.4f} a={val_aux:.4f}) "
            f"[{train_elapsed:.0f}s]{refresh_status}"
        )
        print(msg)
        final_loss = train_combined

        losses = {
            "train_combined": train_combined, "train_huber": train_huber,
            "train_aux": train_aux, "val_combined": val_combined,
            "val_huber": val_huber, "val_aux": val_aux, "lr": lr,
        }
        if output_dir:
            _append_training_log(output_dir, epoch + 1, losses)
        _maybe_save_checkpoint(
            model, optimizer, scheduler, scaler, epoch, config, output_dir, losses,
        )

        # Save best model by val huber loss.
        if output_dir and val_huber < best_val_huber and not math.isnan(val_huber):
            best_val_huber = val_huber
            _save_checkpoint(
                model, optimizer, scheduler, scaler, epoch + 1, output_dir, losses,
                filename="best.pt",
            )
            print(f"  New best val_huber={val_huber:.6f} at epoch {epoch + 1}")

    if prep_thread is not None:
        prep_thread.join()

    return final_loss


def _train_epoch_buffer(
    model: BoundaryNet,
    buf: object,  # GpuRingBuffer
    optimizer: Adam,
    scaler: torch.amp.GradScaler,
    config: TrainConfig,
    device: torch.device,
    steps: int,
) -> tuple[float, float, float]:
    """Run one training epoch from GPU buffer. Returns (combined, huber, aux)."""
    model.train()
    total_combined = total_huber = total_aux = 0.0

    for step in range(steps):
        inp, target, mask, rng, gv, sw = buf.sample_batch(config.batch_size)  # type: ignore[attr-defined]

        with torch.amp.autocast(device_type=device.type):
            pred = model(inp)
            loss, huber, aux = boundary_loss(pred, target, mask, rng, gv, sw,
                                             config.huber_delta, config.aux_loss_weight)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        total_combined += loss.item()
        total_huber += huber.item()
        total_aux += aux.item()

        if (step + 1) % 50 == 0 or step + 1 == steps:
            print(f"\r  [{step + 1}/{steps}]", end="", flush=True)

    print()  # newline after progress
    n = max(steps, 1)
    return total_combined / n, total_huber / n, total_aux / n


@torch.no_grad()
def _val_from_buffer(
    model: BoundaryNet,
    buf: object,  # GpuRingBuffer
    config: TrainConfig,
    device: torch.device,
    num_val_batches: int = 10,
) -> tuple[float, float, float]:
    """Run validation by sampling from the GPU buffer. Returns (combined, huber, aux)."""
    model.eval()
    total_combined = total_huber = total_aux = 0.0

    for _ in range(num_val_batches):
        inp, target, mask, rng, gv, sw = buf.sample_batch(config.batch_size)  # type: ignore[attr-defined]
        pred = model(inp)
        combined, huber, aux = boundary_loss(
            pred, target, mask, rng, gv, sw,
            config.huber_delta, config.aux_loss_weight,
        )
        total_combined += combined.item()
        total_huber += huber.item()
        total_aux += aux.item()

    n = max(num_val_batches, 1)
    return total_combined / n, total_huber / n, total_aux / n


# ---------------------------------------------------------------------------
# DataLoader training path (CPU fallback / tests)
# ---------------------------------------------------------------------------


def _train_with_dataloader(
    model: BoundaryNet,
    optimizer: Adam,
    scheduler: CosineAnnealingLR,
    scaler: torch.amp.GradScaler,
    config: TrainConfig,
    device: torch.device,
    data_path: Path,
    output_dir: Path | None,
    start_epoch: int,
    num_workers: int,
) -> float:
    """Train using standard DataLoader — for CPU or small datasets."""
    dataset = LazyBoundaryDataset.from_path(data_path)
    train_ds, val_ds = _split_dataset(dataset, config.validation_split)
    train_loader = _make_dataloader(train_ds, config.batch_size, shuffle=True,
                                    num_workers=num_workers)
    val_loader = (_make_dataloader(val_ds, config.batch_size, shuffle=False,
                                   num_workers=num_workers) if val_ds else None)

    final_loss = float("inf")
    for epoch in range(start_epoch, config.epochs):
        t0 = time.time()
        train_loss = _train_epoch_loader(model, train_loader, optimizer, scaler, config, device)
        scheduler.step()

        msg = _format_epoch_msg(epoch, config.epochs, scheduler, train_loss, time.time() - t0)

        if val_loader:
            val_combined, val_huber, val_aux = _val_epoch(model, val_loader, config, device)
            msg += f" val={val_combined:.6f} (huber={val_huber:.4f} aux={val_aux:.4f})"

        print(msg)
        final_loss = train_loss

        losses = {"train_loss": train_loss, "lr": scheduler.get_last_lr()[0]}
        if val_loader:
            losses.update({"val_combined": val_combined, "val_huber": val_huber, "val_aux": val_aux})
        if output_dir:
            _append_training_log(output_dir, epoch + 1, losses)
        _maybe_save_checkpoint(
            model, optimizer, scheduler, scaler, epoch, config, output_dir, losses,
        )

    return final_loss


def _train_epoch_loader(
    model: BoundaryNet,
    loader: DataLoader,
    optimizer: Adam,
    scaler: torch.amp.GradScaler,
    config: TrainConfig,
    device: torch.device,
) -> float:
    """Run one training epoch from DataLoader. Returns mean loss."""
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


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def _format_epoch_msg(
    epoch: int, total_epochs: int, scheduler: CosineAnnealingLR,
    train_loss: float, elapsed: float,
) -> str:
    """Format a single epoch log line."""
    lr = scheduler.get_last_lr()[0]
    return f"Epoch {epoch + 1}/{total_epochs} lr={lr:.2e} train={train_loss:.6f} [{elapsed:.0f}s]"


def _split_dataset(
    dataset: LazyBoundaryDataset,
    val_split: float,
) -> tuple:
    """Split dataset into train and val sets."""
    if val_split <= 0.0:
        return dataset, None
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    if val_size == 0:
        return dataset, None
    return random_split(dataset, [train_size, val_size])


def _make_dataloader(
    dataset: object,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
) -> DataLoader:
    """Create a DataLoader with pin_memory for GPU transfer."""
    kwargs: dict = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
        kwargs["persistent_workers"] = True
    return DataLoader(**kwargs)


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


def _save_checkpoint(
    model: BoundaryNet,
    optimizer: Adam,
    scheduler: CosineAnnealingLR,
    scaler: torch.amp.GradScaler,
    epoch: int,
    output_dir: Path,
    losses: dict[str, float] | None = None,
    filename: str | None = None,
) -> None:
    """Save training checkpoint with optional loss values."""
    path = output_dir / (filename or f"checkpoint_epoch{epoch}.pt")
    data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }
    if losses:
        data["losses"] = losses
    torch.save(data, path)
    print(f"  Saved checkpoint: {path}")


def _maybe_save_checkpoint(
    model: BoundaryNet,
    optimizer: Adam,
    scheduler: CosineAnnealingLR,
    scaler: torch.amp.GradScaler,
    epoch: int,
    config: TrainConfig,
    output_dir: Path | None,
    losses: dict[str, float] | None = None,
) -> None:
    """Save checkpoint if conditions are met."""
    if not output_dir or config.checkpoint_every_n_epochs <= 0:
        return
    if (epoch + 1) % config.checkpoint_every_n_epochs == 0:
        _save_checkpoint(model, optimizer, scheduler, scaler, epoch + 1, output_dir, losses)


def _append_training_log(output_dir: Path, epoch: int, losses: dict[str, float]) -> None:
    """Append one row to training_log.csv."""
    log_path = output_dir / "training_log.csv"
    write_header = not log_path.exists()
    cols = sorted(losses.keys())
    with open(log_path, "a") as f:
        if write_header:
            f.write("epoch," + ",".join(cols) + "\n")
        vals = ",".join(f"{losses[c]:.6f}" for c in cols)
        f.write(f"{epoch},{vals}\n")


def _maybe_resume(
    model: BoundaryNet,
    optimizer: Adam,
    scheduler: CosineAnnealingLR,
    scaler: torch.amp.GradScaler,
    output_dir: Path | None,
) -> int:
    """Resume from latest checkpoint if available. Returns start epoch."""
    if output_dir is None:
        return 0
    checkpoints = sorted(
        output_dir.glob("checkpoint_epoch*.pt"),
        key=lambda p: int(p.stem.replace("checkpoint_epoch", "")),
    )
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
