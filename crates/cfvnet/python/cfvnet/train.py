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
    num_workers: int = 4,
) -> TrainResult:
    """Train a BoundaryNet model.

    Args:
        data_path: Path to training data (file or directory).
        config: Training configuration.
        output_dir: Directory for checkpoints (None to skip saving).
        device: Torch device (cpu or cuda).
        num_workers: Number of DataLoader workers (0 for single-process).

    Returns:
        TrainResult with final training loss.
    """
    dataset = LazyBoundaryDataset.from_path(data_path)
    train_ds, val_ds = _split_dataset(dataset, config.validation_split)
    train_loader = _make_dataloader(train_ds, config.batch_size, shuffle=True,
                                    num_workers=num_workers)
    val_loader = (_make_dataloader(val_ds, config.batch_size, shuffle=False,
                                   num_workers=num_workers) if val_ds else None)

    model = BoundaryNet(config.hidden_layers, config.hidden_size).to(device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.lr_min)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = _maybe_resume(model, optimizer, scheduler, scaler, output_dir)

    final_loss = _run_training_loop(
        model, train_loader, val_loader, optimizer, scheduler, scaler,
        config, device, output_dir, start_epoch,
    )

    return TrainResult(final_train_loss=final_loss)


def _run_training_loop(
    model: BoundaryNet,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    optimizer: Adam,
    scheduler: CosineAnnealingLR,
    scaler: torch.amp.GradScaler,
    config: TrainConfig,
    device: torch.device,
    output_dir: Path | None,
    start_epoch: int,
) -> float:
    """Execute the epoch loop. Returns the final training loss."""
    final_loss = float("inf")

    for epoch in range(start_epoch, config.epochs):
        t0 = time.time()
        train_loss = _train_epoch(model, train_loader, optimizer, scaler, config, device)
        scheduler.step()

        msg = _format_epoch_msg(epoch, config.epochs, scheduler, train_loss, time.time() - t0)

        if val_loader:
            val_combined, val_huber, val_aux = _val_epoch(model, val_loader, config, device)
            msg += f" val={val_combined:.6f} (huber={val_huber:.4f} aux={val_aux:.4f})"

        print(msg)
        final_loss = train_loss
        _maybe_save_checkpoint(model, optimizer, scheduler, scaler, epoch, config, output_dir)

    return final_loss


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
) -> tuple[LazyBoundaryDataset, LazyBoundaryDataset | None]:
    """Split dataset into train and val sets.

    Args:
        dataset: Full dataset.
        val_split: Fraction for validation (0.0 to skip).

    Returns:
        Tuple of (train_set, val_set or None).
    """
    if val_split <= 0.0:
        return dataset, None
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    if val_size == 0:
        return dataset, None
    return random_split(dataset, [train_size, val_size])


def _make_dataloader(
    dataset: LazyBoundaryDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
) -> DataLoader:
    """Create a DataLoader with pin_memory for GPU transfer.

    Args:
        dataset: Dataset to load from.
        batch_size: Batch size.
        shuffle: Whether to shuffle.
        num_workers: Number of worker processes (0 for in-process).

    Returns:
        Configured DataLoader.
    """
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


def _save_checkpoint(
    model: BoundaryNet,
    optimizer: Adam,
    scheduler: CosineAnnealingLR,
    scaler: torch.amp.GradScaler,
    epoch: int,
    output_dir: Path,
) -> None:
    """Save training checkpoint.

    Args:
        model: Trained model.
        optimizer: Adam optimizer.
        scheduler: LR scheduler.
        scaler: AMP gradient scaler.
        epoch: Current epoch (1-indexed, already completed).
        output_dir: Directory to save checkpoint.
    """
    path = output_dir / f"checkpoint_epoch{epoch}.pt"
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }, path)
    print(f"  Saved checkpoint: {path}")


def _maybe_save_checkpoint(
    model: BoundaryNet,
    optimizer: Adam,
    scheduler: CosineAnnealingLR,
    scaler: torch.amp.GradScaler,
    epoch: int,
    config: TrainConfig,
    output_dir: Path | None,
) -> None:
    """Save checkpoint if conditions are met."""
    if not output_dir or config.checkpoint_every_n_epochs <= 0:
        return
    if (epoch + 1) % config.checkpoint_every_n_epochs == 0:
        _save_checkpoint(model, optimizer, scheduler, scaler, epoch + 1, output_dir)


def _maybe_resume(
    model: BoundaryNet,
    optimizer: Adam,
    scheduler: CosineAnnealingLR,
    scaler: torch.amp.GradScaler,
    output_dir: Path | None,
) -> int:
    """Resume from latest checkpoint if available.

    Args:
        model: Model to load weights into.
        optimizer: Optimizer to restore state.
        scheduler: LR scheduler to restore state.
        scaler: AMP scaler to restore state.
        output_dir: Directory to search for checkpoints.

    Returns:
        Start epoch number (0 if no checkpoint found).
    """
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
