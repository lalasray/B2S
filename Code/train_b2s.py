from __future__ import annotations

import argparse
import random
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from b2s.config import DataConfig, TrainConfig
from b2s.dataset import B2SDataset, save_dataset
from b2s.losses import compute_losses
from b2s.model import B2SModel


def parse_args() -> argparse.Namespace:
    """Parse trainer CLI arguments."""
    parser = argparse.ArgumentParser(description="Train the B2S prototype on synthetic data.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dataset-dir", type=str, default="Code/data")
    parser.add_argument("--regenerate", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Keep training deterministic enough for debugging and iteration."""
    random.seed(seed)
    torch.manual_seed(seed)


def ensure_dataset(data_config: DataConfig, dataset_dir: str, regenerate: bool) -> dict:
    """Generate synthetic data if the requested split files do not exist yet."""
    split_paths = {split: Path(dataset_dir) / f"{split}.pt" for split in ("train", "val", "test")}
    if regenerate or not all(path.exists() for path in split_paths.values()):
        save_dataset(data_config, dataset_dir)
    return split_paths


def make_loader(path: Path, batch_size: int, shuffle: bool) -> DataLoader:
    """Create a DataLoader for one saved split."""
    dataset = B2SDataset(torch.load(path, weights_only=True))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def move_batch(batch: dict, device: str) -> dict:
    """Move every tensor in the sample dict onto the target device."""
    return {key: value.to(device) for key, value in batch.items()}


def run_epoch(model: B2SModel, loader: DataLoader, optimizer, weights: dict, device: str, train: bool) -> dict:
    """Run one full pass over a loader and aggregate all reported metrics."""
    aggregate = {}
    model.train(train)
    for step, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        outputs = model(batch["sensor"])
        total_loss, metrics = compute_losses(outputs, batch, weights)
        if train:
            # Standard optimizer step for the weighted multi-task objective.
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        for key, value in metrics.items():
            aggregate[key] = aggregate.get(key, 0.0) + value
    return {key: value / max(len(loader), 1) for key, value in aggregate.items()}


def main() -> None:
    """Entry point for prototype training."""
    args = parse_args()
    data_config = DataConfig()
    train_config = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, device=args.device, dataset_dir=args.dataset_dir)
    set_seed(train_config.seed)

    # Create or reuse the synthetic dataset files.
    split_paths = ensure_dataset(data_config, train_config.dataset_dir, args.regenerate)
    train_loader = make_loader(split_paths["train"], train_config.batch_size, shuffle=True)
    val_loader = make_loader(split_paths["val"], train_config.batch_size, shuffle=False)

    # Build model and optimizer.
    model = B2SModel(data_config, train_config).to(train_config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay
    )

    best_val = float("inf")
    output_dir = Path("Code/checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep the checkpoint with the lowest validation loss.
    for epoch in range(1, train_config.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, train_config.weights, train_config.device, train=True)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, optimizer, train_config.weights, train_config.device, train=False)
        print(
            f"epoch={epoch} train_total={train_metrics['total']:.4f} "
            f"val_total={val_metrics['total']:.4f} train_regime={train_metrics['regime']:.4f}"
        )
        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "data_config": asdict(data_config),
                    "train_config": asdict(train_config),
                    "val_total": best_val,
                },
                output_dir / "best.pt",
            )


if __name__ == "__main__":
    main()
