"""
Training script for MPED-RNN skeleton-based anomaly detection.

Trains on normal skeleton sequences from CHAD (unsupervised split 1).
The model learns to reconstruct and predict normal movement patterns;
anomalous movements will produce high reconstruction/prediction error.

Usage:
    python -m src.anomaly.train --data-root data/raw/CHAD/CHAD_Meta
    python -m src.anomaly.train --data-root data/raw/CHAD/CHAD_Meta --cameras 1 3
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.anomaly.data import CHADSkeletonDataset
from src.anomaly.model import MPEDRNN

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def train(
    data_root: str,
    cameras: list[int] | None = None,
    seq_len: int = 12,
    pred_len: int = 6,
    hidden_dim: int = 128,
    num_layers: int = 2,
    batch_size: int = 256,
    epochs: int = 80,
    lr: float = 1e-3,
    recon_weight: float = 0.5,
    output_dir: str | None = None,
    device: str | None = None,
) -> Path:
    """Train MPED-RNN and save checkpoint."""
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    print(f"Using device: {device}")

    # Output directory
    if output_dir is None:
        cam_str = "_".join(str(c) for c in (cameras or [1, 2, 3, 4]))
        output_dir = PROJECT_ROOT / "models" / "anomaly" / f"cam_{cam_str}"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("Loading CHAD training data (normal sequences only)...")
    train_dataset = CHADSkeletonDataset(
        data_root=data_root,
        split="train",
        split_num=1,
        cameras=cameras,
        seq_len=seq_len,
        pred_len=pred_len,
        stride=2,
    )
    print(f"Training sequences: {len(train_dataset)}")

    if len(train_dataset) == 0:
        raise RuntimeError(
            f"No training data found in {data_root}. "
            "Download CHAD_Meta from https://github.com/TeCSAR-UNCC/CHAD"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    # Model
    model = MPEDRNN(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        pred_len=pred_len,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Training config for reproducibility
    config = {
        "cameras": cameras or [1, 2, 3, 4],
        "seq_len": seq_len,
        "pred_len": pred_len,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "recon_weight": recon_weight,
        "num_sequences": len(train_dataset),
    }

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Starting training...")

    best_loss = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_pred = 0.0
        t0 = time.time()

        for batch in train_loader:
            x = batch["input"].to(device)       # (B, seq_len, 34)
            target = batch["target"].to(device)  # (B, pred_len, 34)

            recon, pred = model(x)

            recon_loss = torch.nn.functional.mse_loss(recon, x)
            pred_loss = torch.nn.functional.mse_loss(pred, target)
            loss = recon_weight * recon_loss + (1 - recon_weight) * pred_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_pred += pred_loss.item()

        n_batches = len(train_loader)
        avg_loss = epoch_loss / n_batches
        avg_recon = epoch_recon / n_batches
        avg_pred = epoch_pred / n_batches
        elapsed = time.time() - t0

        scheduler.step(avg_loss)

        history.append({
            "epoch": epoch,
            "loss": avg_loss,
            "recon_loss": avg_recon,
            "pred_loss": avg_pred,
        })

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Loss: {avg_loss:.6f} (R:{avg_recon:.6f} P:{avg_pred:.6f}) | "
                f"{elapsed:.1f}s"
            )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), output_dir / "best_model.pt")

    # Save final model and config
    torch.save(model.state_dict(), output_dir / "final_model.pt")
    config["best_loss"] = best_loss
    config["history"] = history
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nTraining complete. Best loss: {best_loss:.6f}")
    print(f"Model saved to: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Train MPED-RNN on CHAD")
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(PROJECT_ROOT / "data" / "raw" / "CHAD" / "CHAD_Meta"),
        help="Path to CHAD_Meta directory",
    )
    parser.add_argument(
        "--cameras",
        type=int,
        nargs="+",
        default=None,
        help="Camera IDs to use (default: all)",
    )
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--pred-len", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--recon-weight", type=float, default=0.5)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()
    train(
        data_root=args.data_root,
        cameras=args.cameras,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        recon_weight=args.recon_weight,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
