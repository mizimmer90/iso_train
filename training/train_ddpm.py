"""
Lightweight training runner for the 2D DDPM.

Features:
- CLI-driven hyperparameters for quick sweeps.
- Per-run folders with config, metrics CSV, and checkpoints.
- Tracks train/val losses each epoch.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

from data.landscape import create_four_mode_landscape
from models.ddpm import DDPM, MLPScoreNetwork


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DDPM on 2D landscape")
    parser.add_argument("--run-name", type=str, default=None, help="Name for the run folder")
    parser.add_argument("--run-dir", type=str, default="runs", help="Base directory for runs")
    parser.add_argument("--device", type=str, default="cuda", help="'cuda', 'cpu', or specific device id")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Data
    parser.add_argument("--train-samples", type=int, default=20000, help="Number of training samples to draw")
    parser.add_argument("--val-samples", type=int, default=4000, help="Number of validation samples to draw")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--landscape-scale", type=float, default=2.0, help="Mode distance from origin")
    parser.add_argument("--landscape-std", type=float, default=0.5, help="Gaussian std for each mode")

    # Model
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 256, 256, 128], help="MLP hidden sizes")
    parser.add_argument("--time-embed-dim", type=int, default=128, help="Time embedding dimension")
    parser.add_argument("--num-timesteps", type=int, default=1000, help="Diffusion steps")
    parser.add_argument("--beta-start", type=float, default=1e-4, help="Noise schedule start")
    parser.add_argument("--beta-end", type=float, default=2e-2, help="Noise schedule end")

    # Optim
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient norm clip (0 to disable)")

    # Logging/checkpointing
    parser.add_argument("--save-every", type=int, default=20, help="Save checkpoint every N epochs")
    parser.add_argument("--eval-every", type=int, default=1, help="Evaluate every N epochs")
    parser.add_argument("--continuous-loss", action="store_true", help="Use continuous-time loss")

    return parser.parse_args()


def setup_run_dir(base_dir: str, run_name: str = None) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    resolved_name = run_name or f"run-{ts}"
    run_dir = Path(base_dir) / resolved_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    return run_dir


def make_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    landscape = create_four_mode_landscape(scale=args.landscape_scale, std=args.landscape_std, device="cpu")
    train_samples = landscape.sample(args.train_samples).float()
    val_samples = landscape.sample(args.val_samples).float()

    train_ds = TensorDataset(train_samples)
    val_ds = TensorDataset(val_samples)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    return train_loader, val_loader


def log_metrics(run_dir: Path, metrics: Dict) -> None:
    metrics_path = run_dir / "metrics.csv"
    is_new = not metrics_path.exists()
    headers = list(metrics.keys())
    line = ",".join(str(metrics[h]) for h in headers)
    if is_new:
        with metrics_path.open("w") as f:
            f.write(",".join(headers) + "\n")
    with metrics_path.open("a") as f:
        f.write(line + "\n")


def save_checkpoint(run_dir: Path, epoch: int, ddpm: DDPM, optim: torch.optim.Optimizer) -> None:
    ckpt = {
        "epoch": epoch,
        "model": ddpm.score_network.state_dict(),
        "optimizer": optim.state_dict(),
    }
    torch.save(ckpt, run_dir / "checkpoints" / f"epoch_{epoch:04d}.pt")


def train_epoch(ddpm: DDPM, loader: DataLoader, optim: torch.optim.Optimizer, device: torch.device, args: argparse.Namespace) -> float:
    ddpm.score_network.train()
    total_loss = 0.0
    steps = 0

    for (x,) in loader:
        x = x.to(device)
        optim.zero_grad()
        loss = ddpm.loss(x, continuous=args.continuous_loss)
        loss.backward()
        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(ddpm.score_network.parameters(), args.grad_clip)
        optim.step()

        total_loss += loss.item()
        steps += 1

    return total_loss / max(steps, 1)


@torch.no_grad()
def evaluate(ddpm: DDPM, loader: DataLoader, device: torch.device, args: argparse.Namespace) -> float:
    ddpm.score_network.eval()
    total_loss = 0.0
    steps = 0
    for (x,) in loader:
        x = x.to(device)
        loss = ddpm.loss(x, continuous=args.continuous_loss)
        total_loss += loss.item()
        steps += 1
    ddpm.score_network.train()
    return total_loss / max(steps, 1)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device if args.device != "cuda" else ("cuda" if torch.cuda.is_available() else "cpu"))
    run_dir = setup_run_dir(args.run_dir, args.run_name)

    # Persist config for reproducibility
    config_path = run_dir / "config.json"
    with config_path.open("w") as f:
        json.dump(vars(args), f, indent=2)

    train_loader, val_loader = make_dataloaders(args)

    score_network = MLPScoreNetwork(
        input_dim=2,
        hidden_dims=args.hidden_dims,
        time_embed_dim=args.time_embed_dim,
    )
    ddpm = DDPM(
        score_network,
        num_timesteps=args.num_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=str(device),
    )

    optimizer = torch.optim.Adam(ddpm.score_network.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(ddpm, train_loader, optimizer, device, args)
        do_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)
        val_loss = evaluate(ddpm, val_loader, device, args) if do_eval else float("nan")

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
        }
        log_metrics(run_dir, metrics)

        if do_eval and val_loss < best_val:
            best_val = val_loss
            save_checkpoint(run_dir, epoch, ddpm, optimizer)

        if epoch % args.save_every == 0:
            save_checkpoint(run_dir, epoch, ddpm, optimizer)

        print(f"[Epoch {epoch:04d}] train={train_loss:.4f} val={val_loss:.4f} (best {best_val:.4f})")

    print(f"Run complete. Artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
