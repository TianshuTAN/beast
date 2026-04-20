"""Train a small autoencoder on BEAST-CLS features to produce 100-d latents.

Reads a CLS-features z_trials.npz (shape (K, T, V, D_cls)), fits an MLP AE
per camera on the train-split frames, and writes a new z_trials.npz with
100-d latents in the same (K, T, V, D_out) layout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn


class MLPAutoencoder(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_latent: int, dropout: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_in),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat


def _standardize(train: np.ndarray, rest: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + 1e-6
    return (train - mean) / std, [(x - mean) / std for x in rest], mean, std


def _train_one_ae(
    train_x: np.ndarray,
    val_x: np.ndarray,
    d_latent: int,
    d_hidden: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    device: torch.device,
    log_prefix: str,
) -> MLPAutoencoder:
    model = MLPAutoencoder(train_x.shape[1], d_hidden, d_latent, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    x_train = torch.from_numpy(train_x).float().to(device)
    x_val = torch.from_numpy(val_x).float().to(device)

    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0

    n = x_train.shape[0]
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = x_train[idx]
            _, x_hat = model(xb)
            loss = nn.functional.mse_loss(x_hat, xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * xb.shape[0]
        train_loss = total / n

        model.eval()
        with torch.no_grad():
            _, vhat = model(x_val)
            val_loss = nn.functional.mse_loss(vhat, x_val).item()

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        if epoch % 10 == 0 or epoch == max_epochs - 1:
            print(f"  [{log_prefix}] epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}  best {best_val:.4f}")

        if stale >= patience:
            print(f"  [{log_prefix}] early stop at epoch {epoch} (best val {best_val:.4f})")
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def _encode(model: MLPAutoencoder, x: np.ndarray, device: torch.device, batch_size: int = 4096) -> np.ndarray:
    out: list[np.ndarray] = []
    for i in range(0, x.shape[0], batch_size):
        xb = torch.from_numpy(x[i:i + batch_size]).float().to(device)
        z, _ = model(xb)
        out.append(z.cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eid", required=True)
    ap.add_argument("--cls_latent_dir", required=True,
                    help="Parent dir containing <eid>/z_trials.npz with CLS features.")
    ap.add_argument("--output_dir", required=True,
                    help="Will write <output_dir>/<eid>/z_trials.npz")
    ap.add_argument("--d_latent", type=int, default=100)
    ap.add_argument("--d_hidden", type=int, default=384)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--max_epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    src_path = Path(args.cls_latent_dir) / args.eid / "z_trials.npz"
    data = np.load(src_path)
    z = data["z_trials_time"]  # (K, T, V, D_cls)
    trial_split = data["trial_split"]
    K, T, V, D_cls = z.shape
    n_train, n_val, n_test = [int(x) for x in trial_split]
    assert n_train + n_val + n_test == K, (trial_split, K)
    print(f"Loaded CLS z_trials: {z.shape}  split={trial_split.tolist()}")

    train_end = n_train
    val_end = n_train + n_val

    out = np.zeros((K, T, V, args.d_latent), dtype=np.float32)

    for v_idx in range(V):
        cam = ["left", "right"][v_idx] if V == 2 else f"cam{v_idx}"
        print(f"\n=== AE for camera {cam} (view {v_idx}) ===")

        train_flat = z[:train_end, :, v_idx, :].reshape(-1, D_cls)
        val_flat = z[train_end:val_end, :, v_idx, :].reshape(-1, D_cls)
        test_flat = z[val_end:, :, v_idx, :].reshape(-1, D_cls)
        all_flat = z[:, :, v_idx, :].reshape(-1, D_cls)
        print(f"  train frames {train_flat.shape[0]}  val {val_flat.shape[0]}  test {test_flat.shape[0]}")

        train_std, (val_std, test_std, all_std), mean, std = _standardize(train_flat, [val_flat, test_flat, all_flat])

        model = _train_one_ae(
            train_std, val_std,
            d_latent=args.d_latent,
            d_hidden=args.d_hidden,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            device=device,
            log_prefix=cam,
        )

        z_all = _encode(model, all_std, device)  # (K*T, d_latent)
        out[:, :, v_idx, :] = z_all.reshape(K, T, args.d_latent)

        with torch.no_grad():
            x_hat = model.decoder(torch.from_numpy(z_all).float().to(device)).cpu().numpy()
        test_recon_mse = ((x_hat.reshape(K, T, -1)[val_end:] - test_std.reshape(-1, T, D_cls)) ** 2).mean()
        print(f"  test recon MSE (standardized): {test_recon_mse:.4f}")

        ckpt_out = Path(args.output_dir) / args.eid / f"clsae_cam{v_idx}.pt"
        ckpt_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
            'mean': torch.from_numpy(mean.squeeze(0).astype(np.float32)),
            'std': torch.from_numpy(std.squeeze(0).astype(np.float32)),
            'd_in': int(D_cls),
            'd_hidden': int(args.d_hidden),
            'd_latent': int(args.d_latent),
            'dropout': float(args.dropout),
        }, ckpt_out)
        print(f"  saved weights: {ckpt_out}  sigma.min={std.min():.6g}")

    out_dir = Path(args.output_dir) / args.eid
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "z_trials.npz"
    np.savez(out_path, z_trials_time=out, trial_split=trial_split)
    print(f"\nSaved {out_path}")
    print(f"  z_trials_time shape: {out.shape}  dtype={out.dtype}")
    print(f"  trial_split: train={trial_split[0]} val={trial_split[1]} test={trial_split[2]}")


if __name__ == "__main__":
    main()
