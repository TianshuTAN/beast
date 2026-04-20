"""Generate loss curve figure for session 1 BEAST-AE training from tensorboard events."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _read(tb_dir: Path, tag: str) -> tuple[np.ndarray, np.ndarray]:
    a = EventAccumulator(str(tb_dir))
    a.Reload()
    if tag not in a.Tags()["scalars"]:
        return np.array([]), np.array([])
    evs = a.Scalars(tag)
    steps = np.array([e.step for e in evs], dtype=np.int64)
    vals = np.array([e.value for e in evs], dtype=np.float64)
    return steps, vals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", required=True, help="left tb_logs/version_X dir")
    ap.add_argument("--right", required=True, help="right tb_logs/version_X dir")
    ap.add_argument("--out", required=True, help="output PNG path")
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

    for ax, tag, path, title in [
        (axes[0], "LEFT camera", Path(args.left), "LEFT camera"),
        (axes[1], "RIGHT camera", Path(args.right), "RIGHT camera"),
    ]:
        s_train, v_train = _read(path, "train_loss_step")
        s_val, v_val = _read(path, "val_loss")
        s_ep, v_ep = _read(path, "train_loss_epoch")

        if s_train.size:
            # smooth: exponential moving average for clarity
            ema = np.zeros_like(v_train)
            alpha = 0.05
            ema[0] = v_train[0]
            for i in range(1, len(v_train)):
                ema[i] = alpha * v_train[i] + (1 - alpha) * ema[i - 1]
            ax.plot(s_train, v_train, color="#6baed6", alpha=0.35, linewidth=0.6, label="train (step)")
            ax.plot(s_train, ema, color="#08519c", linewidth=1.6, label="train (EMA α=0.05)")
        if s_val.size:
            ax.plot(s_val, v_val, color="#e6550d", marker="o", markersize=3, linewidth=1.2, label="val (per val cycle)")

        first = v_train[0] if v_train.size else float("nan")
        last = v_train[-1] if v_train.size else float("nan")
        ax.set_title(f"{tag} — loss {first:.3f} → {last:.3f}")
        ax.set_xlabel("global step")
        ax.set_ylabel("MSE loss")
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle=":", alpha=0.4)
        ax.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        "BEAST-AE training curves (session 1, EID 4b00df29...)\n"
        "800 epochs × ~16 steps/epoch, mask_ratio=0, num_latents=100",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"Saved {out_path}")

    # also report summary stats for the slides
    for name, path in [("LEFT", Path(args.left)), ("RIGHT", Path(args.right))]:
        s_train, v_train = _read(path, "train_loss_step")
        s_val, v_val = _read(path, "val_loss")
        if v_train.size:
            print(
                f"{name}: train_loss first={v_train[0]:.4f} last={v_train[-1]:.4f} min={v_train.min():.4f} "
                f"steps={len(v_train)}"
            )
        if v_val.size:
            print(
                f"{name}: val_loss   first={v_val[0]:.4f} last={v_val[-1]:.4f} min={v_val.min():.4f} "
                f"cycles={len(v_val)}"
            )


if __name__ == "__main__":
    main()
