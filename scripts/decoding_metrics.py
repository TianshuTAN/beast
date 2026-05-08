"""
pip install 'torchmetrics[image]'

SSIM and PSNR Evaluation Checklist
- Ensure tensors are in shape (B, 3, 320, 320). Resize images to 320×320 using PIL.Image.Resampling.BICUBIC.
- normalize all images to [0, 1]. Then, use a fixed SSIM setting: data_range = 1.0. 

Save metrics to npz file. K is the number of trials, T is the number of time bins, V is the number of views.
- psnr: [K, T, V]
- ssim: [K, T, V]
- neural_trial_idx: [K]
- neural_bin_idx: [K, T]
- trial_split: [K]
- source_files: [K]
"""
from pathlib import Path

import numpy as np
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio


def _flatten_image_batch(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    metric_name: str,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Flatten tensors shaped [B, V, C, H, W] to [B*V, C, H, W]."""
    if pred.shape != target.shape:
        raise ValueError(f"{metric_name} expects matching shapes; got pred={pred.shape} target={target.shape}")
    if pred.ndim != 5:
        raise ValueError(f"{metric_name} expects [B,V,C,H,W]; got {pred.shape}")

    bsz, views, channels, height, width = pred.shape
    return (
        pred.reshape(bsz * views, channels, height, width),
        target.reshape(bsz * views, channels, height, width),
        bsz,
        views,
    )


def _psnr_per_image(pred: torch.Tensor, target: torch.Tensor, *, data_range: float = 1.0) -> torch.Tensor:
    """Compute PSNR for tensors shaped [B, V, C, H, W], returning [B, V]."""
    pred, target, bsz, views = _flatten_image_batch(pred, target, metric_name="PSNR")

    metric = PeakSignalNoiseRatio(data_range=data_range, dim=(1, 2, 3), reduction="none").to(pred.device)
    psnr = metric(pred.detach().float(), target.detach().float())
    return psnr.reshape(bsz, views)


def _ssim_per_image(pred: torch.Tensor, target: torch.Tensor, *, data_range: float = 1.0) -> torch.Tensor:
    """Compute SSIM for tensors shaped [B, V, C, H, W], returning [B, V]."""
    pred, target, bsz, views = _flatten_image_batch(pred, target, metric_name="SSIM")

    if min(pred.shape[-2:]) < 11:
        raise ValueError(f"SSIM with an 11x11 Gaussian window requires H,W >= 11; got {pred.shape[-2:]}")

    metric = StructuralSimilarityIndexMeasure(
        data_range=data_range,
        gaussian_kernel=True,
        kernel_size=11,
        reduction="none",
    ).to(pred.device)
    ssim = metric(pred.detach().float(), target.detach().float())
    return ssim.reshape(bsz, views)


def _image_metrics_by_view(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Return PSNR and SSIM as numpy arrays shaped [B, V]."""
    pred = pred.detach().float().clamp(0.0, 1.0)
    target = target.detach().float()

    data_range = 1.0
    psnr = _psnr_per_image(pred, target, data_range=data_range)
    ssim = _ssim_per_image(pred, target, data_range=data_range)
    return (
        psnr.detach().cpu().numpy().astype(np.float32),
        ssim.detach().cpu().numpy().astype(np.float32),
    )


def load_neural_trial_idx(path: Path, *, k_trials: int, t_bins: int) -> np.ndarray:
    """Return one neural trial id per K row when available, else a stable placeholder."""
    with np.load(path, allow_pickle=True) as d:
        if "neural_trial_idx" not in d.files:
            return np.full((k_trials,), -1, dtype=np.int64)
        trial_idx = np.asarray(d["neural_trial_idx"], dtype=np.int64).reshape(-1)

    if len(trial_idx) == k_trials:
        return trial_idx
    if len(trial_idx) == k_trials * t_bins:
        return trial_idx.reshape(k_trials, t_bins)[:, 0]
    raise ValueError(
        f"{path}: neural_trial_idx length {len(trial_idx)} cannot index z shape "
        f"({k_trials}, {t_bins}, ...)"
    )


def load_neural_bin_idx(path: Path, *, k_trials: int, t_bins: int) -> np.ndarray:
    """Return neural bin ids shaped [K, T] when available, else [0, ..., T-1]."""
    with np.load(path, allow_pickle=True) as d:
        if "neural_bin_idx" not in d.files:
            return np.tile(np.arange(t_bins, dtype=np.int64), (k_trials, 1))
        bin_idx = np.asarray(d["neural_bin_idx"], dtype=np.int64).reshape(-1)

    if len(bin_idx) == k_trials * t_bins:
        return bin_idx.reshape(k_trials, t_bins)
    raise ValueError(
        f"{path}: neural_bin_idx length {len(bin_idx)} cannot index z shape "
        f"({k_trials}, {t_bins}, ...)"
    )


def load_trial_split(path: Path, *, k_trials: int, t_bins: int) -> np.ndarray:
    """Return one split label per K row when available, else a stable placeholder."""
    with np.load(path, allow_pickle=True) as d:
        if "trial_split" not in d.files:
            return np.full((k_trials,), "unknown", dtype=str)
        split = np.asarray(
            [str(x).lower() for x in np.asarray(d["trial_split"], dtype=object).reshape(-1)]
        )

    if len(split) == k_trials:
        return split.astype(str)
    if len(split) == k_trials * t_bins:
        split_by_time = split.reshape(k_trials, t_bins)
        first_split = split_by_time[:, 0]
        if not np.all(split_by_time == first_split[:, None]):
            raise ValueError(f"{path}: trial_split varies across time bins for the same trial")
        return first_split.astype(str)
    raise ValueError(
        f"{path}: trial_split length {len(split)} cannot index z shape "
        f"({k_trials}, {t_bins}, ...)"
    )


def resolve_metrics_npz_path(metrics_npz: Path | None, out_dir: Path) -> Path:
    """Return the explicit metrics path or the shared default under ``out_dir``."""
    return (
        Path(metrics_npz).resolve()
        if metrics_npz is not None
        else Path(out_dir) / "psnr_ssim_metrics.npz"
    )


def collect_psnr_ssim_metrics_block(
    pred: torch.Tensor,
    target: torch.Tensor,
    source_npz: Path,
    *,
    k_trials: int,
    t_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Compute one canonical metrics block for a token file.

    Returns PSNR/SSIM shaped [K, T, V], neural trial ids shaped [K], neural bin ids
    shaped [K, T], split labels shaped [K], and one source-file row per K trial.
    """
    psnr_flat, ssim_flat = _image_metrics_by_view(pred, target)
    metric_views = psnr_flat.shape[1]
    source_npz = Path(source_npz)
    return (
        psnr_flat.reshape(k_trials, t_bins, metric_views).astype(np.float32),
        ssim_flat.reshape(k_trials, t_bins, metric_views).astype(np.float32),
        load_neural_trial_idx(source_npz, k_trials=k_trials, t_bins=t_bins),
        load_neural_bin_idx(source_npz, k_trials=k_trials, t_bins=t_bins),
        load_trial_split(source_npz, k_trials=k_trials, t_bins=t_bins),
        [str(source_npz)] * k_trials,
    )


def save_psnr_ssim_metrics_npz(
    metrics_npz: Path,
    *,
    psnr_blocks: list[np.ndarray],
    ssim_blocks: list[np.ndarray],
    neural_trial_blocks: list[np.ndarray],
    neural_bin_blocks: list[np.ndarray],
    trial_split_blocks: list[np.ndarray],
    source_file_rows: list[str],
    view_names: tuple[str, ...] = ("left", "right"),
) -> dict[str, np.ndarray]:
    """Save PSNR/SSIM metrics with the shared compressed NPZ structure."""
    if not psnr_blocks:
        raise RuntimeError("No metrics were collected.")

    psnr = np.concatenate(psnr_blocks, axis=0).astype(np.float32)
    ssim = np.concatenate(ssim_blocks, axis=0).astype(np.float32)
    neural_trial_idx = np.concatenate(neural_trial_blocks, axis=0).astype(np.int64)
    neural_bin_idx = np.concatenate(neural_bin_blocks, axis=0).astype(np.int64)
    trial_split = np.concatenate(trial_split_blocks, axis=0).astype(str)
    arrays = {
        "psnr": psnr,
        "ssim": ssim,
        "average_psnr": np.asarray(np.nanmean(psnr), dtype=np.float32),
        "average_ssim": np.asarray(np.nanmean(ssim), dtype=np.float32),
        "neural_trial_idx": neural_trial_idx,
        "neural_bin_idx": neural_bin_idx,
        "trial_split": trial_split,
        "source_files": np.asarray(source_file_rows, dtype=str),
        "view_names": np.asarray(view_names, dtype=str),
    }

    metrics_npz = Path(metrics_npz)
    metrics_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(metrics_npz, **arrays)
    return arrays