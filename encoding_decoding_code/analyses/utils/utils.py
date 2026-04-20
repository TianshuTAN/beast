import argparse
import copy
import os
import random

import numpy as np
import torch

from typing import (Callable, Dict, Iterable, List, Literal, Optional, Tuple,
                    Union)
import matplotlib.pyplot as plt
from torcheval.metrics import R2Score
from tqdm import tqdm
from scipy.special import gammaln

def get_args():
    parser = argparse.ArgumentParser(description='Neural Encoding or Decoding')
    parser.add_argument('--eid', type=str, required=True, help='Session / animal id (subfolder name)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--neural_input_dir', type=str, default='input', help='Input directory for neural data')
    parser.add_argument('--latent_input_dir', type=str, default='input', help='Input directory for video latents')
    parser.add_argument('--eval_task', type=str, default='encoding', help='Evaluation task: encoding or decoding')
    args = parser.parse_args()
    return args

def set_seed(seed):
    # set seed for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('seed set to {}'.format(seed))

r2_metric = R2Score()
def r2_score(y_true, y_pred, device="cpu"):
    r2_metric.reset()
    r2_metric.to(device)
    y_true = y_true.to(device)
    y_pred = y_pred.to(device)
    r2_metric.update(y_pred, y_true)
    return r2_metric.compute().item()

def neg_log_likelihood(rates, spikes, zero_warning=True, threshold=1e-9):
    """Calculates Poisson negative log likelihood given rates and spikes.
    formula: -log(e^(-r) / n! * r^n)
           = r - n*log(r) + log(n!)

    Parameters
    ----------
    rates : np.ndarray
        numpy array containing rate predictions
    spikes : np.ndarray
        numpy array containing true spike counts
    zero_warning : bool, optional
        Whether to print out warning about 0 rate
        predictions or not

    Returns
    -------
    float
        Total negative log-likelihood of the data
    """
    assert (
            spikes.shape == rates.shape
    ), f"neg_log_likelihood: Rates and spikes should be of the same shape. spikes: {spikes.shape}, rates: {rates.shape}"

    if np.any(np.isnan(spikes)):
        mask = np.isnan(spikes)
        rates = rates[~mask]
        spikes = spikes[~mask]

    assert not np.any(np.isnan(rates)), "neg_log_likelihood: NaN rate predictions found"

    assert np.all(rates >= 0), "neg_log_likelihood: Negative rate predictions found"
    if np.any(rates == 0):
        if zero_warning:
            logger.warning(
                "neg_log_likelihood: Zero rate predictions found. Replacing zeros with 1e-9"
            )
        rates[rates == 0] = threshold
    result = rates - spikes * np.log(rates) + gammaln(spikes + 1.0)
    return np.sum(result)

def bits_per_spike(rates, spikes, threshold=1e-9):
    """Computes bits per spike of rate predictions given spikes.
    Bits per spike is equal to the difference between the log-likelihoods (in base 2)
    of the rate predictions and the null model (i.e. predicting mean firing rate of each neuron)
    divided by the total number of spikes.

    Parameters
    ----------
    rates : np.ndarray
        3d numpy array containing rate predictions
    spikes : np.ndarray
        3d numpy array containing true spike counts

    Returns
    -------
    float
        Bits per spike of rate predictions
    """
    nll_model = neg_log_likelihood(rates, spikes, threshold=threshold)
    null_rates = np.tile(
        np.nanmean(spikes, axis=tuple(range(spikes.ndim - 1)), keepdims=True),
        spikes.shape[:-1] + (1,),
    )
    nll_null = neg_log_likelihood(null_rates, spikes, threshold=threshold)
    return (nll_null - nll_model) / np.nansum(spikes) / np.log(2)

def plot_gt_pred(gt, pred, epoch=0,modality="behavior"):
    # plot Ground Truth and Prediction in the same figur
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.set_title("Ground Truth")
    im1 = ax1.imshow(gt, aspect='auto', cmap='binary')
    
    ax2.set_title("Prediction")
    im2 = ax2.imshow(pred, aspect='auto', cmap='binary')
    
    # add colorbar
    plt.colorbar(im1, ax=ax1)
    plt.colorbar(im2, ax=ax2)

    fig.suptitle("Epoch: {}, Mod: {}".format(epoch, 
                                             modality))
    return fig

def plot_neurons_r2(gt, pred, epoch=0, neuron_idx=[],modality="behavior"):
    # Create one figure and axis for all plots
    fig, axes = plt.subplots(len(neuron_idx), 1, figsize=(12, 5 * len(neuron_idx)))
    r2_values = []  # To store R2 values for each neuron
    
    for neuron in neuron_idx:
        r2 = r2_score(y_true=gt[:, neuron], y_pred=pred[:, neuron])
        r2_values.append(r2)
        ax = axes if len(neuron_idx) == 1 else axes[neuron_idx.index(neuron)]
        ax.plot(gt[:, neuron].cpu().numpy(), label="Ground Truth", color="blue")
        ax.plot(pred[:, neuron].cpu().numpy(), label="Prediction", color="red")
        ax.set_title("Neuron: {}, R2: {:.4f}".format(neuron, r2))
        ax.legend()
        # x label
        ax.set_xlabel("Time")
        # y label
        ax.set_ylabel("Rate")
    fig.suptitle("Epoch: {}, Mod: {}, Avg R2: {:.4f}".format(epoch, 
                                                            modality, 
                                                            np.mean(r2_values)))
    return fig

def _std(arr):
    # STD_MODE env var toggles: 'per_feature' (default, new: pool over trials+time)
    # vs 'per_timebin' (original: per-time-per-feature).
    mode = os.environ.get('STD_MODE', 'per_feature')
    if mode == 'per_timebin':
        mean = np.mean(arr, axis=0)       # (T, N)
        std = np.std(arr, axis=0)         # (T, N)
    else:
        mean = np.mean(arr, axis=(0, 1))  # (N,)
        std = np.std(arr, axis=(0, 1))    # (N,)
    std = np.clip(std, 1e-8, None)
    arr = (arr - mean) / std
    return arr, mean, std
