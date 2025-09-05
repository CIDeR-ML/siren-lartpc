from __future__ import annotations
from typing import List
import torch
import torch.nn.functional as F


def vis_bias(
    target: torch.Tensor,
    pred: torch.Tensor,
    idx_slice: slice | List[int] = slice(None),
    threshold: float = 0.0,
):
    """
    Function to compute the visibility bias (the mean of 2 * |target - pred| / (target + pred))

    Parameters
    ----------
    target : torch.Tensor
        The reference visibility based on which the bias is calculated.
    pred : torch.Tensor
        The subject visibility for which the bias is calculated.
    threshold : float
        The visibility lowest threshold. The visibility bias is computed only
        for the instances for which the reference (target) tensor contains the
        visibility value above this threshold.

    Returns
    -------
    torch.Tensor
        The model visibility bias.
    """
    if isinstance(idx_slice, list):
        idx_slice = slice(*idx_slice)

    target = target[:, idx_slice]
    pred = pred[:, idx_slice]

    if target.shape != pred.shape:
        raise ValueError(
            f"target and pred must have the same shape {(*target.shape,)} != {(*pred.shape,)}"
        )

    mask = target > threshold
    a = pred[mask]
    b = target[mask]
    bias = (2 * torch.abs(a - b) / (a + b)).mean()
    return bias


def time_bias(
    target: torch.Tensor,
    pred: torch.Tensor,
    threshold: float = 0.0,
    idx_slice: slice | List[int] = slice(None),
    normalize=False,
):
    """
    Function to compute the visibility bias (the mean of 2 * |target - pred| / (target + pred))

    Parameters
    ----------
    target : torch.Tensor
        The reference visibility based on which the bias is calculated.
    pred : torch.Tensor
        The subject visibility for which the bias is calculated.
    threshold : float
        The visibility lowest threshold. The visibility bias is computed only
        for the instances for which the reference (target) tensor contains the
        visibility value above this threshold.

    Returns
    -------
    torch.Tensor
        The model visibility bias.
    """
    if isinstance(idx_slice, list):
        idx_slice = slice(*idx_slice)

    target = target[:, idx_slice]
    pred = pred[:, idx_slice]

    if normalize:
        target = F.normalize(target, p=2, dim=1)
        pred = F.normalize(pred, p=2, dim=1)

    if target.shape != pred.shape:
        raise ValueError(
            f"target and pred must have the same shape {(*target.shape,)} != {(*pred.shape,)}"
        )

    mask = target > threshold
    a = pred[mask]
    b = target[mask]
    bias = (2 * torch.abs(a - b) / (a + b)).mean()
    return bias


def abs_bias(target: torch.Tensor, pred: torch.Tensor, random=0):
    """
    Function to compute the absolute bias (the mean of |target - pred|)

    Parameters
    ----------
    target : torch.Tensor
        Some reference target based on which the bias is calculated.
    pred : torch.Tensor
        Prediction for target on which which the bias is calculated.

    Returns
    -------
    torch.Tensor
        The model absolute bias.

    """
    if target.shape != pred.shape:
        raise ValueError(
            f"target and pred must have the same shape {(*target.shape,)} != {(*pred.shape,)}"
        )

    return torch.abs(target - pred).mean()
