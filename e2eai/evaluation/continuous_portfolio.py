"""Pure continuous long-only portfolio mapping used by Experiment B.

The mapper deliberately has no selection, clipping, cap, or threshold logic:
every finite score on a decision date receives a strictly positive masked
softmax weight.  Keeping these functions independent from the model makes the
Experiment B invariance and allocation tests straightforward.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch


def cross_sectional_zscore(
    scores: Iterable[float] | np.ndarray,
    valid_mask: Iterable[bool] | np.ndarray | None = None,
    *,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Z-score finite valid scores only and return ``(z, mask, mean, std)``.

    Invalid scores are never used to estimate the cross-sectional moments and
    are set to zero in the returned z-score array.  The denominator is exactly
    ``max(std, eps)``; no clipping is applied to the resulting z values.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if eps <= 0:
        raise ValueError("eps must be positive")
    if valid_mask is None:
        mask = np.isfinite(values)
    else:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError("valid_mask must have the same shape as scores")
        mask &= np.isfinite(values)
    if not bool(mask.any()):
        raise ValueError("at least one finite valid score is required")
    valid = values[mask]
    mean = float(valid.mean())
    std = float(valid.std(ddof=0))
    denominator = max(std, float(eps))
    z = np.zeros_like(values, dtype=np.float64)
    z[mask] = (valid - mean) / denominator
    return z, mask, mean, std


def masked_softmax(
    values: Iterable[float] | np.ndarray,
    valid_mask: Iterable[bool] | np.ndarray,
    *,
    temperature: float = 1.0,
) -> np.ndarray:
    """Stable masked softmax with exact zero weights outside ``valid_mask``."""
    scores = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    if scores.ndim != 1 or mask.shape != scores.shape:
        raise ValueError("values and valid_mask must be one-dimensional and aligned")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    if not bool(mask.any()):
        raise ValueError("at least one valid score is required")
    if not np.all(np.isfinite(scores[mask])):
        raise ValueError("valid scores must be finite")
    logits = scores[mask] / float(temperature)
    logits = logits - float(np.max(logits))
    exponentials = np.exp(logits)
    weights_valid = exponentials / float(exponentials.sum())
    weights = np.zeros_like(scores, dtype=np.float64)
    weights[mask] = weights_valid
    return weights


def continuous_long_only_weights(
    scores: Iterable[float] | np.ndarray,
    valid_mask: Iterable[bool] | np.ndarray | None = None,
    *,
    temperature: float = 1.0,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Map raw scores to continuous long-only weights using z-score + softmax."""
    z, mask, mean, std = cross_sectional_zscore(scores, valid_mask, eps=eps)
    return masked_softmax(z, mask, temperature=temperature), z, mean, std


def continuous_long_only_weights_torch(
    scores: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    temperature: float = 1.0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Autograd-compatible one-dimensional implementation for future C runs."""
    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if eps <= 0 or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("eps and temperature must be positive")
    mask = torch.isfinite(scores) if valid_mask is None else valid_mask.bool() & torch.isfinite(scores)
    if not bool(mask.any()):
        raise ValueError("at least one finite valid score is required")
    count = mask.sum().to(dtype=scores.dtype)
    clean = torch.where(mask, scores, torch.zeros_like(scores))
    mean = clean.sum() / count
    centered = torch.where(mask, scores - mean, torch.zeros_like(scores))
    variance = (centered.square().sum() / count)
    std = torch.sqrt(variance)
    denominator = torch.maximum(std, torch.as_tensor(float(eps), dtype=scores.dtype, device=scores.device))
    z = torch.where(mask, centered / denominator, torch.zeros_like(scores))
    logits = torch.where(mask, z / float(temperature), torch.full_like(z, -torch.inf))
    weights = torch.softmax(logits, dim=0)
    return torch.where(mask, weights, torch.zeros_like(weights)), z, mean, std


def continuous_long_only_weights_torch_batch(
    scores: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    temperature: float = 1.0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched autograd-compatible allocator for ``[batch, assets]`` scores."""
    if scores.ndim != 2:
        raise ValueError("scores must have shape [B,N]")
    if eps <= 0 or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("eps and temperature must be positive")
    mask = torch.isfinite(scores) if valid_mask is None else valid_mask.bool() & torch.isfinite(scores)
    if not bool(mask.any(dim=-1).all()):
        raise ValueError("every batch row needs at least one finite valid score")
    count = mask.sum(dim=-1, keepdim=True).to(dtype=scores.dtype)
    clean = torch.where(mask, scores, torch.zeros_like(scores))
    mean = clean.sum(dim=-1, keepdim=True) / count
    centered = torch.where(mask, scores - mean, torch.zeros_like(scores))
    variance = centered.square().sum(dim=-1, keepdim=True) / count
    std = torch.sqrt(variance)
    denominator = torch.maximum(std, torch.as_tensor(float(eps), dtype=scores.dtype, device=scores.device))
    z = torch.where(mask, centered / denominator, torch.zeros_like(scores))
    logits = torch.where(mask, z / float(temperature), torch.full_like(z, -torch.inf))
    weights = torch.softmax(logits, dim=-1)
    return torch.where(mask, weights, torch.zeros_like(weights)), z, mean.squeeze(-1), std.squeeze(-1)


__all__ = [
    "cross_sectional_zscore",
    "masked_softmax",
    "continuous_long_only_weights",
    "continuous_long_only_weights_torch",
    "continuous_long_only_weights_torch_batch",
]
