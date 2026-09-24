"""Leakage-safe cumulative direction estimates for E2EAI.

The buffers in this module are model state, but they are not learnable model
parameters.  A trainer must therefore read the directions before producing a
portfolio for a date and call :meth:`DirectionalBuffer.update` only after that
portfolio decision has been made.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

DirectionNormalization = Literal["none", "cross_factor_zscore"]


def masked_cross_sectional_ic(
    x: Tensor,
    y: Tensor,
    mask: Tensor,
    eps: float = 1e-6,
    min_stocks: int = 3,
) -> Tensor:
    """Compute a masked Pearson information coefficient over the last axis.

    ``x`` and ``y`` may have arbitrary broadcast-compatible leading axes; the
    last axis is always the stock cross-section.  ``mask`` must be broadcastable
    to the resulting shape.  For example, inputs shaped ``[B, K, M, N]`` return
    ICs shaped ``[B, K, M]``.  Non-finite pairs are excluded.  Cross-sections
    with fewer than ``min_stocks`` usable observations or effectively zero
    variance return exactly zero rather than NaN.

    The implementation remains differentiable with respect to finite selected
    values, making it suitable both for loss functions and detached buffer
    updates.

    Args:
        x: First values, with stocks on the final dimension.
        y: Second values, broadcast-compatible with ``x``.
        mask: Boolean observation mask, broadcast-compatible with ``x``/``y``.
        eps: Numerical threshold for variances and the correlation denominator.
        min_stocks: Minimum number of finite selected stocks required per IC.

    Returns:
        Pearson correlations with the stock dimension removed.
    """
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if min_stocks < 2:
        raise ValueError("min_stocks must be at least 2 for Pearson correlation")
    if not (x.is_floating_point() and y.is_floating_point()):
        raise TypeError("x and y must be floating-point tensors")
    if x.device != y.device or x.device != mask.device:
        raise ValueError("x, y, and mask must be on the same device")
    if x.ndim < 1 or y.ndim < 1:
        raise ValueError("x and y must include a final stock dimension")

    try:
        x_broadcast, y_broadcast = torch.broadcast_tensors(x, y)
        valid_mask = torch.broadcast_to(mask.bool(), x_broadcast.shape)
    except RuntimeError as error:
        raise ValueError("x, y, and mask are not broadcast-compatible") from error

    if x_broadcast.shape[-1] != y_broadcast.shape[-1]:
        raise ValueError("x and y must have the same stock dimension")

    promoted = torch.promote_types(x_broadcast.dtype, y_broadcast.dtype)
    work_dtype = (
        torch.float32 if promoted in (torch.float16, torch.bfloat16) else promoted
    )
    x_work = x_broadcast.to(dtype=work_dtype)
    y_work = y_broadcast.to(dtype=work_dtype)
    valid = valid_mask & torch.isfinite(x_work) & torch.isfinite(y_work)

    x_clean = torch.where(valid, x_work, torch.zeros_like(x_work))
    y_clean = torch.where(valid, y_work, torch.zeros_like(y_work))
    count = valid.sum(dim=-1, keepdim=True)
    denominator_count = count.clamp_min(1).to(dtype=work_dtype)

    x_mean = x_clean.sum(dim=-1, keepdim=True) / denominator_count
    y_mean = y_clean.sum(dim=-1, keepdim=True) / denominator_count
    x_centered = torch.where(valid, x_work - x_mean, torch.zeros_like(x_work))
    y_centered = torch.where(valid, y_work - y_mean, torch.zeros_like(y_work))

    covariance_numerator = (x_centered * y_centered).sum(dim=-1)
    x_sum_squares = x_centered.square().sum(dim=-1)
    y_sum_squares = y_centered.square().sum(dim=-1)
    correlation_denominator = torch.sqrt(x_sum_squares * y_sum_squares)

    usable = (
        (count.squeeze(-1) >= min_stocks)
        & (x_sum_squares > eps)
        & (y_sum_squares > eps)
    )
    correlation = covariance_numerator / correlation_denominator.clamp_min(eps)
    correlation = torch.where(usable, correlation, torch.zeros_like(correlation))
    return torch.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)


class DirectionalBuffer(nn.Module):
    """Maintain cumulative original-factor and deep-factor IC directions.

    E2EAI's directional state has two distinct roles:

    * ``original_factor_ic_sum[k, m]`` records cumulative
      ``IC(F[:, m], deep_factor[k])`` and drives the interpretable signed linear
      approximation in Eqs. (7)-(8).
    * ``deep_factor_ic_sum[k]`` records cumulative
      ``IC(forward_return[k], deep_factor[k])`` and determines whether horizon
      ``k`` is used long or short by the portfolio module.

    Both tensors are registered buffers with shapes ``[K, M]`` and ``[K]``.
    They move with the module and are included in ``state_dict`` while remaining
    absent from ``parameters()``.

    Args:
        num_horizons: Number of return/deep-factor horizons ``K``.
        num_factors: Number of original factors ``M``.
        normalization: Whether to sign raw cumulative factor ICs or their
            per-horizon cross-factor z-scores.
        min_stocks: Minimum usable cross-sectional observations for an update.
        eps: Positive initialization and numerical tolerance.
        zero_direction: Deterministic sign used when a score equals zero.
    """

    def __init__(
        self,
        num_horizons: int,
        num_factors: int,
        normalization: DirectionNormalization = "cross_factor_zscore",
        min_stocks: int = 3,
        eps: float = 1e-6,
        zero_direction: Literal[-1, 1] = 1,
    ) -> None:
        super().__init__()
        if num_horizons < 1:
            raise ValueError("num_horizons must be positive")
        if num_factors < 1:
            raise ValueError("num_factors must be positive")
        if normalization not in ("none", "cross_factor_zscore"):
            raise ValueError(
                "normalization must be 'none' or 'cross_factor_zscore'"
            )
        if min_stocks < 2:
            raise ValueError("min_stocks must be at least 2")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        if zero_direction not in (-1, 1):
            raise ValueError("zero_direction must be either -1 or +1")

        self.num_horizons = num_horizons
        self.num_factors = num_factors
        self.normalization = normalization
        self.min_stocks = min_stocks
        self.eps = eps
        self.zero_direction = zero_direction

        self.register_buffer(
            "original_factor_ic_sum",
            torch.full((num_horizons, num_factors), eps, dtype=torch.float32),
        )
        self.register_buffer(
            "deep_factor_ic_sum",
            torch.full((num_horizons,), eps, dtype=torch.float32),
        )

    def reset(self) -> None:
        """Reset both cumulative statistics to the configured positive epsilon."""
        with torch.no_grad():
            self.original_factor_ic_sum.fill_(self.eps)
            self.deep_factor_ic_sum.fill_(self.eps)

    def _directions_from_scores(self, scores: Tensor) -> Tensor:
        positive = torch.ones_like(scores)
        negative = -positive
        fallback = torch.full_like(scores, float(self.zero_direction))
        return torch.where(
            scores > 0.0,
            positive,
            torch.where(scores < 0.0, negative, fallback),
        )

    def factor_directions(self) -> Tensor:
        """Return original-factor directions with shape ``[K, M]``.

        E2EAI Eq. (7) uses signs derived from cumulative IC.  In the default
        specification, each horizon's score vector is first standardized across
        its ``M`` factors.  A zero-variance row therefore uses the configured
        deterministic fallback direction for every factor.
        """
        scores = self.original_factor_ic_sum
        if self.normalization == "cross_factor_zscore":
            work = (
                scores.float()
                if scores.dtype in (torch.float16, torch.bfloat16)
                else scores
            )
            centered = work - work.mean(dim=-1, keepdim=True)
            scale = centered.square().mean(dim=-1, keepdim=True).sqrt()
            scores = torch.where(
                scale > self.eps,
                centered / scale.clamp_min(self.eps),
                torch.zeros_like(centered),
            ).to(dtype=self.original_factor_ic_sum.dtype)
        return self._directions_from_scores(scores)

    def deep_directions(self) -> Tensor:
        """Return deep-factor return directions with shape ``[K]``."""
        return self._directions_from_scores(self.deep_factor_ic_sum)

    def forward(self) -> tuple[Tensor, Tensor]:
        """Return a read-only-style snapshot of factor and deep directions."""
        return self.factor_directions(), self.deep_directions()

    @torch.no_grad()
    def update(
        self,
        original_factors: Tensor,
        deep_factors: Tensor,
        forward_returns: Tensor,
        asset_mask: Tensor,
        return_mask: Tensor | None = None,
        update_factor_statistics: bool = True,
    ) -> dict[str, Tensor]:
        """Accumulate detached, per-date masked IC observations.

        This method must be called *after* the current portfolio forward pass.
        The method itself enforces ``no_grad`` and detaches all observations, but
        call ordering remains the trainer's responsibility.

        Args:
            original_factors: Selected/masked factors shaped ``[B, N, M]``.
            deep_factors: Learned deep factors shaped ``[B, K, N]``.
            forward_returns: Realized forward returns shaped ``[B, K, N]``.
            asset_mask: Valid-stock mask shaped ``[B, N]``.
            return_mask: Optional observed-label mask shaped ``[B, N]`` or
                ``[B, K, N]``.  It affects deep-factor/return IC only; original
                factor/deep-factor IC continues to use every valid stock.
            update_factor_statistics: Set false when releasing delayed return
                labels in walk-forward evaluation, avoiding duplicate updates
                to the label-free original-factor/deep-factor statistic.

        Returns:
            Detached batch IC increments and post-update directions for logging.
        """
        self._validate_update_shapes(
            original_factors, deep_factors, forward_returns, asset_mask
        )
        if original_factors.device != self.original_factor_ic_sum.device:
            raise ValueError("inputs and DirectionalBuffer must be on the same device")

        factors = original_factors.detach()
        deep = deep_factors.detach()
        returns = forward_returns.detach()
        valid_assets = asset_mask.detach().bool()

        # [B, 1, M, N] with [B, K, 1, N] -> [B, K, M] IC increments.
        factor_values = factors.permute(0, 2, 1).unsqueeze(1)
        deep_for_factors = deep.unsqueeze(2)
        factor_valid = valid_assets[:, None, None, :]
        factor_ic = masked_cross_sectional_ic(
            factor_values,
            deep_for_factors,
            factor_valid,
            eps=self.eps,
            min_stocks=self.min_stocks,
        )

        deep_valid = valid_assets[:, None, :].expand_as(deep)
        if return_mask is not None:
            observed = return_mask.detach().bool()
            if observed.device != valid_assets.device:
                raise ValueError("return_mask must be on the same device as inputs")
            if observed.shape == valid_assets.shape:
                observed = observed[:, None, :].expand_as(deep)
            elif observed.shape != deep.shape:
                raise ValueError("return_mask must have shape [B, N] or [B, K, N]")
            deep_valid = deep_valid & observed

        deep_ic = masked_cross_sectional_ic(
            returns,
            deep,
            deep_valid,
            eps=self.eps,
            min_stocks=self.min_stocks,
        )

        factor_increment = factor_ic.sum(dim=0).to(
            device=self.original_factor_ic_sum.device,
            dtype=self.original_factor_ic_sum.dtype,
        )
        deep_increment = deep_ic.sum(dim=0).to(
            device=self.deep_factor_ic_sum.device,
            dtype=self.deep_factor_ic_sum.dtype,
        )
        if update_factor_statistics:
            self.original_factor_ic_sum.add_(factor_increment)
        else:
            factor_increment.zero_()
        self.deep_factor_ic_sum.add_(deep_increment)

        return {
            "original_factor_ic_increment": factor_increment.clone(),
            "deep_factor_ic_increment": deep_increment.clone(),
            "factor_direction": self.factor_directions().clone(),
            "deep_direction": self.deep_directions().clone(),
        }

    def _validate_update_shapes(
        self,
        original_factors: Tensor,
        deep_factors: Tensor,
        forward_returns: Tensor,
        asset_mask: Tensor,
    ) -> None:
        if original_factors.ndim != 3:
            raise ValueError("original_factors must have shape [B, N, M]")
        if deep_factors.ndim != 3:
            raise ValueError("deep_factors must have shape [B, K, N]")
        if forward_returns.shape != deep_factors.shape:
            raise ValueError("forward_returns must match deep_factors [B, K, N]")
        batch, stocks, factors = original_factors.shape
        if factors != self.num_factors:
            raise ValueError(
                f"expected {self.num_factors} factors, received {factors}"
            )
        if deep_factors.shape != (batch, self.num_horizons, stocks):
            raise ValueError(
                "deep_factors dimensions must align with original_factors and configured K"
            )
        if asset_mask.shape != (batch, stocks):
            raise ValueError("asset_mask must have shape [B, N]")
        devices = {
            original_factors.device,
            deep_factors.device,
            forward_returns.device,
            asset_mask.device,
        }
        if len(devices) != 1:
            raise ValueError("all update tensors must be on the same device")

    def extra_repr(self) -> str:
        """Return concise module configuration for model summaries."""
        return (
            f"num_horizons={self.num_horizons}, num_factors={self.num_factors}, "
            f"normalization={self.normalization!r}, min_stocks={self.min_stocks}"
        )
