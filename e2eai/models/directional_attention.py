"""Directional original-factor attention and linear deep-factor approximation."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn

from e2eai.utils.masking import masked_softmax


class DirectionalAttentionOutput(NamedTuple):
    """Outputs of :class:`DirectionalFactorAttention`.

    Attributes:
        factor_attentions: Per-stock factor attention ``[B, K, N, M]``.
        factor_attention_mean: Valid-stock mean attention ``[B, K, M]``.
        signed_factor_coefficients: Mean attention multiplied by the historical
            factor direction, shaped ``[B, K, M]``.
        deep_factor_approx: Directional linear approximation ``[B, K, N]``.
    """

    factor_attentions: Tensor
    factor_attention_mean: Tensor
    signed_factor_coefficients: Tensor
    deep_factor_approx: Tensor


def _masked_cross_sectional_zscore(
    values: Tensor,
    asset_mask: Tensor,
    eps: float,
) -> Tensor:
    """Standardize ``[B, N, M]`` values over valid stocks independently by date."""
    valid = asset_mask.bool().unsqueeze(-1) & torch.isfinite(values)
    clean = torch.where(valid, values, torch.zeros_like(values))
    count = valid.sum(dim=1, keepdim=True)
    denominator = count.clamp_min(1).to(dtype=values.dtype)
    mean = clean.sum(dim=1, keepdim=True) / denominator
    centered = torch.where(valid, values - mean, torch.zeros_like(values))
    variance = centered.square().sum(dim=1, keepdim=True) / denominator
    normalized = centered / torch.sqrt(variance + eps)
    return torch.where(valid, normalized, torch.zeros_like(normalized))


class DirectionalFactorAttention(nn.Module):
    """Approximate each deep factor with signed attended original factors.

    Each horizon owns an independent ``Linear(M, M)`` projection.  E2EAI
    Eq. (7) is applied stockwise and normalized over the *factor* dimension.
    Invalid stocks receive an all-zero attention row.  The valid-stock mean
    attention is combined with a pre-update direction snapshot and Eq. (8) is
    implemented in its dimensionally consistent form::

        beta[b, k, m] = mean_attention[b, k, m] * direction[b, k, m]
        f_hat[b, k, n] = sum_m F[b, n, m] * beta[b, k, m]

    NOTE:
        The paper's printed transpose notation is dimensionally ambiguous.  The
        matrix-vector form above maps ``[N, M] @ [M]`` to ``[N]`` and is the
        explicit resolution required by this implementation.

    Args:
        num_factors: Fixed number of original factor columns ``M``.
        num_horizons: Number of independent attention heads ``K``.
        leaky_relu_slope: Leaky-ReLU negative slope after each projection.
        zscore_approximation_input: If true, standardize each factor across
            valid stocks before constructing ``f_hat``.  Attention projections
            still consume the provided selected/masked factors.
        eps: Numerical constant used by optional cross-sectional standardization.
    """

    def __init__(
        self,
        num_factors: int,
        num_horizons: int,
        leaky_relu_slope: float = 0.2,
        zscore_approximation_input: bool = False,
        eps: float = 1e-6,
        *,
        negative_slope: float | None = None,
    ) -> None:
        super().__init__()
        # Backward-compatible keyword for early integration code.  New callers
        # should use the project-wide ``leaky_relu_slope`` spelling.
        if negative_slope is not None:
            leaky_relu_slope = negative_slope
        if num_factors < 1:
            raise ValueError("num_factors must be positive")
        if num_horizons < 1:
            raise ValueError("num_horizons must be positive")
        if leaky_relu_slope < 0.0:
            raise ValueError("leaky_relu_slope must be nonnegative")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        self.num_factors = num_factors
        self.num_horizons = num_horizons
        self.leaky_relu_slope = leaky_relu_slope
        self.zscore_approximation_input = zscore_approximation_input
        self.eps = eps
        self.projections = nn.ModuleList(
            nn.Linear(num_factors, num_factors) for _ in range(num_horizons)
        )
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_slope)

    def forward(
        self,
        selected_factors: Tensor,
        asset_mask: Tensor,
        factor_directions: Tensor,
    ) -> DirectionalAttentionOutput:
        """Generate per-horizon factor attention and linear approximations.

        Args:
            selected_factors: Fixed-width gated factors ``[B, N, M]``.
            asset_mask: Valid-stock mask ``[B, N]``.
            factor_directions: Direction snapshot ``[K, M]`` or ``[B, K, M]``.

        Returns:
            A :class:`DirectionalAttentionOutput` with the shapes documented on
            its fields.  No future-return labels are accepted by this API.
        """
        batch, stocks = self._validate_inputs(
            selected_factors, asset_mask, factor_directions
        )
        valid_assets = asset_mask.bool()
        valid_features = valid_assets.unsqueeze(-1)
        factors = torch.where(
            valid_features & torch.isfinite(selected_factors),
            selected_factors,
            torch.zeros_like(selected_factors),
        )

        # E2EAI Eq. (7): independent stockwise Linear(M, M) attention per horizon.
        logits = torch.stack(
            [self.activation(projection(factors)) for projection in self.projections],
            dim=1,
        )
        attention_mask = valid_assets[:, None, :, None].expand_as(logits)
        factor_attentions = masked_softmax(logits, attention_mask, dim=-1)

        stock_count = valid_assets.sum(dim=-1, keepdim=True).clamp_min(1)
        factor_attention_mean = factor_attentions.sum(dim=2) / stock_count[:, None].to(
            dtype=factor_attentions.dtype
        )

        directions = self._expand_directions(
            factor_directions,
            batch=batch,
            device=selected_factors.device,
            dtype=factor_attentions.dtype,
        )
        signed_factor_coefficients = factor_attention_mean * directions

        approximation_input = factors
        if self.zscore_approximation_input:
            work = (
                factors.float()
                if factors.dtype in (torch.float16, torch.bfloat16)
                else factors
            )
            approximation_input = _masked_cross_sectional_zscore(
                work, valid_assets, self.eps
            )

        # E2EAI Eqs. (7)-(8), dimensional correction: [B,N,M] x [B,K,M].
        approximation_coefficients = signed_factor_coefficients.to(
            dtype=approximation_input.dtype
        )
        deep_factor_approx = torch.einsum(
            "bnm,bkm->bkn", approximation_input, approximation_coefficients
        )
        deep_factor_approx = torch.where(
            valid_assets[:, None, :],
            deep_factor_approx,
            torch.zeros_like(deep_factor_approx),
        )

        return DirectionalAttentionOutput(
            factor_attentions=factor_attentions,
            factor_attention_mean=factor_attention_mean,
            signed_factor_coefficients=signed_factor_coefficients,
            deep_factor_approx=deep_factor_approx,
        )

    def _validate_inputs(
        self,
        selected_factors: Tensor,
        asset_mask: Tensor,
        factor_directions: Tensor,
    ) -> tuple[int, int]:
        if selected_factors.ndim != 3:
            raise ValueError("selected_factors must have shape [B, N, M]")
        if not selected_factors.is_floating_point():
            raise TypeError("selected_factors must be floating point")
        batch, stocks, factors = selected_factors.shape
        if factors != self.num_factors:
            raise ValueError(
                f"expected {self.num_factors} factors, received {factors}"
            )
        if asset_mask.shape != (batch, stocks):
            raise ValueError("asset_mask must have shape [B, N]")
        valid_direction_shapes = {
            (self.num_horizons, self.num_factors),
            (batch, self.num_horizons, self.num_factors),
            (1, self.num_horizons, self.num_factors),
        }
        if tuple(factor_directions.shape) not in valid_direction_shapes:
            raise ValueError("factor_directions must have shape [K, M] or [B, K, M]")
        if not factor_directions.is_floating_point():
            raise TypeError("factor_directions must be floating point")
        if not (
            selected_factors.device == asset_mask.device == factor_directions.device
        ):
            raise ValueError("all inputs must be on the same device")
        return batch, stocks

    def _expand_directions(
        self,
        directions: Tensor,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if directions.ndim == 2:
            directions = directions.unsqueeze(0)
        return directions.to(device=device, dtype=dtype).expand(
            batch, self.num_horizons, self.num_factors
        )

    def extra_repr(self) -> str:
        """Return concise module configuration for model summaries."""
        return (
            f"num_factors={self.num_factors}, num_horizons={self.num_horizons}, "
            f"leaky_relu_slope={self.leaky_relu_slope}, "
            f"zscore_approximation_input={self.zscore_approximation_input}"
        )
