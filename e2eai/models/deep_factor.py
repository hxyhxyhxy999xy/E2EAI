"""Multi-horizon deep-factor heads from cascaded stock representations."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class MultiHorizonDeepFactor(nn.Module):
    """Generate an independent deep factor for each forward horizon.

    E2EAI Eq. (6) concatenates stock context, industry-neutral context, and
    universe-neutral context to width ``3H`` before applying a horizon-specific
    ``Linear(3H, 1)`` head and LeakyReLU.

    Args:
        hidden_dim: Width ``H`` of each input representation.
        horizons: Ordered positive forward horizons. Defaults to the paper setup.
        share_head: Explicit ablation that shares one final head across horizons.
        leaky_relu_slope: Negative slope after the final projection.

    Shape:
        - each representation: ``[B,N,H]`` or ``[N,H]``.
        - asset_mask: ``[B,N]`` or ``[N]``.
        - output: ``[B,K,N]`` or ``[K,N]``.
    """

    def __init__(
        self,
        hidden_dim: int,
        horizons: Sequence[int] = (3, 5, 10, 15, 20),
        share_head: bool = False,
        leaky_relu_slope: float = 0.2,
    ) -> None:
        super().__init__()
        horizon_tuple = tuple(int(horizon) for horizon in horizons)
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if not horizon_tuple or any(horizon <= 0 for horizon in horizon_tuple):
            raise ValueError("horizons must contain positive integers")
        if leaky_relu_slope < 0.0:
            raise ValueError("leaky_relu_slope must be nonnegative")

        self.hidden_dim = int(hidden_dim)
        self.horizons = horizon_tuple
        self.share_head = bool(share_head)
        number_of_heads = 1 if share_head else len(horizon_tuple)
        self.heads = nn.ModuleList(
            nn.Linear(3 * hidden_dim, 1) for _ in range(number_of_heads)
        )
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_slope)

    @property
    def num_horizons(self) -> int:
        """Return the number ``K`` of configured horizons."""
        return len(self.horizons)

    def forward(
        self,
        stock_context: Tensor,
        industry_neutral: Tensor,
        universe_neutral: Tensor,
        asset_mask: Tensor,
    ) -> Tensor:
        """Return deep-factor values with shape ``[B,K,N]``."""
        squeeze_batch = stock_context.ndim == 2
        context = stock_context.unsqueeze(0) if squeeze_batch else stock_context
        industry = (
            industry_neutral.unsqueeze(0)
            if industry_neutral.ndim == 2
            else industry_neutral
        )
        universe = (
            universe_neutral.unsqueeze(0)
            if universe_neutral.ndim == 2
            else universe_neutral
        )
        mask = asset_mask.unsqueeze(0) if asset_mask.ndim == 1 else asset_mask

        if context.ndim != 3:
            raise ValueError("representations must have shape [B,N,H] or [N,H]")
        if industry.shape != context.shape or universe.shape != context.shape:
            raise ValueError("all three representations must have identical shapes")
        if context.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Expected hidden width {self.hidden_dim}, received {context.shape[-1]}"
            )
        if mask.ndim != 2 or mask.shape != context.shape[:2]:
            raise ValueError("asset_mask must match representation dimensions [B,N]")

        clean_parts = tuple(
            torch.where(torch.isfinite(part), part, torch.zeros_like(part))
            for part in (context, industry, universe)
        )
        # E2EAI Eq. (6): f_k = LeakyReLU(W_k^T(C || C_I_bar || C_U_bar)).
        deep_input = torch.cat(clean_parts, dim=-1)
        if self.share_head:
            shared = self.activation(self.heads[0](deep_input).squeeze(-1))
            deep_factors = shared.unsqueeze(1).expand(-1, self.num_horizons, -1)
        else:
            deep_factors = torch.stack(
                [self.activation(head(deep_input).squeeze(-1)) for head in self.heads],
                dim=1,
            )
        deep_factors = deep_factors * mask.bool().unsqueeze(1).to(
            dtype=deep_factors.dtype
        )
        return deep_factors.squeeze(0) if squeeze_batch else deep_factors
