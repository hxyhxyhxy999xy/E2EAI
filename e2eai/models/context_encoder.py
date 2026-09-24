"""Stock context encoder used by the E2EAI model."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from e2eai.models.normalization import MaskedCrossSectionalNorm

NormalizationMode = Literal["cross_sectional", "torch_batchnorm"]


class StockContextEncoder(nn.Module):
    """Encode selected factors into a masked stock representation.

    The default path implements E2EAI Eq. (3) using date-local cross-sectional
    normalization.  The optional BatchNorm path excludes padded rows, but—as a
    paper-comparison ablation—does pool valid observations across batch dates.

    Args:
        num_factors: Input factor width ``M``.
        hidden_dim: Output context width ``H``.
        normalization: Normalization implementation.
        dropout: Dropout probability between linear layers.
        eps: Numerical epsilon for normalization.
        leaky_relu_slope: Negative slope of the hidden activation.
    """

    def __init__(
        self,
        num_factors: int,
        hidden_dim: int,
        normalization: NormalizationMode = "cross_sectional",
        dropout: float = 0.1,
        eps: float = 1e-6,
        leaky_relu_slope: float = 0.2,
    ) -> None:
        super().__init__()
        if num_factors < 1 or hidden_dim < 1:
            raise ValueError("num_factors and hidden_dim must be positive")
        if normalization not in {"cross_sectional", "torch_batchnorm"}:
            raise ValueError(f"Unsupported normalization mode: {normalization}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if leaky_relu_slope < 0.0:
            raise ValueError("leaky_relu_slope must be nonnegative")

        self.num_factors = int(num_factors)
        self.hidden_dim = int(hidden_dim)
        self.normalization_mode = normalization
        if normalization == "cross_sectional":
            self.normalizer: nn.Module = MaskedCrossSectionalNorm(eps=eps)
        else:
            self.normalizer = nn.BatchNorm1d(num_factors, eps=eps)
        self.mlp = nn.Sequential(
            nn.Linear(num_factors, hidden_dim),
            nn.LeakyReLU(negative_slope=leaky_relu_slope),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _masked_batch_norm(self, values: Tensor, mask: Tensor) -> Tensor:
        """Apply BatchNorm only to valid rows and scatter exact-zero padding."""
        if not isinstance(self.normalizer, nn.BatchNorm1d):
            raise RuntimeError("BatchNorm helper called with a different normalizer")
        clean = torch.where(torch.isfinite(values), values, torch.zeros_like(values))
        valid_rows = mask.bool()
        normalized = torch.zeros_like(clean)
        selected = clean[valid_rows]
        if selected.shape[0] == 0:
            return normalized
        if self.training and selected.shape[0] < 2:
            selected_normalized = functional.batch_norm(
                selected,
                self.normalizer.running_mean,
                self.normalizer.running_var,
                self.normalizer.weight,
                self.normalizer.bias,
                training=False,
                momentum=0.0,
                eps=self.normalizer.eps,
            )
        else:
            selected_normalized = self.normalizer(selected)
        normalized[valid_rows] = selected_normalized
        return normalized

    def forward(self, selected_factors: Tensor, asset_mask: Tensor) -> Tensor:
        """Return stock contexts with shape ``[B,N,H]`` and zero padding."""
        squeeze_batch = selected_factors.ndim == 2
        values = selected_factors.unsqueeze(0) if squeeze_batch else selected_factors
        mask = asset_mask.unsqueeze(0) if asset_mask.ndim == 1 else asset_mask
        if values.ndim != 3:
            raise ValueError("selected_factors must have shape [B,N,M] or [N,M]")
        if values.shape[-1] != self.num_factors:
            raise ValueError(
                f"Expected {self.num_factors} factors, received {values.shape[-1]}"
            )
        if mask.ndim != 2 or mask.shape != values.shape[:2]:
            raise ValueError("asset_mask must match selected_factors dimensions [B,N]")

        # E2EAI Eq. (3): C_t = MLP(Norm(F_t)).
        if self.normalization_mode == "cross_sectional":
            if not isinstance(self.normalizer, MaskedCrossSectionalNorm):
                raise RuntimeError("Cross-sectional normalizer is not initialized")
            normalized = self.normalizer(values, mask)
        else:
            normalized = self._masked_batch_norm(values, mask)
        context = self.mlp(normalized)
        context = context * mask.bool().unsqueeze(-1).to(dtype=context.dtype)
        return context.squeeze(0) if squeeze_batch else context
