"""Small scoring heads used by daily-strategy baselines and ablations."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SimpleMLPScorer(nn.Module):
    """A minimal stockwise ``input -> hidden -> 1`` scorer."""

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim < 1:
            raise ValueError("input_dim and hidden_dim must be positive")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.output = nn.Linear(hidden_dim, 1)

    def encode(self, factors: Tensor, asset_mask: Tensor) -> Tensor:
        if factors.ndim != 3 or factors.shape[-1] != self.input_dim:
            raise ValueError("factors must have shape [B,N,input_dim]")
        clean = torch.where(torch.isfinite(factors), factors, torch.zeros_like(factors))
        hidden = self.activation(self.hidden(clean))
        return hidden * asset_mask.bool().unsqueeze(-1).to(hidden.dtype)

    def score(self, representation: Tensor, asset_mask: Tensor) -> Tensor:
        if representation.ndim != 3 or representation.shape[-1] != self.hidden_dim:
            raise ValueError("representation must have shape [B,N,hidden_dim]")
        score = self.output(representation).squeeze(-1)
        return torch.where(asset_mask.bool(), score, torch.zeros_like(score))

    def forward(self, factors: Tensor, asset_mask: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.encode(factors, asset_mask)
        return hidden, self.score(hidden, asset_mask)


def factor_mean_scores(factors: Tensor, asset_mask: Tensor) -> Tensor:
    """Return the equal-factor score used by the non-learned baseline."""
    if factors.ndim != 3:
        raise ValueError("factors must have shape [B,N,M]")
    valid = torch.isfinite(factors)
    clean = torch.where(valid, factors, torch.zeros_like(factors))
    score = clean.sum(dim=-1) / valid.sum(dim=-1).clamp_min(1).to(clean.dtype)
    return torch.where(asset_mask.bool(), score, torch.zeros_like(score))


__all__ = ["SimpleMLPScorer", "factor_mean_scores"]
