"""Dynamic industry and universe graph construction."""

from __future__ import annotations

import torch
from torch import Tensor


class IndustryGraphBuilder:
    """Build same-industry adjacency matrices for changing universes."""

    def __init__(self, include_self: bool = True, unknown_self_only: bool = True) -> None:
        self.include_self = include_self
        self.unknown_self_only = unknown_self_only

    def __call__(self, industry_ids: Tensor, asset_mask: Tensor) -> Tensor:
        """Return boolean adjacency with shape ``[B,N,N]`` or ``[N,N]``."""
        squeeze = industry_ids.ndim == 1
        ids = industry_ids.unsqueeze(0) if squeeze else industry_ids
        valid = asset_mask.unsqueeze(0) if squeeze else asset_mask
        if ids.ndim != 2 or valid.shape != ids.shape:
            raise ValueError("industry_ids and asset_mask must have matching [B,N] shapes")
        adjacency = (ids.unsqueeze(-1) == ids.unsqueeze(-2))
        pair_valid = valid.unsqueeze(-1) & valid.unsqueeze(-2)
        adjacency &= pair_valid
        if self.unknown_self_only:
            known_pair = (ids.unsqueeze(-1) >= 0) & (ids.unsqueeze(-2) >= 0)
            eye = torch.eye(ids.shape[-1], dtype=torch.bool, device=ids.device).unsqueeze(0)
            adjacency &= known_pair | eye
        if not self.include_self:
            eye = torch.eye(ids.shape[-1], dtype=torch.bool, device=ids.device).unsqueeze(0)
            adjacency &= ~eye
        return adjacency.squeeze(0) if squeeze else adjacency


class UniverseGraphBuilder:
    """Build full-universe or cross-industry adjacency matrices."""

    def __init__(
        self,
        mode: str = "all_valid",
        include_self: bool = True,
    ) -> None:
        if mode not in {"all_valid", "cross_industry_only"}:
            raise ValueError(f"Unsupported universe graph mode: {mode}")
        self.mode = mode
        self.include_self = include_self

    def __call__(self, industry_ids: Tensor, asset_mask: Tensor) -> Tensor:
        """Return boolean adjacency with shape ``[B,N,N]`` or ``[N,N]``."""
        squeeze = industry_ids.ndim == 1
        ids = industry_ids.unsqueeze(0) if squeeze else industry_ids
        valid = asset_mask.unsqueeze(0) if squeeze else asset_mask
        pair_valid = valid.unsqueeze(-1) & valid.unsqueeze(-2)
        if self.mode == "all_valid":
            adjacency = pair_valid.clone()
        else:
            adjacency = pair_valid & (ids.unsqueeze(-1) != ids.unsqueeze(-2))
        eye = torch.eye(ids.shape[-1], dtype=torch.bool, device=ids.device).unsqueeze(0)
        if self.include_self:
            adjacency |= eye & pair_valid
        else:
            adjacency &= ~eye
        return adjacency.squeeze(0) if squeeze else adjacency

