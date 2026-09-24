"""Padding-aware collation for batches of decision dates."""

from __future__ import annotations

from typing import Any

import torch

from e2eai.data.graphs import IndustryGraphBuilder, UniverseGraphBuilder


def collate_cross_sections(
    samples: list[dict[str, Any]],
    universe_graph_mode: str = "all_valid",
    unknown_industry_self_only: bool = True,
) -> dict[str, Any]:
    """Pad date cross-sections and construct both dynamic graphs."""
    if not samples:
        raise ValueError("Cannot collate an empty sample list")
    batch_size = len(samples)
    max_stocks = max(int(sample["raw_factors"].shape[0]) for sample in samples)
    num_factors = int(samples[0]["raw_factors"].shape[1])
    num_horizons = int(samples[0]["forward_returns"].shape[0])
    raw = torch.zeros(batch_size, max_stocks, num_factors, dtype=torch.float32)
    returns = torch.zeros(batch_size, num_horizons, max_stocks, dtype=torch.float32)
    return_mask = torch.zeros(batch_size, num_horizons, max_stocks, dtype=torch.bool)
    asset_mask = torch.zeros(batch_size, max_stocks, dtype=torch.bool)
    industry_ids = torch.full((batch_size, max_stocks), -1, dtype=torch.long)
    asset_ids: list[list[str]] = []
    dates: list[Any] = []
    benchmarks = torch.full((batch_size, num_horizons), torch.nan, dtype=torch.float32)
    has_benchmark = False
    execution_returns = torch.zeros(batch_size, max_stocks, dtype=torch.float32)
    execution_return_mask = torch.zeros(batch_size, max_stocks, dtype=torch.bool)
    has_execution = False
    for batch_index, sample in enumerate(samples):
        count = int(sample["raw_factors"].shape[0])
        if sample["raw_factors"].shape[1] != num_factors:
            raise ValueError("Every date must have the same factor width")
        raw[batch_index, :count] = sample["raw_factors"]
        returns[batch_index, :, :count] = sample["forward_returns"]
        return_mask[batch_index, :, :count] = sample["return_mask"]
        asset_mask[batch_index, :count] = True
        industry_ids[batch_index, :count] = sample["industry_ids"]
        asset_ids.append(list(sample["asset_ids"]))
        dates.append(sample["date"])
        if sample.get("benchmark_returns") is not None:
            benchmarks[batch_index] = sample["benchmark_returns"]
            has_benchmark = True
        if sample.get("execution_1d_return") is not None:
            execution_returns[batch_index, :count] = sample["execution_1d_return"]
            execution_return_mask[batch_index, :count] = sample["execution_return_mask"]
            has_execution = True
    industry_graph = IndustryGraphBuilder(
        include_self=True,
        unknown_self_only=unknown_industry_self_only,
    )(industry_ids, asset_mask)
    universe_graph = UniverseGraphBuilder(
        mode=universe_graph_mode,
        include_self=True,
    )(industry_ids, asset_mask)
    return {
        "raw_factors": raw,
        "forward_returns": returns,
        "return_mask": return_mask,
        "asset_mask": asset_mask,
        "asset_ids": asset_ids,
        "industry_ids": industry_ids,
        "industry_adjacency": industry_graph,
        "universe_adjacency": universe_graph,
        "dates": dates,
        "benchmark_returns": benchmarks if has_benchmark else None,
        "execution_1d_return": execution_returns if has_execution else None,
        "execution_return_mask": execution_return_mask if has_execution else None,
    }
