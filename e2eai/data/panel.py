"""AlphaGAT-style dense panel adapter with memory-mapped directory support."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from e2eai.config import DataConfig
from e2eai.data.index_universe import IndexMembership


def _decode(values: np.ndarray) -> np.ndarray:
    """Return stable Unicode strings for bytes or string-like arrays."""
    array = np.asarray(values)
    if array.dtype.kind == "S":
        return np.char.decode(array, "utf-8")
    return array.astype(str)


def _validated_manifest_factor_names(config: DataConfig) -> tuple[str, ...] | None:
    """Read the frozen factor list when a daily-strategy manifest is supplied."""
    if not config.factor_selection_manifest_path:
        return None
    manifest_path = Path(config.factor_selection_manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing factor-selection manifest: {manifest_path}")
    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    factor_list_path = manifest.get("factor_list_path")
    if not isinstance(factor_list_path, str):
        raise ValueError("factor-selection manifest must contain factor_list_path")
    list_path = Path(factor_list_path)
    if not list_path.is_file():
        raise FileNotFoundError(f"Missing frozen factor list: {list_path}")
    values = json.loads(list_path.read_text(encoding="utf-8"))
    names = values.get("factor_names") if isinstance(values, dict) else values
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError("factor_list.json must contain a string factor_names list")
    return tuple(names)


def _factor_direction_vector(config: DataConfig, factor_names: tuple[str, ...]) -> np.ndarray | None:
    """Return a frozen external direction vector without altering stored panel values."""
    if not config.apply_factor_direction:
        return None
    if not config.factor_directions_path:
        raise ValueError("apply_factor_direction requires factor_directions_path")
    direction_path = Path(config.factor_directions_path)
    if not direction_path.is_file():
        raise FileNotFoundError(f"Missing factor directions: {direction_path}")
    import json

    values = json.loads(direction_path.read_text(encoding="utf-8"))
    mapping = values.get("directions", values) if isinstance(values, dict) else None
    if not isinstance(mapping, dict):
        raise ValueError("factor_directions.json must be a mapping or contain directions")
    missing = [name for name in factor_names if name not in mapping]
    if missing:
        raise ValueError(f"Factor directions miss panel factors: {missing[:5]}")
    vector = np.asarray([mapping[name] for name in factor_names], dtype=np.float32)
    if not np.all(np.isin(vector, [-1.0, 1.0])):
        raise ValueError("Frozen factor directions must be -1 or +1")
    return vector


@dataclass(frozen=True)
class IndustryLookup:
    """Lazy as-of lookup for the external ``index/columns/values`` industry NPZ."""

    dates_ns: np.ndarray
    values: np.ndarray
    panel_asset_positions: np.ndarray

    def row(self, date: pd.Timestamp) -> np.ndarray:
        output = np.full(len(self.panel_asset_positions), -1, dtype=np.int64)
        timestamp = np.datetime64(date.to_datetime64(), "ns").astype(np.int64)
        row_index = int(np.searchsorted(self.dates_ns, timestamp, side="right") - 1)
        if row_index < 0:
            return output
        valid_assets = self.panel_asset_positions >= 0
        if not valid_assets.any():
            return output
        selected = self.values[row_index, self.panel_asset_positions[valid_assets]]
        if np.issubdtype(selected.dtype, np.number):
            finite = np.isfinite(selected)
            target = np.flatnonzero(valid_assets)[finite]
            output[target] = np.rint(selected[finite]).astype(np.int64)
        else:
            text = _decode(selected)
            nonmissing = (text != "") & (text != "nan") & (text != "None")
            target = np.flatnonzero(valid_assets)[nonmissing]
            # Stable per-value IDs; equality, not their magnitude, defines graph edges.
            output[target] = np.asarray(
                [
                    int.from_bytes(value.encode("utf-8")[:8], "little", signed=True)
                    for value in text[nonmissing]
                ],
                dtype=np.int64,
            )
        return output


@dataclass(frozen=True)
class AlphaPanel:
    """Validated arrays from an AlphaGAT Stage-II panel."""

    dates: tuple[pd.Timestamp, ...]
    asset_ids: np.ndarray
    alpha_names: tuple[str, ...]
    alpha_tensor: np.ndarray
    forward_returns: np.ndarray
    execution_returns: np.ndarray | None
    benchmark_returns: np.ndarray | None
    eligible_mask: np.ndarray | None
    factor_valid_mask: np.ndarray | None
    alpha_directions: np.ndarray | None
    industry_ids: np.ndarray | None
    industry_lookup: IndustryLookup | None
    market_cap: np.ndarray | None


def _load_industry_lookup(path: Path, panel_assets: np.ndarray) -> IndustryLookup:
    with np.load(path, allow_pickle=False) as archive:
        required = {"index", "columns", "values"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"Industry NPZ is missing keys: {sorted(missing)}")
        source_dates = pd.to_datetime(_decode(archive["index"]), errors="raise")
        source_assets = _decode(archive["columns"])
        values = np.asarray(archive["values"])
    if values.shape != (len(source_dates), len(source_assets)):
        raise ValueError(
            "Industry values must have shape [num_dates, num_assets]; got "
            f"{values.shape}"
        )
    if not source_dates.is_monotonic_increasing or source_dates.has_duplicates:
        raise ValueError("Industry NPZ dates must be unique and increasing")
    source_position = {value: index for index, value in enumerate(source_assets.tolist())}
    panel_positions = np.asarray(
        [source_position.get(value, -1) for value in panel_assets.tolist()], dtype=np.int64
    )
    return IndustryLookup(
        dates_ns=source_dates.to_numpy(dtype="datetime64[ns]").astype(np.int64),
        values=values,
        panel_asset_positions=panel_positions,
    )


def load_alpha_panel(path: str | Path, config: DataConfig) -> AlphaPanel:
    """Load an AlphaGAT directory of NPY files or a single NPZ archive.

    Directory arrays remain memory mapped.  A train-only daily-strategy panel
    stores raw processed values; its optional frozen direction vector is
    applied later by :class:`AlphaPanelDataset` only when the configuration
    explicitly sets ``apply_factor_direction``.
    """
    source = Path(path)
    archive: Any | None = None
    if source.is_dir():
        def read(key: str, required: bool = False) -> np.ndarray | None:
            target = source / f"{key}.npy"
            if not target.is_file():
                if required:
                    raise FileNotFoundError(f"Required panel array is missing: {target}")
                return None
            return np.load(target, mmap_mode=config.mmap_mode, allow_pickle=False)
    elif source.suffix.lower() == ".npz" and source.is_file():
        archive = np.load(source, allow_pickle=False)

        def read(key: str, required: bool = False) -> np.ndarray | None:
            if key not in archive.files:
                if required:
                    raise KeyError(f"Required panel key is missing: {key}")
                return None
            return np.asarray(archive[key])
    else:
        raise ValueError("Alpha panel path must be a directory of .npy files or a .npz file")

    try:
        dates_array = read("dates", required=True)
        assets_array = read("asset_ids", required=True)
        alpha = read(config.alpha_key, required=True)
        # A daily strategy may carry multi-horizon labels in an external file.
        # In that case an in-panel forward-return array is optional and no
        # duplicate execution-return matrix is required in a new panel.
        returns = read(config.return_key, required=config.return_path is None)
        execution_returns = read(config.execution_return_key)
        names_array = read(config.alpha_names_key)
        benchmark = read(config.benchmark_key)
        eligible = read(config.eligibility_key)
        factor_valid = read(config.factor_validity_key)
        directions = read(config.alpha_directions_key)
        industry_ids = read(config.industry_ids_key)
        market_cap = read(config.market_cap_key)
        start_dates = read("forward_return_start_dates")
        end_dates = read("forward_return_end_dates")
    finally:
        if archive is not None:
            archive.close()

    if config.return_path:
        returns = np.load(
            Path(config.return_path), mmap_mode=config.mmap_mode, allow_pickle=False
        )
    if config.execution_return_path:
        execution_returns = np.load(
            Path(config.execution_return_path), mmap_mode=config.mmap_mode, allow_pickle=False
        )
    if execution_returns is not None and execution_returns.ndim == 3:
        if execution_returns.shape[1] != 1:
            raise ValueError("3-D execution returns must have shape [T,1,N]")
        execution_returns = execution_returns[:, 0, :]
    if config.benchmark_path:
        benchmark = np.load(
            Path(config.benchmark_path), mmap_mode=config.mmap_mode, allow_pickle=False
        )

    assert dates_array is not None and assets_array is not None
    assert alpha is not None and returns is not None
    date_index = pd.DatetimeIndex(pd.to_datetime(_decode(dates_array), errors="raise"))
    assets = _decode(assets_array)
    if not date_index.is_monotonic_increasing or date_index.has_duplicates:
        raise ValueError("Panel dates must be unique and increasing")
    if len(set(assets.tolist())) != len(assets):
        raise ValueError("Panel asset_ids must be unique")
    if alpha.ndim != 3:
        raise ValueError(f"alpha_tensor must have shape [T,M,N]; got {alpha.shape}")
    num_dates, num_factors, num_assets = alpha.shape
    if num_dates != len(date_index) or num_assets != len(assets):
        raise ValueError("alpha_tensor axes do not align with dates and asset_ids")
    if returns.ndim == 2:
        expected_return_tail = (num_dates, num_assets)
        if returns.shape != expected_return_tail:
            raise ValueError(
                f"2-D forward_returns must have shape {expected_return_tail}; got {returns.shape}"
            )
    elif returns.ndim == 3:
        if returns.shape[0] != num_dates or returns.shape[2] != num_assets:
            raise ValueError("3-D forward_returns must have shape [T,H,N]")
    else:
        raise ValueError("forward_returns must have shape [T,N] or [T,H,N]")

    names = (
        tuple(_decode(names_array).tolist())
        if names_array is not None
        else tuple(f"alpha_{index:03d}" for index in range(num_factors))
    )
    if len(names) != num_factors:
        raise ValueError("alpha_names length must equal alpha_tensor factor width")
    manifest_names = _validated_manifest_factor_names(config)
    if manifest_names is not None and names != manifest_names:
        raise ValueError("Panel alpha_names do not exactly match the frozen factor manifest")

    def validate_optional(value: np.ndarray | None, shape: tuple[int, ...], label: str) -> None:
        if value is not None and value.shape != shape:
            raise ValueError(f"{label} must have shape {shape}; got {value.shape}")

    validate_optional(eligible, (num_dates, num_assets), config.eligibility_key)
    validate_optional(factor_valid, (num_dates, num_factors, num_assets), config.factor_validity_key)
    validate_optional(industry_ids, (num_dates, num_assets), config.industry_ids_key)
    validate_optional(market_cap, (num_dates, num_assets), config.market_cap_key)
    validate_optional(
        execution_returns,
        (num_dates, num_assets),
        config.execution_return_key,
    )
    if directions is not None and directions.shape != (num_factors,):
        raise ValueError("alpha_directions must have shape [M]")
    if benchmark is not None:
        allowed = {(num_dates,), (num_dates, returns.shape[1])} if returns.ndim == 3 else {(num_dates,)}
        if benchmark.shape not in allowed:
            raise ValueError(f"benchmark_returns has incompatible shape {benchmark.shape}")

    for boundary, label in ((start_dates, "forward_return_start_dates"), (end_dates, "forward_return_end_dates")):
        if boundary is not None and boundary.shape != (num_dates,):
            raise ValueError(f"{label} must have shape [T]")
    if start_dates is not None:
        starts = pd.DatetimeIndex(pd.to_datetime(_decode(start_dates), errors="raise"))
        if np.any(starts <= date_index):
            raise ValueError("Every forward-return start date must be after its decision date")
    if start_dates is not None and end_dates is not None:
        ends = pd.DatetimeIndex(pd.to_datetime(_decode(end_dates), errors="raise"))
        if np.any(ends < starts):
            raise ValueError("Forward-return end dates must not precede start dates")

    industry_lookup = None
    if industry_ids is None and config.industry_npz_path:
        industry_lookup = _load_industry_lookup(Path(config.industry_npz_path), assets)
    if config.asset_selection == "market_cap" and market_cap is None:
        raise ValueError(
            f"asset_selection='market_cap' requires panel array {config.market_cap_key!r}"
        )
    return AlphaPanel(
        dates=tuple(pd.Timestamp(value) for value in date_index),
        asset_ids=assets,
        alpha_names=names,
        alpha_tensor=alpha,
        forward_returns=returns,
        execution_returns=execution_returns,
        benchmark_returns=benchmark,
        eligible_mask=eligible,
        factor_valid_mask=factor_valid,
        alpha_directions=directions,
        industry_ids=industry_ids,
        industry_lookup=industry_lookup,
        market_cap=market_cap,
    )


class AlphaPanelDataset(Dataset[dict[str, Any]]):
    """Expose a dense panel as one variable-width cross-section per decision date."""

    def __init__(
        self,
        panel: AlphaPanel,
        horizons: list[int],
        config: DataConfig,
        execution_horizon: int | None = None,
    ) -> None:
        self.panel = panel
        self.horizons = tuple(int(value) for value in horizons)
        self.factor_columns = panel.alpha_names
        self.factor_direction_vector = _factor_direction_vector(config, self.factor_columns)
        self.dates = panel.dates
        self.max_assets_per_date = config.max_assets_per_date
        self.asset_selection = config.asset_selection
        self.single_return_policy = config.single_return_policy
        self.execution_horizon = execution_horizon
        # A pure execution-only daily core has no multi-day panel labels.  In
        # that case the only return is supplied through ``execution_returns``
        # and the non-execution horizon set is intentionally empty.  For the
        # historical/general case, infer panel horizons from the model
        # horizons after removing the dedicated execution horizon.
        inferred_panel_horizons = tuple(
            horizon for horizon in self.horizons if horizon != execution_horizon
        )
        self.panel_return_horizons = tuple(
            int(value)
            for value in (
                config.panel_return_horizons
                if config.panel_return_horizons
                else inferred_panel_horizons
            )
        )
        self.membership = (IndexMembership(config.membership_path, config.index_universe)
                           if config.index_universe else None)
        if self.membership and (config.max_assets_per_date is not None or config.asset_selection != 'axis_order'):
            raise ValueError('Index candidates must not be pre-truncated')
        self._asset_position = {asset: i for i, asset in enumerate(panel.asset_ids)}
        model_panel_horizons = tuple(
            horizon for horizon in self.horizons if horizon != execution_horizon
        )
        if model_panel_horizons != self.panel_return_horizons:
            raise ValueError(
                "data.panel_return_horizons must exactly match the non-execution "
                f"model horizons; got {list(self.panel_return_horizons)} and "
                f"{list(model_panel_horizons)}"
            )
        if execution_horizon is not None:
            if panel.execution_returns is None:
                raise ValueError(
                    "model.execution_horizon requires data.execution_return_path or "
                    f"panel key {config.execution_return_key!r}"
                )
        if execution_horizon is not None and self.horizons.count(execution_horizon) != 1:
            raise ValueError(
                "model.horizons must contain execution_horizon exactly once"
            )
        if panel.forward_returns.ndim == 2:
            # In execution-only mode the 2-D array is a compatibility
            # placeholder; the actual h=2 labels are read from
            # ``panel.execution_returns`` below.
            if len(self.panel_return_horizons) == 0 and execution_horizon is not None:
                pass
            elif len(self.panel_return_horizons) != 1 and self.single_return_policy != "broadcast":
                raise ValueError(
                    "The panel has one forward-return label, but the model has "
                    f"{len(self.panel_return_horizons)} panel horizons. Configure one panel horizon or explicitly set "
                    "data.single_return_policy='broadcast'."
                )
        elif panel.forward_returns.shape[1] != len(self.panel_return_horizons):
            raise ValueError(
                "The panel return-horizon width must equal model.horizons; got "
                f"{panel.forward_returns.shape[1]} and {len(self.panel_return_horizons)}"
            )

    def __len__(self) -> int:
        return len(self.dates)

    def _selected_assets(self, index: int) -> np.ndarray:
        if self.membership is not None:
            positions, _ = self.universe_diagnostics(index)
            if len(positions) == 0:
                raise ValueError(f'No valid index candidates on {self.dates[index]}')
            return positions
        num_assets = len(self.panel.asset_ids)
        eligible = (
            np.ones(num_assets, dtype=bool)
            if self.panel.eligible_mask is None
            else np.asarray(self.panel.eligible_mask[index], dtype=bool)
        )
        positions = np.flatnonzero(eligible)
        if positions.size == 0:
            raise ValueError(f"No eligible assets on panel date {self.dates[index].date()}")
        limit = self.max_assets_per_date
        if limit is not None and positions.size > limit:
            if self.asset_selection == "market_cap":
                assert self.panel.market_cap is not None
                caps = np.nan_to_num(
                    np.asarray(self.panel.market_cap[index, positions], dtype=np.float64),
                    nan=-np.inf,
                    posinf=np.inf,
                    neginf=-np.inf,
                )
                order = np.argsort(-caps, kind="stable")
                positions = positions[order[:limit]]
            else:
                positions = positions[:limit]
        return positions

    def universe_diagnostics(self, index):
        if self.membership is None:
            raise ValueError('Index diagnostics require dated membership')
        date = self.dates[index]
        members = self.membership.asset_set(date)
        matched = np.asarray(sorted(self._asset_position[a] for a in members
                                    if a in self._asset_position), dtype=np.int64)
        eligible = (np.ones(len(matched), bool) if self.panel.eligible_mask is None
                    else np.asarray(self.panel.eligible_mask[index, matched], bool))
        factors = np.asarray(self.panel.alpha_tensor[index])[:, matched]
        valid = np.isfinite(factors)
        if self.panel.factor_valid_mask is not None:
            valid &= np.asarray(self.panel.factor_valid_mask[index])[:, matched]
        # Partially observed rows keep the existing zero imputation. Entirely
        # missing factor rows are excluded, using only decision-date features.
        has_factors = valid.any(axis=0)
        positions = matched[eligible & has_factors]
        if self.panel.industry_ids is not None:
            industries = np.asarray(self.panel.industry_ids[index])[positions]
        elif self.panel.industry_lookup is not None:
            industries = self.panel.industry_lookup.row(date)[positions]
        else:
            industries = np.full(len(positions), -1)
        return positions, {
            'date': date, 'index': self.membership.index_name,
            'raw_members': len(members), 'matched_panel': len(matched),
            'tradable_members': int(eligible.sum()),
            'missing_factor_stocks': len(members)-len(matched)+int((~has_factors).sum()),
            'partial_factor_stocks': int((has_factors & ~valid.all(axis=0)).sum()),
            'missing_industry_stocks': int(((industries < 0) | ~np.isfinite(industries)).sum()),
            'candidate_count': len(positions),
            'index_adjustment_day': self.membership.changed(date),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        positions = self._selected_assets(index)
        factor_values = np.asarray(self.panel.alpha_tensor[index, :, positions], dtype=np.float32)
        # NumPy's advanced indexing moves the selected-asset axis to the front.
        if factor_values.shape == (len(positions), len(self.factor_columns)):
            raw_values = factor_values
        else:
            raw_values = factor_values.T
        if self.panel.factor_valid_mask is not None:
            validity = np.asarray(
                self.panel.factor_valid_mask[index, :, positions], dtype=bool
            )
            if validity.shape != raw_values.shape:
                validity = validity.T
            raw_values = np.where(validity, raw_values, 0.0)
        raw_values = np.nan_to_num(raw_values, nan=0.0, posinf=0.0, neginf=0.0)
        if self.factor_direction_vector is not None:
            raw_values = raw_values * self.factor_direction_vector[None, :]

        if self.panel.forward_returns.ndim == 2:
            one_return = np.asarray(
                self.panel.forward_returns[index, positions], dtype=np.float32
            )[None, :]
            returns = (
                np.repeat(one_return, len(self.panel_return_horizons), axis=0)
                if len(self.panel_return_horizons) > 1
                else one_return
            )
        else:
            panel_returns = np.asarray(
                self.panel.forward_returns[index, :, positions], dtype=np.float32
            )
            if panel_returns.shape == (len(positions), len(self.panel_return_horizons)):
                panel_returns = panel_returns.T
            returns = panel_returns
        by_horizon = {
            horizon: returns[position]
            for position, horizon in enumerate(self.panel_return_horizons)
        }
        execution = None
        if self.execution_horizon is not None:
            assert self.panel.execution_returns is not None
            execution = np.asarray(
                self.panel.execution_returns[index, positions], dtype=np.float32
            )
            by_horizon[int(self.execution_horizon)] = execution
        returns = np.stack([by_horizon[horizon] for horizon in self.horizons], axis=0)
        return_mask = np.isfinite(returns)
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

        benchmark = None
        if self.panel.benchmark_returns is not None:
            source = self.panel.benchmark_returns
            if source.ndim == 1:
                panel_value = np.repeat(
                    np.asarray([source[index]], dtype=np.float32),
                    len(self.panel_return_horizons),
                )
            else:
                panel_value = np.asarray(source[index], dtype=np.float32)
            benchmark_by_horizon = {
                horizon: panel_value[position]
                for position, horizon in enumerate(self.panel_return_horizons)
            }
            # A standalone 1-D benchmark is the execution benchmark for the
            # dedicated h=2 daily horizon, not a panel multi-horizon label.
            if source.ndim == 1 and self.execution_horizon is not None:
                benchmark_by_horizon[int(self.execution_horizon)] = float(source[index])
            value = np.asarray(
                [benchmark_by_horizon.get(horizon, np.nan) for horizon in self.horizons],
                dtype=np.float32,
            )
            benchmark = torch.from_numpy(value.copy())

        if self.panel.industry_ids is not None:
            all_industries = np.asarray(self.panel.industry_ids[index], dtype=np.float64)
            selected_industries = np.nan_to_num(all_industries[positions], nan=-1).astype(np.int64)
        elif self.panel.industry_lookup is not None:
            selected_industries = self.panel.industry_lookup.row(self.dates[index])[positions]
        else:
            selected_industries = np.full(len(positions), -1, dtype=np.int64)

        execution_mask = None if execution is None else np.isfinite(execution)
        clean_execution = (
            None
            if execution is None
            else np.nan_to_num(execution, nan=0.0, posinf=0.0, neginf=0.0)
        )
        return {
            "raw_factors": torch.from_numpy(np.ascontiguousarray(raw_values)),
            "forward_returns": torch.from_numpy(np.ascontiguousarray(returns)),
            "return_mask": torch.from_numpy(np.ascontiguousarray(return_mask)),
            "asset_mask": torch.ones(len(positions), dtype=torch.bool),
            "asset_ids": self.panel.asset_ids[positions].tolist(),
            "industry_ids": torch.from_numpy(np.ascontiguousarray(selected_industries)),
            "date": self.dates[index],
            "benchmark_returns": benchmark,
            "execution_1d_return": (
                None
                if clean_execution is None
                else torch.from_numpy(np.ascontiguousarray(clean_execution))
            ),
            "execution_return_mask": (
                None
                if execution_mask is None
                else torch.from_numpy(np.ascontiguousarray(execution_mask))
            ),
        }
