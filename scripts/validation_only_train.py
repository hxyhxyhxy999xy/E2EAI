"""Train one strict-cap CSI500 experiment and evaluate purged validation only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from e2eai.config import load_config
from e2eai.data.loaders import build_dataloaders
from e2eai.evaluation.index_evaluation import _factor_equal_weight_portfolio
from e2eai.evaluation.metrics import (
    compute_performance_metrics,
    maximum_drawdown,
    turnover_from_asset_weights,
)
from e2eai.models.e2eai import E2EAIModel
from e2eai.models.portfolio import project_capped_simplex
from e2eai.training.trainer import E2EAITrainer
from e2eai.utils.reproducibility import seed_everything
from e2eai.workflows import load_configured_data, move_batch_to_device, write_metrics
from rolling_train import _configure_fold


def _state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _scalar(value: torch.Tensor) -> float:
    return float(torch.nan_to_num(value.detach()).cpu())


@torch.no_grad()
def _validation_diagnostics(trainer, loader) -> tuple[dict, pd.DataFrame]:
    config = trainer.config
    horizons = config.model.horizons
    loss_totals: dict[str, float] = {}
    total_dates = 0
    rows: list[dict] = []
    previous_weights: dict[int, dict[str, float] | None] = {
        horizon: None for horizon in horizons
    }
    trainer.model.eval()
    for raw in loader:
        batch = move_batch_to_device(raw, trainer.device)
        factor_directions, deep_directions = trainer._direction_snapshot()
        output = trainer._forward_model(
            batch,
            factor_directions,
            deep_directions,
            cap_mode=config.model.portfolio.cap_mode_eval,
        )
        losses = trainer.loss_fn.from_output(
            output,
            batch["forward_returns"],
            deep_directions,
            batch["asset_mask"],
            batch["return_mask"],
            theta=config.model.portfolio.theta,
            differentiable_inner_loop=False,
        )
        batch_dates = int(batch["raw_factors"].shape[0])
        total_dates += batch_dates
        for name in (
            "total_loss", "portfolio_loss", "portfolio_return_loss", "upper_bound_loss",
            "stability_loss", "factor_return_loss", "attention_estimate_loss",
        ):
            loss_totals[name] = loss_totals.get(name, 0.0) + _scalar(losses[name]) * batch_dates

        detached = output.detached_cpu()
        asset_mask = batch["asset_mask"].detach().cpu().bool()
        return_mask = batch["return_mask"].detach().cpu().bool()
        forward_returns = batch["forward_returns"].detach().cpu()
        benchmark_returns = batch["benchmark_returns"].detach().cpu()
        ic = losses["ic_by_date_horizon"].detach().cpu()
        valid_ic = losses["valid_ic_by_date_horizon"].detach().cpu().bool()
        for b, date in enumerate(raw["dates"]):
            asset_ids = raw["asset_ids"][b]
            asset_count = len(asset_ids)
            valid_count = int(asset_mask[b].sum())
            if config.model.portfolio.gamma_p_mode == "inverse_n":
                threshold = config.model.portfolio.gamma_p / max(valid_count, 1)
            else:
                threshold = config.model.portfolio.gamma_p
            for k, horizon in enumerate(horizons):
                weights = detached.portfolio_weights[b, k]
                final_mask = detached.stock_selection_mask[b, k].bool()
                raw_mask = (detached.portfolio_attention[b, k] >= threshold) & asset_mask[b]
                label_valid = return_mask[b, k] & asset_mask[b]
                observed_weight = float(weights[label_valid].sum())
                portfolio_return = (
                    float((weights[label_valid] * forward_returns[b, k, label_valid]).sum() / observed_weight)
                    if observed_weight > 0 else np.nan
                )
                current_weights = {
                    asset_id: float(weight)
                    for asset_id, weight in zip(asset_ids, weights[:asset_count])
                }
                prior = previous_weights[horizon]
                turnover = (
                    {"one_way_turnover": np.nan, "gross_turnover": np.nan}
                    if prior is None
                    else turnover_from_asset_weights([prior, current_weights])
                )
                previous_weights[horizon] = current_weights
                rows.append({
                    "date": date,
                    "horizon": horizon,
                    "portfolio_return": portfolio_return,
                    "benchmark_return": float(benchmark_returns[b, k]),
                    "ic": float(ic[b, k]) if bool(valid_ic[b, k]) else np.nan,
                    "threshold": float(threshold),
                    "raw_threshold_count": int(raw_mask.sum()),
                    "final_selected_count": int(final_mask.sum()),
                    "feasibility_added_count": int((final_mask & ~raw_mask).sum()),
                    "effective_holdings": float(1.0 / weights.square().sum().clamp_min(1e-12)),
                    "max_weight": float(weights.max()),
                    "concentration_hhi": float(weights.square().sum()),
                    "cap_violation_count": int((weights > config.model.portfolio.theta + 1e-6).sum()),
                    "candidate_count": valid_count,
                    **turnover,
                })
    frame = pd.DataFrame(rows)
    losses = {name: value / total_dates for name, value in loss_totals.items()}
    losses.update({
        "weighted_upper_bound_loss": config.loss.lambda_up * losses["upper_bound_loss"],
        "weighted_stability_loss": config.loss.lambda_s * losses["stability_loss"],
        "weighted_factor_return_loss": config.loss.lambda_f * losses["factor_return_loss"],
        "weighted_attention_estimate_loss": config.loss.lambda_e * losses["attention_estimate_loss"],
    })
    by_horizon: dict[str, dict] = {}
    for horizon in horizons:
        selected = frame[frame.horizon == horizon]
        ic_values = selected.ic.dropna().to_numpy(dtype=float)
        performance_valid = (
            np.isfinite(selected.portfolio_return.to_numpy(dtype=float))
            & np.isfinite(selected.benchmark_return.to_numpy(dtype=float))
        )
        portfolio_returns = selected.loc[performance_valid, "portfolio_return"].to_numpy(dtype=float)
        benchmark_returns = selected.loc[performance_valid, "benchmark_return"].to_numpy(dtype=float)
        active = portfolio_returns - benchmark_returns
        annualization = 252.0 / horizon
        performance = compute_performance_metrics(
            portfolio_returns,
            benchmark_returns=benchmark_returns,
            annualization=annualization,
        )
        # Labels overlap by construction. Drawdown is therefore reported on
        # every-hth decision date, matching the existing index evaluation.
        non_overlapping_long = portfolio_returns[::horizon]
        non_overlapping_active = active[::horizon]
        by_horizon[str(horizon)] = {
            "mean_return": float(selected.portfolio_return.mean()),
            "mean_excess_return": float(np.mean(active)),
            "annualized_return": float(performance["annualized_return"]),
            "sharpe": float(performance["sharpe"]),
            "annualized_excess_return": float(performance["alpha"]),
            "excess_sharpe": float(performance["information_ratio"]),
            "max_drawdown": maximum_drawdown(non_overlapping_long),
            "max_active_drawdown": maximum_drawdown(non_overlapping_active),
            "one_way_turnover": float(selected.one_way_turnover.mean()),
            "gross_turnover": float(selected.gross_turnover.mean()),
            "mean_ic": float(np.mean(ic_values)) if len(ic_values) else np.nan,
            "ic_std": float(np.std(ic_values, ddof=0)) if len(ic_values) else np.nan,
            "icir": float(np.mean(ic_values) / np.std(ic_values, ddof=0))
            if len(ic_values) and np.std(ic_values, ddof=0) > 1e-12 else 0.0,
            "average_raw_threshold_count": float(selected.raw_threshold_count.mean()),
            "average_final_selected_count": float(selected.final_selected_count.mean()),
            "average_feasibility_added_count": float(selected.feasibility_added_count.mean()),
            "average_effective_holdings": float(selected.effective_holdings.mean()),
            "average_max_weight": float(selected.max_weight.mean()),
            "p95_max_weight": float(selected.max_weight.quantile(0.95)),
            "maximum_single_weight": float(selected.max_weight.max()),
            "cap_violation_count": int(selected.cap_violation_count.sum()),
            "threshold_min": float(selected.threshold.min()),
            "threshold_mean": float(selected.threshold.mean()),
            "threshold_max": float(selected.threshold.max()),
        }
    return {
        "losses": losses,
        "by_horizon": by_horizon,
        "performance_basis": (
            "Overlapping h-day validation labels; arithmetic annualization 252/h. "
            "Drawdown uses every h-th decision starting with the first validation date."
        ),
        "turnover_basis": (
            "Daily target-weight turnover by horizon on the stable asset-id union. "
            "One-way turnover is half gross turnover; first observation is excluded. "
            "No holding drift, execution constraints, or transaction costs."
        ),
    }, frame


@torch.no_grad()
def _capped_factor_baseline(loader, theta: float, horizons: list[int]) -> dict:
    records: list[dict] = []
    for raw in loader:
        factors = raw["raw_factors"].numpy()
        returns = raw["forward_returns"].numpy()
        return_mask = raw["return_mask"].numpy().astype(bool)
        asset_mask = raw["asset_mask"].numpy().astype(bool)
        benchmark = raw["benchmark_returns"].numpy()
        for b, date in enumerate(raw["dates"]):
            n = len(raw["asset_ids"][b])
            base = _factor_equal_weight_portfolio(factors[b, :n], asset_mask[b, :n])
            value = torch.as_tensor(base, dtype=torch.float32).unsqueeze(0)
            mask = torch.as_tensor(asset_mask[b, :n]).unsqueeze(0)
            weights = project_capped_simplex(value, mask, theta).squeeze(0).numpy()
            for k, horizon in enumerate(horizons):
                valid = return_mask[b, k, :n] & asset_mask[b, :n]
                observed = float(weights[valid].sum())
                portfolio_return = (
                    float(np.sum(weights[valid] * returns[b, k, :n][valid]) / observed)
                    if observed > 0 else np.nan
                )
                records.append({
                    "date": date, "horizon": horizon,
                    "portfolio_return": portfolio_return,
                    "benchmark_return": float(benchmark[b, k]),
                    "effective_holdings": float(1.0 / np.sum(weights * weights)),
                    "max_weight": float(weights.max()),
                    "cap_violation_count": int(np.sum(weights > theta + 1e-6)),
                })
    frame = pd.DataFrame(records)
    return {
        str(h): {
            "mean_return": float(frame.loc[frame.horizon == h, "portfolio_return"].mean()),
            "mean_excess_return": float(
                (frame.loc[frame.horizon == h, "portfolio_return"].to_numpy()
                 - frame.loc[frame.horizon == h, "benchmark_return"].to_numpy()).mean()
            ),
            "average_effective_holdings": float(frame.loc[frame.horizon == h, "effective_holdings"].mean()),
            "maximum_single_weight": float(frame.loc[frame.horizon == h, "max_weight"].max()),
            "cap_violation_count": int(frame.loc[frame.horizon == h, "cap_violation_count"].sum()),
        }
        for h in horizons
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper_index_universe_no_cap.yaml")
    parser.add_argument("--fold-config", default="configs/rolling_fold0_2018_2022.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", default="output/validation_sensitivity/csi500/fold_0")
    parser.add_argument("--theta", type=float, required=True)
    parser.add_argument("--gamma-p", type=float, required=True)
    parser.add_argument("--gamma-p-mode", choices=["fixed", "inverse_n"], required=True)
    parser.add_argument("--lambda-s", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    base = load_config(args.config)
    fold = yaml.safe_load(Path(args.fold_config).read_text(encoding="utf-8"))["folds"][0]
    destination = Path(args.output_root) / args.run_id
    config = _configure_fold(base, fold, Path(args.output_root), "csi500", args.device, None)
    data_root = Path(base.data.membership_path).parent
    config.data.index_universe = "csi500"
    config.data.membership_path = str(data_root / "csi500_membership.npz")
    config.data.benchmark_path = str(data_root / f"csi500_benchmark_{config.data.label_definition}.npy")
    config.model.portfolio.allocation_mode = "legacy_cap"
    config.model.portfolio.theta = args.theta
    config.model.portfolio.gamma_p = args.gamma_p
    config.model.portfolio.gamma_p_mode = args.gamma_p_mode
    config.model.portfolio.cap_mode_train = "capped_simplex"
    config.model.portfolio.cap_mode_eval = "capped_simplex"
    config.model.portfolio.min_selected_stocks = 1
    config.loss.lambda_s = args.lambda_s
    config.training.epochs = args.epochs
    config.training.early_stopping_patience = args.patience
    # A common selection metric is required because changing lambda_s changes
    # the numerical scale of validation_loss mechanically.
    config.training.validation_target = "validation_ir"
    config.logging.output_dir = str(destination)
    config.logging.tensorboard = False
    config.validate()

    destination.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed)
    source, _ = load_configured_data(config)
    loaders = build_dataloaders(config, source)
    model = E2EAIModel(config.model, config.ablation)
    initial_hash = _state_hash(model)
    trainer = E2EAITrainer(model, config)
    fit = trainer.fit(loaders.train, loaders.validation, epochs=args.epochs, output_dir=destination)
    trainer.load_checkpoint(destination / "best.pt", load_optimizer=False, restore_rng=False)
    diagnostics, by_date = _validation_diagnostics(trainer, loaders.validation)
    diagnostics["factor_equal_weight_same_cap"] = _capped_factor_baseline(
        loaders.validation, args.theta, config.model.horizons
    )
    diagnostics["experiment"] = {
        "run_id": args.run_id,
        "theta": args.theta,
        "gamma_p": args.gamma_p,
        "gamma_p_mode": args.gamma_p_mode,
        "lambda_s": args.lambda_s,
        "seed": config.seed,
        "initial_state_sha256": initial_hash,
        "best_epoch": fit.best_epoch,
        "best_validation_ir": fit.best_metric,
        "epochs_completed": len(fit.history),
        "stopped_early": fit.stopped_early,
        "oos_evaluated": False,
    }
    by_date.to_parquet(destination / "validation_by_date_horizon.parquet", index=False)
    pd.DataFrame(fit.history).to_parquet(destination / "epoch_history.parquet", index=False)
    (destination / "resolved_config.yaml").write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
    )
    (destination / "split_metadata.json").write_text(
        json.dumps(loaders.dataset.split_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_metrics(destination / "validation_metrics.json", diagnostics)
    trainer.close()
    print(json.dumps(diagnostics["experiment"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
