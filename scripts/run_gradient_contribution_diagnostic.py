"""Diagnose Experiment C score/portfolio gradient contributions.

This script is intentionally read-only with respect to Experiment C checkpoints:
it loads the three saved model states, performs separate score and gross-return
backward passes on deterministic *training* batches, and never creates an
optimizer or calls ``optimizer.step``.  It writes all diagnostics below the
existing ``experiment_c/gradient_diagnostic`` directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from e2eai.config import load_config
from e2eai.data.loaders import build_dataloaders
from e2eai.models.e2eai import E2EAIModel
from e2eai.utils.reproducibility import resolve_device, seed_everything
from e2eai.workflows import load_configured_data, write_metrics
from scripts.run_daily_experiment_c import (
    _gross_loss,
    _mlp_parameters,
    _raw_scores,
    _score_loss,
    _sha256,
    _move_batch,
)


def _date_string(value: Any) -> str:
    try:
        return pd.Timestamp(value).date().isoformat()
    except Exception:
        return str(value)


def _finite_number(value: float, label: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise FloatingPointError(f"non-finite {label}: {value}")
    return value


def _parameter_layout(model: E2EAIModel) -> list[tuple[str, torch.nn.Parameter]]:
    if model.simple_mlp_scorer is None:
        raise RuntimeError("Experiment C simple_mlp_scorer is required")
    layout = [
        (name, parameter)
        for name, parameter in model.simple_mlp_scorer.named_parameters()
        if parameter.requires_grad
    ]
    if not layout:
        raise RuntimeError("No trainable MLP scorer parameters")
    return layout


def _gradient_vector(layout: list[tuple[str, torch.nn.Parameter]], label: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    values: list[np.ndarray] = []
    by_name: dict[str, np.ndarray] = {}
    for name, parameter in layout:
        if parameter.grad is None:
            raise RuntimeError(f"missing {label} gradient for {name}")
        value = parameter.grad.detach().cpu().numpy().astype(np.float64, copy=True).reshape(-1)
        if not np.isfinite(value).all():
            raise FloatingPointError(f"non-finite {label} gradient for {name}")
        by_name[name] = value
        values.append(value)
    vector = np.concatenate(values)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0:
        raise FloatingPointError(f"{label} gradient norm is not positive: {norm}")
    return vector, by_name


def _loss_gradient(
    model: E2EAIModel,
    batch: dict[str, Any],
    layout: list[tuple[str, torch.nn.Parameter]],
    kind: str,
) -> tuple[float, np.ndarray, dict[str, np.ndarray]]:
    model.zero_grad(set_to_none=True)
    scores = _raw_scores(model, batch)
    if kind == "score":
        loss, _ = _score_loss(scores, batch)
    elif kind == "gross":
        loss, _, _, _ = _gross_loss(scores, batch)
    else:
        raise ValueError(kind)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite {kind} loss")
    loss.backward()
    vector, by_name = _gradient_vector(layout, kind)
    value = _finite_number(loss.detach().cpu(), f"{kind} loss")
    model.zero_grad(set_to_none=True)
    return value, vector, by_name


def _component_metrics(g_score: np.ndarray, g_gross_raw: np.ndarray, lambda_portfolio: float) -> dict[str, float]:
    weighted = lambda_portfolio * g_gross_raw
    total = g_score + weighted
    score_norm = float(np.linalg.norm(g_score))
    raw_norm = float(np.linalg.norm(g_gross_raw))
    weighted_norm = float(np.linalg.norm(weighted))
    total_norm = float(np.linalg.norm(total))
    dot = float(np.dot(g_score, weighted))
    cosine = dot / (score_norm * weighted_norm)
    cancellation = total_norm / (score_norm + weighted_norm)
    if not all(np.isfinite(x) for x in (score_norm, raw_norm, weighted_norm, total_norm, cosine, cancellation)):
        raise FloatingPointError("non-finite gradient contribution metric")
    return {
        "score_grad_norm": score_norm,
        "gross_grad_norm_raw": raw_norm,
        "gross_grad_norm_weighted": weighted_norm,
        "gradient_ratio": weighted_norm / score_norm,
        "gradient_cosine": cosine,
        "total_grad_norm": total_norm,
        "cancellation_ratio": cancellation,
        "portfolio_on_score_projection": dot / score_norm,
        "score_on_portfolio_projection": dot / weighted_norm,
    }


def _layer_metrics(
    score_by_name: dict[str, np.ndarray],
    gross_by_name: dict[str, np.ndarray],
    lambda_portfolio: float,
) -> list[dict[str, float | str]]:
    groups = {
        "first_linear": ["hidden.weight", "hidden.bias"],
        "output_linear": ["output.weight", "output.bias"],
    }
    rows: list[dict[str, float | str]] = []
    for layer, names in groups.items():
        if any(name not in score_by_name or name not in gross_by_name for name in names):
            raise RuntimeError(f"Expected scorer parameter names not found for {layer}: {names}")
        g_score = np.concatenate([score_by_name[name] for name in names])
        g_gross = np.concatenate([gross_by_name[name] for name in names])
        rows.append(
            {
                "layer": layer,
                "parameter_names": ",".join(f"simple_mlp_scorer.{name}" for name in names),
                **_component_metrics(g_score, g_gross, lambda_portfolio),
            }
        )
    return rows


def _batch_info(batch: dict[str, Any], batch_index: int, group: str) -> dict[str, Any]:
    dates = [_date_string(value) for value in batch["dates"]]
    return {
        "batch_group": group,
        "batch_index": int(batch_index),
        "date_start": min(dates),
        "date_end": max(dates),
        "dates": dates,
        "date_count": len(dates),
    }


def _summary_rows(batch_frame: pd.DataFrame) -> pd.DataFrame:
    metric_names = [
        "score_grad_norm",
        "gross_grad_norm_weighted",
        "gradient_ratio",
        "gradient_cosine",
        "cancellation_ratio",
    ]
    rows: list[dict[str, Any]] = []
    for (checkpoint, step, group), frame in batch_frame.groupby(["checkpoint", "optimizer_step", "batch_group"], sort=False):
        row: dict[str, Any] = {
            "checkpoint": checkpoint,
            "optimizer_step": int(step),
            "batch_group": group,
            "batch_count": int(len(frame)),
        }
        for metric in metric_names:
            values = frame[metric].to_numpy(dtype=np.float64)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_p10"] = float(np.quantile(values, 0.10))
            row[f"{metric}_p90"] = float(np.quantile(values, 0.90))
            if metric == "gradient_cosine":
                row[f"{metric}_positive_ratio"] = float(np.mean(values > 0))
                row[f"{metric}_negative_ratio"] = float(np.mean(values < 0))
        rows.append(row)
    return pd.DataFrame(rows)


def _state_parameter_vector(payload: dict[str, Any], prefix: str = "simple_mlp_scorer.") -> np.ndarray:
    state = payload["model_state_dict"]
    names = sorted(name for name in state if name.startswith(prefix))
    if not names:
        raise RuntimeError("No MLP scorer parameters found in checkpoint")
    return np.concatenate([state[name].detach().cpu().numpy().astype(np.float64).reshape(-1) for name in names])


def _parameter_distances(checkpoints: list[dict[str, Any]]) -> pd.DataFrame:
    baseline = checkpoints[0]
    base_vector = _state_parameter_vector(baseline["payload"])
    base_norm = float(np.linalg.norm(base_vector))
    rows: list[dict[str, Any]] = []
    for item in checkpoints:
        vector = _state_parameter_vector(item["payload"])
        distance = float(np.linalg.norm(vector - base_vector))
        rows.append(
            {
                "checkpoint": item["checkpoint"],
                "optimizer_step": item["optimizer_step"],
                "checkpoint_sha256": item["checkpoint_sha256"],
                "baseline_checkpoint": baseline["checkpoint"],
                "parameter_l2_distance": distance,
                "relative_parameter_l2_distance": distance / max(base_norm, 1e-12),
            }
        )
    return pd.DataFrame(rows)


def _load_checkpoints(checkpoint_dir: Path, device: torch.device) -> list[dict[str, Any]]:
    paths = sorted(checkpoint_dir.glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    preferred = {"initial_from_a.pt": 0, "best.pt": 1, "last.pt": 2}
    paths.sort(key=lambda path: (preferred.get(path.name, 99), path.name))
    items: list[dict[str, Any]] = []
    for path in paths:
        payload = torch.load(path, map_location=device, weights_only=False)
        # initial_from_a is byte-identical to A's checkpoint, whose payload may
        # retain A's own training counter.  In this diagnostic it is explicitly
        # the C step-0 state, so report optimizer_step=0 without rewriting it.
        step = 0 if path.stem == "initial_from_a" else int(payload.get("global_step", -1))
        if step < 0:
            raise RuntimeError(f"Checkpoint has no optimizer_step/global_step: {path}")
        items.append(
            {
                "checkpoint": path.stem,
                "path": path,
                "optimizer_step": step,
                "checkpoint_sha256": _sha256(path),
                "payload": payload,
            }
        )
    names = {item["checkpoint"] for item in items}
    missing = {"initial_from_a", "best", "last"} - names
    if missing:
        raise FileNotFoundError(f"Required checkpoints missing: {sorted(missing)}")
    return items


def _write_report(
    path: Path,
    *,
    calibration: dict[str, Any],
    summaries: pd.DataFrame,
    distances: pd.DataFrame,
    layer_frame: pd.DataFrame,
    lambda_portfolio: float,
) -> None:
    def median(group: str, checkpoint: str, metric: str) -> float:
        row = summaries[(summaries["batch_group"] == group) & (summaries["checkpoint"] == checkpoint)]
        if row.empty:
            return float("nan")
        return float(row.iloc[0][f"{metric}_median"])

    def layer_value(checkpoint: str, layer: str, metric: str) -> float:
        frame = layer_frame[(layer_frame["checkpoint"] == checkpoint) & (layer_frame["layer"] == layer)]
        frame = frame[frame["batch_group"] == "calibration_batches"]
        return float(frame[metric].median()) if not frame.empty else float("nan")

    step0 = "initial_from_a"
    best = "best"
    last = "last"
    ratio_lines = []
    for group in ["calibration_batches", "holdout_train_batches"]:
        ratio_lines.append(
            f"| {group} | {median(group, step0, 'gradient_ratio'):.6g} | {median(group, best, 'gradient_ratio'):.6g} | {median(group, last, 'gradient_ratio'):.6g} |"
        )
    cosine_lines = []
    for group in ["calibration_batches", "holdout_train_batches"]:
        cosine_lines.append(
            f"| {group} | {median(group, step0, 'gradient_cosine'):.6g} | {median(group, best, 'gradient_cosine'):.6g} | {median(group, last, 'gradient_cosine'):.6g} |"
        )
    step20_cancel = median("calibration_batches", best, "cancellation_ratio")
    best_cos = median("calibration_batches", best, "gradient_cosine")
    best_ratio = median("calibration_batches", best, "gradient_ratio")
    first_ratio = layer_value(best, "first_linear", "gradient_ratio")
    output_ratio = layer_value(best, "output_linear", "gradient_ratio")
    first_cos = layer_value(best, "first_linear", "gradient_cosine")
    output_cos = layer_value(best, "output_linear", "gradient_cosine")
    best_distance = float(distances.loc[distances["checkpoint"] == best, "relative_parameter_l2_distance"].iloc[0])
    last_distance = float(distances.loc[distances["checkpoint"] == last, "relative_parameter_l2_distance"].iloc[0])
    direction = "增强" if median("calibration_batches", last, "gradient_ratio") > median("calibration_batches", step0, "gradient_ratio") else "衰减"
    conflict = "冲突" if best_cos < -0.2 else ("协同" if best_cos > 0.2 else "基本正交")
    allocator_support = "支持 allocator 平滑限制更重要" if abs(best_ratio - 1.0) <= 3.0 and best_cos > -0.2 else "需要谨慎区分 loss 权重与 allocator 影响"
    distance_table = [
        "| checkpoint | optimizer_step | checkpoint_sha256 | parameter_l2_distance | relative_parameter_l2_distance |",
        "|---|---:|---|---:|---:|",
    ]
    for _, row in distances.iterrows():
        distance_table.append(
            f"| {row['checkpoint']} | {int(row['optimizer_step'])} | {row['checkpoint_sha256']} | {row['parameter_l2_distance']:.8g} | {row['relative_parameter_l2_distance']:.8g} |"
        )
    lines = [
        "# Experiment C Gradient Contribution Diagnostic",
        "",
        "本报告只使用 2018–2020 training split；没有读取或评价 2021 validation / 2022 数据，没有重新训练 C。",
        "",
        "## 结论摘要",
        "",
        f"- Step 0 原始 calibration 复现：**{calibration['status']}**；重算 lambda={calibration['recomputed_lambda_portfolio']:.10f}，历史 lambda={lambda_portfolio:.10f}，差异={calibration['absolute_difference']:.3g}。",
        f"- Best step 20 的 calibration-batch median gradient ratio={best_ratio:.6g}，median cosine={best_cos:.6g}，cancellation ratio={step20_cancel:.6g}。",
        f"- 在 calibration batches 上，portfolio gradient 相对 step 0 到 step 220 整体{direction}；这不是 turnover 证据，而是直接梯度证据。",
        f"- Best step 20 两个目标总体{conflict}；当前证据{allocator_support}。",
        f"- Step 20 相对 step 0 参数距离={best_distance:.6g}，step 220 相对 step 0 参数距离={last_distance:.6g}。",
        "",
        "## Gradient ratio",
        "",
        "定义：weighted portfolio gradient / score gradient，weighted portfolio gradient=70.2102429073×raw gross gradient。",
        "",
        "| batch group | step 0 | best step 20 | last step 220 |",
        "|---|---:|---:|---:|",
        *ratio_lines,
        "",
        "## Gradient cosine",
        "",
        "| batch group | step 0 | best step 20 | last step 220 |",
        "|---|---:|---:|---:|",
        *cosine_lines,
        "",
        "## Layer contribution at best step 20",
        "",
        f"- first layer (`hidden.weight` + `hidden.bias`): ratio={first_ratio:.6g}, cosine={first_cos:.6g}。",
        f"- output layer (`output.weight` + `output.bias`): ratio={output_ratio:.6g}, cosine={output_cos:.6g}。",
        "",
        "## Required interpretation",
        "",
        f"1. Step 0 calibration：{calibration['status']}。",
        f"2. Step 20 weighted portfolio/score ratio：{best_ratio:.6g}（calibration batches）。",
        f"3. Last-step ratio：{median('calibration_batches', last, 'gradient_ratio'):.6g}（calibration batches）。",
        f"4. Best-step cosine：{best_cos:.6g}；两个目标判断为：{conflict}。",
        f"5. Portfolio gradient 在训练过程中：{direction}（按 ratio 的 step0→step220 比较）。",
        "6. C 的 3.9078% 换手率和较高 effective N 不能单独证明 lambda 过小；本轮没有用 turnover 判断 lambda。",
        "7. 是否进入 D：**不建议立即进入**；先把本诊断中的 gradient balance、layer contribution 和 allocator 集中度约束作为后续设计依据。",
        "8. 本轮没有自动修改 lambda、allocator、temperature，也没有运行 D。",
        "",
        "## Checkpoint distance",
        "",
        *distance_table,
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    config_path: Path,
    *,
    c_output: Path,
    output: Path | None = None,
) -> Path:
    c_output = c_output.resolve()
    output = (output or c_output / "gradient_diagnostic").resolve()
    checkpoints_dir = c_output / "checkpoints"
    if not c_output.is_dir():
        raise FileNotFoundError(c_output)
    if not checkpoints_dir.is_dir():
        raise FileNotFoundError(checkpoints_dir)
    calibration_file = c_output / "gradient_scale_calibration.json"
    resolved_file = c_output / "resolved_config.yaml"
    if not calibration_file.is_file() or not resolved_file.is_file():
        raise FileNotFoundError("Experiment C resolved config/calibration output is required")
    c_calibration = json.loads(calibration_file.read_text(encoding="utf-8"))
    resolved = yaml.safe_load(resolved_file.read_text(encoding="utf-8"))
    lambda_portfolio = float(resolved["loss"]["lambda_portfolio"])
    if not np.isfinite(lambda_portfolio) or lambda_portfolio <= 0:
        raise ValueError("Invalid Experiment C lambda_portfolio")
    config = load_config(config_path)
    if config.model.horizons != [2] or config.model.execution_horizon != 2:
        raise ValueError("Gradient diagnostic requires h=2 daily-core execution")
    if config.data.test_start is not None or config.data.test_end is not None:
        raise ValueError("Gradient diagnostic config must not contain an OOS/test range")
    output.mkdir(parents=True, exist_ok=True)
    for child in output.iterdir():
        if child.is_dir():
            import shutil

            shutil.rmtree(child)
        else:
            child.unlink()

    checkpoint_items = _load_checkpoints(checkpoints_dir, torch.device("cpu"))
    hashes_before = {item["checkpoint"]: item["checkpoint_sha256"] for item in checkpoint_items}
    seed_everything(config.seed)
    source, synthetic = load_configured_data(config)
    loaders = build_dataloaders(config, source)
    all_train_batches = [dict(batch) for batch in loaders.train]
    if len(all_train_batches) < 32:
        raise RuntimeError(f"Need at least 32 train batches, found {len(all_train_batches)}")
    train_dates = [_date_string(date) for batch in all_train_batches for date in batch["dates"]]
    if any(date >= "2021-01-01" for date in train_dates):
        raise RuntimeError("Gradient diagnostic train loader contains validation/test dates")
    group_a_indices = list(range(16))
    group_b_indices = list(range(len(all_train_batches) - 16, len(all_train_batches)))
    if set(group_a_indices) & set(group_b_indices):
        raise RuntimeError("Diagnostic batch groups overlap")
    batch_groups: dict[str, list[int]] = {
        "calibration_batches": group_a_indices,
        "holdout_train_batches": group_b_indices,
    }
    batch_manifest: dict[str, Any] = {
        "data_split": "train only",
        "train_date_range": ["2018-01-02", "2020-12-29"],
        "train_batch_count": len(all_train_batches),
        "groups": {},
        "used_2021": False,
        "used_2022": False,
        "synthetic": synthetic,
    }
    for group, indices in batch_groups.items():
        batch_manifest["groups"][group] = [_batch_info(all_train_batches[index], index, group) for index in indices]
    (output / "gradient_diagnostic_batches.json").write_text(json.dumps(batch_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    device = resolve_device(config.device)
    model = E2EAIModel(config.model, config.ablation).to(device)
    layout = _parameter_layout(model)
    batch_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    checkpoint_order: list[dict[str, Any]] = []
    for item in checkpoint_items:
        payload = torch.load(item["path"], map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        item_for_distance = {**item, "payload": payload}
        checkpoint_order.append(item_for_distance)
        for group, indices in batch_groups.items():
            for batch_index in indices:
                batch = _move_batch(dict(all_train_batches[batch_index]), device)
                info = _batch_info(all_train_batches[batch_index], batch_index, group)
                score_loss, g_score, score_by_name = _loss_gradient(model, batch, layout, "score")
                gross_loss, g_gross_raw, gross_by_name = _loss_gradient(model, batch, layout, "gross")
                metrics = _component_metrics(g_score, g_gross_raw, lambda_portfolio)
                row = {
                    "checkpoint": item["checkpoint"],
                    "optimizer_step": item["optimizer_step"],
                    "checkpoint_sha256": item["checkpoint_sha256"],
                    **info,
                    "score_loss": score_loss,
                    "gross_loss": gross_loss,
                    "score_loss_value": score_loss,
                    "gross_loss_value": gross_loss,
                    "weighted_gross_loss_value": lambda_portfolio * gross_loss,
                    **metrics,
                }
                batch_rows.append(row)
                layer_rows.extend(
                    {
                        "checkpoint": item["checkpoint"],
                        "optimizer_step": item["optimizer_step"],
                        "checkpoint_sha256": item["checkpoint_sha256"],
                        **info,
                        **layer,
                    }
                    for layer in _layer_metrics(score_by_name, gross_by_name, lambda_portfolio)
                )

    batch_frame = pd.DataFrame(batch_rows)
    layer_frame = pd.DataFrame(layer_rows)
    summaries = _summary_rows(batch_frame)
    distances = _parameter_distances(checkpoint_order)
    batch_frame.to_csv(output / "gradient_contribution_by_batch.csv", index=False, encoding="utf-8-sig")
    layer_frame.to_csv(output / "gradient_contribution_by_layer.csv", index=False, encoding="utf-8-sig")
    summaries.to_csv(output / "gradient_contribution_summary.csv", index=False, encoding="utf-8-sig")
    distances.to_csv(output / "checkpoint_parameter_distance.csv", index=False, encoding="utf-8-sig")

    step0_rows = batch_frame[(batch_frame["checkpoint"] == "initial_from_a") & (batch_frame["batch_group"] == "calibration_batches")]
    c_batch_indices = {int(row["batch_index"]) for row in c_calibration.get("batch_gradients", [])}
    actual_batch_indices = set(step0_rows["batch_index"].astype(int))
    median_score = float(step0_rows["score_grad_norm"].median())
    median_gross = float(step0_rows["gross_grad_norm_raw"].median())
    lambda_recomputed = median_score / median_gross
    absolute_difference = abs(lambda_recomputed - lambda_portfolio)
    relative_difference = absolute_difference / max(abs(lambda_portfolio), 1e-12)
    calibration_status = "PASS" if c_batch_indices == actual_batch_indices and relative_difference <= 1e-4 else "CALIBRATION_REPRODUCTION_MISMATCH"
    calibration_result = {
        "status": calibration_status,
        "historical_lambda_portfolio": lambda_portfolio,
        "recomputed_lambda_portfolio": lambda_recomputed,
        "median_score_grad": median_score,
        "median_raw_gross_grad": median_gross,
        "absolute_difference": absolute_difference,
        "relative_difference": relative_difference,
        "expected_batch_indices": sorted(c_batch_indices),
        "actual_batch_indices": sorted(actual_batch_indices),
        "tolerance_relative": 1e-4,
        "calibration_source": str(calibration_file),
        "historical_calibration_lambda": float(c_calibration.get("lambda_portfolio", float("nan"))),
        "uses_train_only": True,
        "used_2021": False,
        "used_2022": False,
    }
    write_metrics(output / "calibration_reproduction.json", calibration_result)
    _write_report(output / "gradient_contribution_report.md", calibration=calibration_result, summaries=summaries, distances=distances, layer_frame=layer_frame, lambda_portfolio=lambda_portfolio)

    hashes_after = {path.stem: _sha256(path) for path in checkpoints_dir.glob("*.pt")}
    checkpoint_hashes_unchanged = hashes_before == hashes_after
    if not checkpoint_hashes_unchanged:
        raise RuntimeError(f"Checkpoint hash changed during read-only diagnostic: before={hashes_before}, after={hashes_after}")
    manifest = {
        "experiment": "experiment_c_gradient_contribution_diagnostic",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_experiment_c": str(c_output),
        "output_dir": str(output),
        "checkpoint_files": [
            {
                "checkpoint": item["checkpoint"],
                "optimizer_step": item["optimizer_step"],
                "checkpoint_sha256": item["checkpoint_sha256"],
            }
            for item in checkpoint_items
        ],
        "checkpoint_hashes_unchanged": checkpoint_hashes_unchanged,
        "optimizer_created": False,
        "optimizer_step_called": False,
        "training_performed": False,
        "lambda_portfolio_unchanged": True,
        "allocator_unchanged": True,
        "temperature_unchanged": True,
        "used_train_only": True,
        "used_2021": False,
        "used_2022": False,
        "experiment_d_run": False,
        "files": sorted(path.name for path in output.iterdir()),
    }
    write_metrics(output / "run_manifest.json", manifest)
    print(f"[gradient-diagnostic] status={calibration_status} output={output}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--c-output", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(run(args.config, c_output=args.c_output, output=args.output))


if __name__ == "__main__":
    main()
