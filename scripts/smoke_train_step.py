"""Run one real-data E2EAI optimization step and persist finite-value diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from e2eai.config import load_config
from e2eai.data.loaders import build_dataloaders
from e2eai.models.e2eai import E2EAIModel
from e2eai.training.trainer import E2EAITrainer
from e2eai.utils.reproducibility import seed_everything
from e2eai.workflows import load_configured_data, write_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="output/interface_smoke")
    parser.add_argument(
        "--disable-mixed-precision",
        action="store_true",
        help="Run the diagnostic step entirely in float32.",
    )
    parser.add_argument(
        "--disable-differentiable-inner-loop",
        action="store_true",
        help="Detach the local cross-sectional regression from the global graph.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config.device = args.device
    config.logging.output_dir = args.output_dir
    config.logging.tensorboard = False
    config.logging.wandb = False
    if args.disable_mixed_precision:
        config.training.mixed_precision = False
    if args.disable_differentiable_inner_loop:
        config.local_optimizer.differentiable = False
    config.validate()
    seed_everything(config.seed)
    source, synthetic = load_configured_data(config)
    if synthetic:
        raise RuntimeError("This smoke test requires configured real data")
    loaders = build_dataloaders(config, source)
    batch = next(iter(loaders.train))
    trainer = E2EAITrainer(E2EAIModel(config.model, config.ablation), config)
    try:
        diagnostics = trainer.train_batch(batch)
    except RuntimeError:
        bad_gradients = [
            name
            for name, parameter in trainer.model.named_parameters()
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        ]
        print({"nonfinite_gradient_parameters": bad_gradients}, flush=True)
        raise
    output = trainer.last_output
    if output is None:
        raise RuntimeError("Training step did not retain a model output")
    finite = all(
        torch.isfinite(value).all().item()
        for value in output.__dict__.values()
        if isinstance(value, torch.Tensor) and value.is_floating_point()
    )
    result = {
        "data_shape": list(batch["raw_factors"].shape),
        "return_shape": list(batch["forward_returns"].shape),
        "dates": [str(value.date()) for value in batch["dates"]],
        "diagnostics": diagnostics,
        "weight_sum_min": float(output.portfolio_weights.sum(dim=-1).min().detach().cpu()),
        "weight_sum_max": float(output.portfolio_weights.sum(dim=-1).max().detach().cpu()),
        "max_weight": float(output.portfolio_weights.max().detach().cpu()),
        "all_outputs_finite": bool(finite),
    }
    if not finite or not all(torch.isfinite(torch.tensor(value)) for value in diagnostics.values()):
        raise FloatingPointError("Non-finite value found in real-data smoke step")
    target = write_metrics(Path(args.output_dir) / "smoke_train_step.json", result)
    trainer.close()
    print(f"smoke_metrics={target}")
    print(result)


if __name__ == "__main__":
    main()
