"""Top-level end-to-end E2EAI model without label inputs."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import Tensor, nn

from e2eai.config import AblationConfig, ModelConfig
from e2eai.models.baselines import SimpleMLPScorer, factor_mean_scores
from e2eai.models.context_encoder import StockContextEncoder
from e2eai.models.deep_factor import MultiHorizonDeepFactor
from e2eai.models.directional_attention import DirectionalFactorAttention
from e2eai.models.directional_buffer import DirectionalBuffer
from e2eai.models.factor_selection import FactorSelector
from e2eai.models.neutralization import CascadedRelationalNeutralization
from e2eai.models.portfolio import CapMode, GatedPortfolioAllocator


@dataclass(frozen=True)
class E2EAIOutput:
    """All prediction, loss, and interpretation tensors from one forward pass."""

    factor_attention: Tensor
    factor_selection_mask: Tensor
    selected_factors: Tensor
    stock_context: Tensor
    industry_neutral: Tensor
    universe_neutral: Tensor
    industry_gat_component: Tensor
    universe_gat_component: Tensor
    deep_factors: Tensor
    factor_attentions: Tensor
    factor_attention_mean: Tensor
    signed_factor_coefficients: Tensor
    deep_factor_approx: Tensor
    portfolio_attention: Tensor
    stock_selection_mask: Tensor
    portfolio_weights: Tensor
    factor_directions: Tensor
    deep_directions: Tensor

    def tensor_dict(self) -> dict[str, Tensor]:
        """Return every named tensor for finite-value audits."""
        return {item.name: getattr(self, item.name) for item in fields(self)}

    def detached_cpu(self) -> "E2EAIOutput":
        """Return a fully detached CPU copy suitable for export/visualization."""
        return E2EAIOutput(
            **{
                name: value.detach().cpu()
                for name, value in self.tensor_dict().items()
            }
        )


class E2EAIModel(nn.Module):
    """Joint factor-selection, graph-factor, interpretation, and allocation model.

    The forward signature deliberately excludes future-return labels.  Direction
    snapshots can be supplied by :class:`~e2eai.training.trainer.Trainer`; if
    omitted, the model reads its persistent pre-existing buffers without mutating
    them.  Buffer updates are an explicit post-decision training action.
    """

    def __init__(
        self,
        config: ModelConfig,
        ablation: AblationConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.ablation = ablation or AblationConfig()
        num_factors = config.num_factors
        hidden_dim = config.context.hidden_dim
        num_horizons = len(config.horizons)
        factor = config.factor_selection
        gat = config.gat
        direction = config.direction
        portfolio = config.portfolio

        self.factor_selector = FactorSelector(
            num_factors=num_factors,
            hidden_dim=factor.hidden_dim,
            gamma_f=factor.gamma_f,
            pooling=factor.pooling,
            gate_mode_train=factor.gate_mode_train,
            gate_mode_eval=factor.gate_mode_eval,
            min_selected_factors=factor.min_selected_factors,
        )
        self.context_encoder = StockContextEncoder(
            num_factors=num_factors,
            hidden_dim=hidden_dim,
            normalization=config.context.normalization,
            dropout=config.context.dropout,
            eps=config.context.eps,
            leaky_relu_slope=gat.leaky_relu_slope,
        )
        self.simple_mlp_scorer = (
            SimpleMLPScorer(num_factors, hidden_dim)
            if config.strategy_head == "simple_mlp"
            else None
        )
        self.neutralization = CascadedRelationalNeutralization(
            hidden_dim=hidden_dim,
            num_heads=gat.heads,
            dropout=gat.dropout,
            leaky_relu_slope=gat.leaky_relu_slope,
            add_self_loops=gat.add_self_loops,
        )
        self.deep_factor = MultiHorizonDeepFactor(
            hidden_dim=hidden_dim,
            horizons=config.horizons,
            share_head=config.share_deep_factor_head,
            leaky_relu_slope=gat.leaky_relu_slope,
        )
        self.directional_buffer = DirectionalBuffer(
            num_horizons=num_horizons,
            num_factors=num_factors,
            normalization=direction.normalization,
            min_stocks=direction.min_stocks,
            eps=direction.eps,
            zero_direction=direction.zero_direction,
        )
        self.directional_attention = DirectionalFactorAttention(
            num_factors=num_factors,
            num_horizons=num_horizons,
            negative_slope=gat.leaky_relu_slope,
            zscore_approximation_input=direction.zscore_approximation_input,
            eps=direction.eps,
        )
        self.portfolio_allocator = GatedPortfolioAllocator(
            allocation_mode=portfolio.allocation_mode,
            hidden_dim=hidden_dim,
            num_horizons=num_horizons,
            gamma_p=portfolio.gamma_p,
            gamma_p_mode=portfolio.gamma_p_mode,
            theta=portfolio.theta,
            min_selected_stocks=portfolio.min_selected_stocks,
            cap_mode=portfolio.cap_mode_train,
            projection_iterations=portfolio.projection_iterations,
            leaky_relu_slope=gat.leaky_relu_slope,
            topk_ratio=portfolio.topk_ratio,
            topk=portfolio.topk,
        )

    @property
    def horizons(self) -> tuple[int, ...]:
        """Ordered model horizons."""
        return tuple(self.config.horizons)

    def snapshot_directions(self) -> tuple[Tensor, Tensor]:
        """Clone current non-trainable directions for a leakage-safe decision."""
        if not self.ablation.use_direction_buffer:
            device = self.directional_buffer.deep_factor_ic_sum.device
            return (
                torch.ones(
                    len(self.config.horizons),
                    self.config.num_factors,
                    device=device,
                ),
                torch.ones(len(self.config.horizons), device=device),
            )
        factor_direction = self.directional_buffer.factor_directions().detach().clone()
        deep_direction = self.directional_buffer.deep_directions().detach().clone()
        return factor_direction, deep_direction

    def forward(
        self,
        raw_factors: Tensor,
        asset_mask: Tensor,
        industry_adjacency: Tensor,
        universe_adjacency: Tensor,
        *,
        factor_directions: Tensor | None = None,
        deep_directions: Tensor | None = None,
        cap_mode: CapMode | None = None,
    ) -> E2EAIOutput:
        """Produce portfolios and intermediate tensors from features and graphs.

        Args:
            raw_factors: Point-in-time factors ``[B,N,M]``.
            asset_mask: Valid-stock indicator ``[B,N]``.
            industry_adjacency: Industry graph ``[B,N,N]``.
            universe_adjacency: Universe graph ``[B,N,N]``.
            factor_directions: Optional pre-update direction snapshot ``[K,M]``.
            deep_directions: Optional pre-update direction snapshot ``[K]``.
            cap_mode: Per-call portfolio allocation mode.

        Future returns are intentionally not accepted by this method.
        """
        self._validate_inputs(
            raw_factors,
            asset_mask,
            industry_adjacency,
            universe_adjacency,
        )
        if factor_directions is None or deep_directions is None:
            buffered_factor, buffered_deep = self.snapshot_directions()
            factor_directions = buffered_factor if factor_directions is None else factor_directions
            deep_directions = buffered_deep if deep_directions is None else deep_directions
        factor_directions = factor_directions.to(
            device=raw_factors.device, dtype=raw_factors.dtype
        )
        deep_directions = deep_directions.to(
            device=raw_factors.device, dtype=raw_factors.dtype
        )

        # E2EAI Eqs. (1)-(2): fixed-width dynamic factor gate.
        if self.ablation.use_factor_selector:
            selection = self.factor_selector(raw_factors, asset_mask)
            factor_attention = selection.factor_attention
            factor_selection_mask = selection.factor_selection_mask
            selected_factors = selection.selected_factors
        else:
            selected_factors = torch.where(
                asset_mask.bool().unsqueeze(-1) & torch.isfinite(raw_factors),
                raw_factors,
                torch.zeros_like(raw_factors),
            )
            factor_attention = torch.full(
                (raw_factors.shape[0], raw_factors.shape[-1]),
                1.0 / raw_factors.shape[-1],
                device=raw_factors.device,
                dtype=raw_factors.dtype,
            )
            factor_selection_mask = torch.ones_like(factor_attention, dtype=torch.bool)

        if self.simple_mlp_scorer is not None:
            stock_context = self.simple_mlp_scorer.encode(selected_factors, asset_mask)
        else:
            stock_context = self.context_encoder(selected_factors, asset_mask)

        industry_neutral = stock_context
        industry_gat_component = torch.zeros_like(stock_context)
        if self.ablation.use_industry_gat:
            industry_output = self.neutralization.industry_block.forward_with_diagnostics(
                stock_context, industry_adjacency, asset_mask
            )
            industry_neutral = industry_output.neutral
            industry_gat_component = industry_output.gat_component
        universe_neutral = industry_neutral
        universe_gat_component = torch.zeros_like(industry_neutral)
        if self.ablation.use_universe_gat:
            universe_output = self.neutralization.universe_block.forward_with_diagnostics(
                industry_neutral, universe_adjacency, asset_mask
            )
            universe_neutral = universe_output.neutral
            universe_gat_component = universe_output.gat_component
        # E2EAI Eq. (6): independent horizon-specific deep-factor heads.
        portfolio_scores: Tensor | None = None
        if self.simple_mlp_scorer is not None:
            portfolio_scores = self.simple_mlp_scorer.score(universe_neutral, asset_mask)
            deep_factors = portfolio_scores[:, None, :].expand(
                -1, len(self.config.horizons), -1
            )
        elif self.config.strategy_head == "factor_mean":
            portfolio_scores = factor_mean_scores(selected_factors, asset_mask)
            deep_factors = portfolio_scores[:, None, :].expand(
                -1, len(self.config.horizons), -1
            )
        else:
            deep_factors = self.deep_factor(
                stock_context,
                industry_neutral,
                universe_neutral,
                asset_mask,
            )
        # E2EAI Eqs. (7)-(8): dimensionality-correct signed approximation.
        if self.ablation.use_directional_approximation:
            interpretation = self.directional_attention(
                selected_factors,
                asset_mask,
                factor_directions,
            )
            factor_attentions = interpretation.factor_attentions
            factor_attention_mean = interpretation.factor_attention_mean
            signed_factor_coefficients = interpretation.signed_factor_coefficients
            deep_factor_approx = interpretation.deep_factor_approx
        else:
            batch, stocks, factors = selected_factors.shape
            horizons = len(self.config.horizons)
            factor_attentions = selected_factors.new_zeros(batch, horizons, stocks, factors)
            factor_attention_mean = selected_factors.new_zeros(batch, horizons, factors)
            signed_factor_coefficients = selected_factors.new_zeros(batch, horizons, factors)
            deep_factor_approx = torch.zeros_like(deep_factors)
        # E2EAI Eqs. (9)-(10): true masked stock gate and soft allocation.
        portfolio = self.portfolio_allocator(
            stock_context,
            deep_factors,
            deep_directions,
            asset_mask,
            cap_mode=cap_mode,
            portfolio_scores=portfolio_scores,
        )
        return E2EAIOutput(
            factor_attention=factor_attention,
            factor_selection_mask=factor_selection_mask,
            selected_factors=selected_factors,
            stock_context=stock_context,
            industry_neutral=industry_neutral,
            universe_neutral=universe_neutral,
            industry_gat_component=industry_gat_component,
            universe_gat_component=universe_gat_component,
            deep_factors=deep_factors,
            factor_attentions=factor_attentions,
            factor_attention_mean=factor_attention_mean,
            signed_factor_coefficients=signed_factor_coefficients,
            deep_factor_approx=deep_factor_approx,
            portfolio_attention=portfolio.portfolio_attention,
            stock_selection_mask=portfolio.stock_mask,
            portfolio_weights=portfolio.portfolio_weights,
            factor_directions=factor_directions,
            deep_directions=deep_directions,
        )

    @torch.no_grad()
    def update_direction_buffers(
        self,
        output: E2EAIOutput,
        forward_returns: Tensor,
        asset_mask: Tensor,
        return_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Update cumulative state after a completed current-batch decision."""
        if not self.ablation.use_direction_buffer:
            factor_direction, deep_direction = self.snapshot_directions()
            return {
                "factor_direction": factor_direction,
                "deep_direction": deep_direction,
            }
        return self.directional_buffer.update(
            output.selected_factors,
            output.deep_factors,
            forward_returns,
            asset_mask,
            return_mask,
        )

    def _validate_inputs(
        self,
        raw_factors: Tensor,
        asset_mask: Tensor,
        industry_adjacency: Tensor,
        universe_adjacency: Tensor,
    ) -> None:
        if raw_factors.ndim != 3:
            raise ValueError("raw_factors must have shape [B,N,M]")
        batch, stocks, factors = raw_factors.shape
        if factors != self.config.num_factors:
            raise ValueError(
                f"Expected M={self.config.num_factors}, received M={factors}"
            )
        if asset_mask.shape != (batch, stocks):
            raise ValueError("asset_mask must have shape [B,N]")
        expected_graph = (batch, stocks, stocks)
        if industry_adjacency.shape != expected_graph:
            raise ValueError("industry_adjacency must have shape [B,N,N]")
        if universe_adjacency.shape != expected_graph:
            raise ValueError("universe_adjacency must have shape [B,N,N]")
        if (asset_mask.sum(dim=-1) == 0).any():
            raise ValueError("Every date must contain at least one valid stock")
        devices = {
            raw_factors.device,
            asset_mask.device,
            industry_adjacency.device,
            universe_adjacency.device,
        }
        if len(devices) != 1:
            raise ValueError("All model inputs must be on the same device")
