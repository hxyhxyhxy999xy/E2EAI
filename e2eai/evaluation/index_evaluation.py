"""Full OOS exports and separate horizon target-weight turnover for index runs."""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from e2eai.evaluation.metrics import compute_performance_metrics, turnover_from_asset_weights
from e2eai.workflows import move_batch_to_device, write_metrics


def summarize_dates(frame, horizons):
    result = {}
    for h in horizons:
        f = frame[frame.horizon == h]
        good = np.isfinite(f.portfolio_return) & np.isfinite(f.benchmark_return)
        # h-day labels overlap. Annualization is 252/h, and drawdown is on
        # an explicitly labelled every-hth-decision subseries, not daily NAV.
        metrics = compute_performance_metrics(f.loc[good, 'portfolio_return'].to_numpy(),
            benchmark_returns=f.loc[good, 'benchmark_return'].to_numpy(), annualization=252/h)
        from e2eai.evaluation.metrics import maximum_drawdown
        metrics['max_drawdown'] = maximum_drawdown(f.portfolio_return.iloc[::h].to_numpy())
        metrics['annualized_excess_return'] = metrics.pop('alpha')
        average_selected_stocks = float(f.holdings.mean())
        average_effective_holdings = float(f.effective_holdings.mean())
        metrics.update({
            # ``average_holdings`` is retained as a compatibility alias, but
            # now means effective holdings. The literal threshold-selected
            # count is exported separately as ``average_selected_stocks``.
            'average_holdings': average_effective_holdings,
            'average_selected_stocks': average_selected_stocks,
            'average_effective_holdings': average_effective_holdings,
            'min_selected_stocks':int(f.holdings.min()),
            'max_selected_stocks':int(f.holdings.max()),
            'average_max_weight':float(f.max_weight.mean()),
            'maximum_single_weight':float(f.max_weight.max()),
            'concentration_hhi':float(f.concentration.mean()),
            'effective_holdings':float(f.effective_holdings.mean()),
            'one_way_turnover':float(f.one_way_turnover.mean()),
            'gross_turnover':float(f.gross_turnover.mean()),
            'ic':float(f.ic.mean()),
            'icir':float(f.ic.mean()/f.ic.std(ddof=0)) if f.ic.std(ddof=0)>1e-12 else 0.0,
            'factor_return':float(f.factor_return.mean()),
            'valid_samples':int(good.sum()), 'valid_ic_samples':int(f.ic.notna().sum()),
            'average_candidates':float(f.candidate_count.mean()),
            'turnover_adjustment_days':float(f.loc[f.index_adjustment_day == True,'one_way_turnover'].mean()),
            'turnover_non_adjustment_days':float(f.loc[f.index_adjustment_day == False,'one_way_turnover'].mean()),
            'gross_turnover_adjustment_days':float(f.loc[f.index_adjustment_day == True,'gross_turnover'].mean()),
            'gross_turnover_non_adjustment_days':float(f.loc[f.index_adjustment_day == False,'gross_turnover'].mean()),
        })
        result[str(h)] = metrics
    return result


def evaluate_index(trainer, loaders, destination, max_batches=None):
    destination = Path(destination)
    horizons = trainer.config.model.horizons
    dataset = loaders.dataset
    rows = []
    previous = {h:None for h in horizons}
    writers = {}
    trainer.model.eval()
    try:
        for batch_index, raw in enumerate(loaders.test):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = move_batch_to_device(raw, trainer.device)
            fd, dd = trainer._direction_snapshot()
            with torch.no_grad():
                out = trainer._forward_model(batch, fd, dd, cap_mode=trainer.config.model.portfolio.cap_mode_eval)
                losses = trainer.loss_fn.from_output(out,batch['forward_returns'],dd,batch['asset_mask'],
                    batch['return_mask'],differentiable_inner_loop=False)
            o = out.detached_cpu()
            predictions = []
            interpretations = []
            for b,date in enumerate(raw['dates']):
                ids = raw['asset_ids'][b]
                n = len(ids)
                idx = dataset.dates.index(pd.Timestamp(date))
                _, diag = dataset.universe_diagnostics(idx)
                for k,h in enumerate(horizons):
                    w = o.portfolio_weights[b,k,:n].numpy()
                    current = dict(zip(ids,w.tolist()))
                    tt = ({'one_way_turnover':np.nan,'gross_turnover':np.nan} if previous[h] is None
                          else turnover_from_asset_weights([previous[h],current]))
                    previous[h] = current
                    observed_weight = float((batch['return_mask'][b,k]*out.portfolio_weights[b,k]).sum())
                    ic = float(losses['ic_by_date_horizon'][b,k]) if bool(losses['valid_ic_by_date_horizon'][b,k]) else np.nan
                    rows.append({'date':date,'horizon':h,'index':dataset.membership.index_name,
                        'portfolio_return':float(losses['portfolio_return_by_date_horizon'][b,k]) if observed_weight>0 else np.nan,
                        'benchmark_return':float(raw['benchmark_returns'][b,k]),
                        'observed_weight':observed_weight,
                        'holdings':int(o.stock_selection_mask[b,k,:n].sum()),
                        'candidate_count':n,'max_weight':float(w.max()),
                        'concentration':float((w*w).sum()),'effective_holdings':float(1/(w*w).sum()),
                        'ic':ic,'factor_return':float(losses['local_psi'][b,k]),
                        'index_adjustment_day':diag['index_adjustment_day'],**tt})
                    predictions.append(pd.DataFrame({'date':date,'horizon':h,'asset_id':ids,
                        'deep_factor':o.deep_factors[b,k,:n].numpy(),
                        'deep_factor_approx':o.deep_factor_approx[b,k,:n].numpy(),
                        'portfolio_attention':o.portfolio_attention[b,k,:n].numpy(),
                        'selected':o.stock_selection_mask[b,k,:n].numpy(),'portfolio_weight':w}))
                    interpretations.append(pd.DataFrame({'date':date,'horizon':h,
                        'factor_name':dataset.factor_columns,
                        'factor_attention':o.factor_attention_mean[b,k].numpy(),
                        'direction':o.factor_directions[k].numpy(),
                        'signed_contribution':o.signed_factor_coefficients[b,k].numpy()}))
            for name,frames in [('predictions',predictions),('factor_interpretability',interpretations)]:
                table = pa.Table.from_pandas(pd.concat(frames,ignore_index=True),preserve_index=False)
                if name not in writers:
                    writers[name] = pq.ParquetWriter(destination/f'{name}.parquet',table.schema)
                writers[name].write_table(table)
    finally:
        for writer in writers.values():
            writer.close()
    frame = pd.DataFrame(rows)
    frame.to_parquet(destination/'turnover_by_date.parquet',index=False)
    metrics = {'by_horizon':summarize_dates(frame,horizons),
        'holdings_basis':'average_holdings is effective holdings 1/sum(w_i^2); average_selected_stocks is the threshold-selected count.',
        'turnover_basis':'Separate horizon daily target weights, stable asset_id union; first observation excluded. Not executed-trade turnover; no drift or costs.',
        'performance_basis':'Overlapping h-day labels; arithmetic annualization 252/h. Max drawdown uses every h-th decision starting with first OOS date. Not a unique realized daily NAV.',
        'benchmark_basis':'Decision-date index-weighted adjusted VWAP return proxy; missing labels renormalized',
        'concentration_warning':'No position cap; concentration is reported only.'}
    write_metrics(destination/'test_metrics.json',metrics)
    return metrics


def _factor_equal_weight_portfolio(raw_factors, asset_mask):
    """Average 64 single-factor softmax portfolios into an equal-factor baseline."""
    values = np.asarray(raw_factors, dtype=np.float64)
    valid_assets = np.asarray(asset_mask, dtype=bool)
    factor_weights = []
    for factor in range(values.shape[1]):
        valid = valid_assets & np.isfinite(values[:, factor])
        if valid.sum() < 2:
            continue
        score = values[valid, factor]
        scale = score.std()
        if not np.isfinite(scale) or scale < 1e-12:
            continue
        score = (score - score.mean()) / scale
        score -= score.max()
        exp_score = np.exp(np.clip(score, -80.0, 80.0))
        weights = np.zeros(values.shape[0], dtype=np.float64)
        weights[valid] = exp_score / exp_score.sum()
        factor_weights.append(weights)
    if not factor_weights:
        weights = np.zeros(values.shape[0], dtype=np.float64)
        weights[valid_assets] = 1.0 / max(int(valid_assets.sum()), 1)
        return weights
    return np.mean(np.stack(factor_weights, axis=0), axis=0)


def _score_return_statistics(score, returns, valid):
    selected = valid & np.isfinite(score) & np.isfinite(returns)
    if selected.sum() < 3:
        return np.nan, np.nan
    x = score[selected]
    y = returns[selected]
    xc = x - x.mean()
    yc = y - y.mean()
    variance = float(np.mean(xc * xc))
    if variance <= 1e-12:
        return np.nan, np.nan
    return float(np.mean(xc * yc) / np.sqrt(variance * np.mean(yc * yc) + 1e-24)), float(
        np.mean(xc * yc) / variance
    )


def evaluate_factor_equal_weight(trainer, loaders, destination, max_batches=None):
    """Evaluate the equal-factor baseline on the same OOS dates and labels."""
    destination = Path(destination)
    horizons = trainer.config.model.horizons
    dataset = loaders.dataset
    rows = []
    prediction_rows = []
    previous = {h: None for h in horizons}
    for batch_index, raw in enumerate(loaders.test):
        if max_batches is not None and batch_index >= max_batches:
            break
        factors = raw['raw_factors'].numpy()
        returns = raw['forward_returns'].numpy()
        return_mask = raw['return_mask'].numpy().astype(bool)
        asset_mask = raw['asset_mask'].numpy().astype(bool)
        for b, date in enumerate(raw['dates']):
            n = len(raw['asset_ids'][b])
            weights = _factor_equal_weight_portfolio(factors[b, :n], asset_mask[b, :n])
            idx = dataset.dates.index(pd.Timestamp(date))
            _, diag = dataset.universe_diagnostics(idx)
            for k, horizon in enumerate(horizons):
                current = dict(zip(raw['asset_ids'][b], weights.tolist()))
                turnover = ({'one_way_turnover': np.nan, 'gross_turnover': np.nan}
                            if previous[horizon] is None else
                            turnover_from_asset_weights([previous[horizon], current]))
                previous[horizon] = current
                valid = return_mask[b, k, :n]
                portfolio_return = float(np.sum(np.where(valid, weights * returns[b, k, :n], 0.0)))
                ic, factor_return = _score_return_statistics(
                    weights, returns[b, k, :n], valid
                )
                rows.append({
                    'date': date, 'horizon': horizon,
                    'index': dataset.membership.index_name,
                    'portfolio_return': portfolio_return,
                    'benchmark_return': float(raw['benchmark_returns'][b, k]),
                    'holdings': int((weights > 1e-8).sum()),
                    'candidate_count': n,
                    'max_weight': float(weights.max()),
                    'concentration': float(np.sum(weights * weights)),
                    'effective_holdings': float(1.0 / np.sum(weights * weights)),
                    'ic': ic, 'factor_return': factor_return,
                    'index_adjustment_day': diag['index_adjustment_day'], **turnover,
                })
                prediction_rows.extend(
                    {
                        'date': date,
                        'horizon': horizon,
                        'asset_id': asset_id,
                        'portfolio_weight': float(weight),
                        'selected': bool(weight > 1e-8),
                    }
                    for asset_id, weight in zip(raw['asset_ids'][b], weights)
                )
    frame = pd.DataFrame(rows)
    frame.to_parquet(destination / 'factor_equal_weight_by_date.parquet', index=False)
    pd.DataFrame(prediction_rows).to_parquet(
        destination / 'factor_equal_weight_predictions.parquet', index=False
    )
    metrics = summarize_dates(frame, horizons)
    result = {
        'by_horizon': metrics,
        'holdings_basis':'average_holdings is effective holdings 1/sum(w_i^2); average_selected_stocks is the threshold-selected count.',
        'definition': 'Equal-weight average of valid single-factor cross-sectional softmax portfolios; no learned factor attention.',
        'turnover_basis': 'Same stable asset_id target-weight turnover as the model, calculated separately by horizon.',
    }
    write_metrics(destination / 'factor_equal_weight_metrics.json', result)
    return result
