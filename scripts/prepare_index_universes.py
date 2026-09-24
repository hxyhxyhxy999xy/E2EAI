"""Extract historical daily index snapshots and matching VWAP benchmark labels."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from scripts.build_multihorizon_labels import MarketArrays, _adjusted_vwap, _as_dates, _decode

FIELDS = {'csi300': 'hs300_weight', 'csi500': 'zz500_weight', 'csi1000': 'zz1000_weight'}
LABEL_DEFINITIONS = ('paper_formula', 'legacy_t_plus_1')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--market', default='/cloud/hdf5/historical/china_astock_2018.h5')
    p.add_argument('--panel', default='/cloud/E2EAI/data/daily_strategy_trainonly64_corr08_panel')
    p.add_argument('--output', default='/cloud/E2EAI/data/index_universes')
    p.add_argument('--label-definition', choices=LABEL_DEFINITIONS, default='paper_formula')
    args = p.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    dates = _as_dates(np.load(Path(args.panel)/'dates.npy', allow_pickle=False))
    with h5py.File(args.market, 'r') as f:
        market = MarketArrays(f, 'h5')
        lookup = {d:i for i,d in enumerate(market.dates)}
        rows = np.asarray([lookup[d] for d in dates])
        columns = np.arange(len(market.assets))
        missing = set(FIELDS.values())-set(f.keys())
        if missing:
            raise ValueError(f'Missing historical membership fields: {missing}')
        for name,key in FIELDS.items():
            weights = np.asarray(f[key][:])
            members = np.isfinite(weights) & (weights > 0)
            if not members.any(axis=1).all():
                raise ValueError(f'Empty historical snapshots in {key}')
            np.savez_compressed(out/f'{name}_membership.npz',
                dates=market.dates.strftime('%Y-%m-%d').to_numpy(dtype='U10'),
                asset_ids=market.assets.astype(str), members=members, index_name=name)
            benchmark = np.full((len(dates),5), np.nan, np.float32)
            coverage = np.full_like(benchmark, np.nan)
            for start in range(0,len(rows),64):
                base = rows[start:start+64]
                entry = _adjusted_vwap(market, base+1, columns)
                # Use decision-date index weights; never future membership.
                w = np.where(members[base], weights[base], 0)
                for j,h in enumerate([3,5,10,15,20]):
                    exit_offset = h if args.label_definition == 'paper_formula' else h + 1
                    exit_price = _adjusted_vwap(market, base+exit_offset, columns)
                    ret = exit_price/entry-1
                    valid = np.isfinite(ret) & (w>0)
                    observed = np.where(valid,w,0).sum(axis=1)
                    coverage[start:start+len(base),j] = observed/w.sum(axis=1)
                    benchmark[start:start+len(base),j] = np.divide(
                        np.where(valid, w*np.nan_to_num(ret), 0).sum(axis=1), observed,
                        out=np.full(len(base),np.nan), where=observed>0)
            suffix = args.label_definition
            np.save(out/f'{name}_benchmark_{suffix}.npy', benchmark)
            np.save(out/f'{name}_benchmark_coverage_{suffix}.npy', coverage)
            meta = {'index':name,'source':args.market,'field':key,
                'coverage':[str(market.dates[0].date()),str(market.dates[-1].date())],
                'raw_count_min':int(members.sum(1).min()),'raw_count_max':int(members.sum(1).max()),
                'membership_change_dates':int(np.any(members[1:]!=members[:-1],axis=1).sum()),
                'as_of':'Exact decision-date historical snapshot; never backfilled from later rows',
                'publication_timestamp':'Not provided by source; historical effective-date semantics assumed',
                'benchmark':'Decision-date index-weighted VWAP return proxy; missing returns excluded and weights renormalized',
                'label_definition':args.label_definition,
                'return_definition':(
                    'adjusted VWAP[t+h]/adjusted VWAP[t+1]-1'
                    if args.label_definition == 'paper_formula'
                    else 'adjusted VWAP[t+h+1]/adjusted VWAP[t+1]-1'),
                'horizons':[3,5,10,15,20]}
            (out/f'{name}_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
            print(json.dumps(meta),flush=True)

if __name__ == '__main__':
    main()
