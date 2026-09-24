"""Extract the historical level-1 CITICS field aligned to the selected panel."""
from pathlib import Path
import h5py
import numpy as np
from scripts.build_multihorizon_labels import _as_dates, _decode

def main():
    market_path = '/cloud/hdf5/historical/china_astock_2018.h5'
    panel_path = Path('/cloud/E2EAI/data/daily_strategy_trainonly64_corr08_panel')
    output = Path('/cloud/E2EAI/data/index_universes/industry_level1_h5_trainonly64_2018_2021.npz')
    panel_dates = _as_dates(np.load(panel_path/'dates.npy', allow_pickle=False))
    panel_assets = _decode(np.load(panel_path/'asset_ids.npy', allow_pickle=False))
    with h5py.File(market_path, 'r') as store:
        source_dates = _as_dates(np.asarray(store['tradeDate'][:]))
        source_assets = _decode(np.asarray(store['ticker'][:]))
        date_pos = {date:i for i,date in enumerate(source_dates)}
        asset_pos = {asset:i for i,asset in enumerate(source_assets)}
        rows = np.asarray([date_pos[date] for date in panel_dates], dtype=np.int64)
        cols = np.asarray([asset_pos[asset] for asset in panel_assets], dtype=np.int64)
        values = np.asarray(store['citics_sector_level1'][rows, :][:, cols], dtype=np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, index=panel_dates.strftime('%Y-%m-%d').to_numpy(dtype='U10'),
                        columns=panel_assets, values=values)
    print({'output':str(output), 'shape':list(values.shape),
           'missing_fraction':float((~np.isfinite(values)).mean()),
           'date_range':[str(panel_dates[0].date()),str(panel_dates[-1].date())]}, flush=True)

if __name__ == '__main__':
    main()
