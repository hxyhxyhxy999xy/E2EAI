"""Dated historical membership snapshots: exact daily/as-of lookup, fail closed."""
from pathlib import Path
import numpy as np
import pandas as pd


class IndexMembership:
    def __init__(self, path, index_name):
        with np.load(Path(path), allow_pickle=False) as f:
            self.dates = pd.DatetimeIndex(pd.to_datetime(f['dates'].astype(str)))
            self.assets = f['asset_ids'].astype(str)
            self.members = f['members'].astype(bool)
            self.index_name = str(f['index_name'].item())
        if self.index_name != index_name:
            raise ValueError('Membership index does not match requested universe')
        if not self.dates.is_monotonic_increasing or self.dates.has_duplicates:
            raise ValueError('Membership dates must be unique and increasing')
        if self.members.shape != (len(self.dates), len(self.assets)):
            raise ValueError('Membership axes do not align')
        if len(set(self.assets)) != len(self.assets):
            raise ValueError('Duplicate membership asset IDs')

    def row(self, date):
        date = pd.Timestamp(date)
        pos = self.dates.searchsorted(date, side='right') - 1
        # Daily snapshot contract: do not silently forward fill missing trading
        # dates or carry the last source row beyond its coverage.
        if pos < 0 or self.dates[pos] != date:
            raise ValueError(f'Missing effective historical {self.index_name} membership on {date.date()}')
        return self.members[pos]

    def asset_set(self, date):
        return set(self.assets[self.row(date)])

    def changed(self, date):
        pos = self.dates.get_loc(pd.Timestamp(date))
        return None if pos == 0 else bool(np.any(self.members[pos] != self.members[pos-1]))
