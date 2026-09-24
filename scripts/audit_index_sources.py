"""Read-only inventory of market fields and historical index membership."""
from pathlib import Path
import json
import h5py
import numpy as np

def main():
    root = Path('/cloud/hdf5')
    for path in sorted(root.rglob('*')):
        if path.suffix.lower() not in {'.h5', '.hdf5', '.pkl', '.npz'}:
            continue
        print('FILE', path, flush=True)
        if path.suffix.lower() not in {'.h5', '.hdf5'}:
            continue
        with h5py.File(path, 'r') as f:
            print('FIELDS', list(f.keys()), flush=True)
            for key in f.keys():
                if any(s in key.lower() for s in ('weight', '300', '500', '1000', 'date')):
                    x = f[key]
                    if not isinstance(x, h5py.Dataset):
                        continue
                    print(key, x.shape, str(x.dtype), dict(x.attrs), flush=True)
                    if x.ndim == 1:
                        print('BOUNDS', str(x[0]), str(x[-1]))
                    elif x.ndim == 2:
                        for i in sorted(set([0, min(490, x.shape[0]-1), x.shape[0]-1])):
                            a=x[i]
                            print('SAMPLE', i, int(np.sum(np.isfinite(a) & (a>0))))

if __name__ == '__main__':
    main()
