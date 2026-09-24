import h5py
path = '/cloud/hdf5/historical/china_astock_2018.h5'
with h5py.File(path, 'r') as store:
    for key in store.keys():
        if 'citics' in key.lower():
            item = store[key]
            print(key, item.shape, str(item.dtype), flush=True)
