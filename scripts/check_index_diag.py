from e2eai.config import load_config
from e2eai.workflows import load_configured_data
from e2eai.data.panel import AlphaPanelDataset
from e2eai.data.loaders import split_indices

config = load_config('configs/paper_index_universe_no_cap.yaml')
config.data.index_universe = 'csi500'
config.data.membership_path = '/cloud/E2EAI/data/index_universes/csi500_membership.npz'
config.data.benchmark_path = '/cloud/E2EAI/data/index_universes/csi500_benchmark.npy'
config.data.train_start, config.data.train_end = '2018-01-02', '2020-12-31'
config.data.validation_start, config.data.validation_end = '2021-01-04', '2021-12-31'
config.data.test_start, config.data.test_end = '2022-01-04', '2022-12-30'
source, _ = load_configured_data(config)
dataset = AlphaPanelDataset(source, config.model.horizons, config.data)
for date in ['2018-01-02','2018-06-29','2019-12-31','2020-12-31','2021-12-31','2022-01-04']:
    i = dataset.dates.index(__import__('pandas').Timestamp(date))
    print(dataset.universe_diagnostics(i)[1], flush=True)
