from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
import torch
from e2eai.config import ExperimentConfig, load_config
from e2eai.data.index_universe import IndexMembership
from e2eai.data.panel import load_alpha_panel, AlphaPanelDataset
from e2eai.models.portfolio import GatedPortfolioAllocator
from e2eai.training.losses import E2EAILoss
from e2eai.evaluation.metrics import turnover_from_asset_weights


def test_no_cap_does_not_call_projection_and_return_gradient():
    torch.manual_seed(8)
    model = GatedPortfolioAllocator(4,2,gamma_p=0.04,allocation_mode='long_only_softmax')
    context = torch.randn(2,12,4)
    deep = torch.randn(2,2,12)
    with patch('e2eai.models.portfolio.project_capped_simplex',side_effect=AssertionError('projection called')):
        out = model(context,deep,torch.ones(2),torch.ones(2,12,dtype=torch.bool),cap_mode='capped_simplex')
    assert torch.all(out.portfolio_weights[~out.stock_mask] == 0)
    assert torch.all(out.portfolio_weights >= 0)
    assert torch.allclose(out.portfolio_weights.sum(-1),torch.ones(2,2))
    ret = torch.randn(2,2,12)*0.01
    loss = E2EAILoss(allocation_mode='long_only_softmax')
    d = loss(deep,deep,out.portfolio_weights,ret,torch.ones(2),alpha=torch.zeros(2,2),psi=torch.zeros(2,2))
    assert 'upper_bound_loss' not in d
    assert torch.equal(d['portfolio_loss'],d['portfolio_return_loss'])
    d['portfolio_loss'].backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_threshold_and_one_stock_fallback():
    model = GatedPortfolioAllocator(2,1,gamma_p=0.99,allocation_mode='long_only_softmax')
    out = model(torch.zeros(1,6,2),torch.zeros(1,1,6),torch.ones(1),torch.ones(1,6,dtype=torch.bool))
    assert out.stock_mask.sum() == 1
    assert out.portfolio_weights.max() == 1


def test_membership_has_no_future_backfill_and_independent_indices(tmp_path):
    dates=np.array(['2020-01-02','2020-01-03'])
    assets=np.array(['A','B','C'])
    path=tmp_path/'members.npz'
    np.savez(path,dates=dates,asset_ids=assets,members=[[1,0,0],[0,1,1]],index_name='csi300')
    m=IndexMembership(path,'csi300')
    assert m.asset_set('2020-01-02') == {'A'}
    assert m.asset_set('2020-01-03') == {'B','C'}
    assert m.changed('2020-01-03')
    for date in ['2020-01-01','2020-01-04']:
        with pytest.raises(ValueError): m.row(date)
    with pytest.raises(ValueError): IndexMembership(path,'csi500')


def test_index_pool_over_200_and_decision_date_filters(tmp_path):
    n=305
    dates=np.array(['2020-01-02','2020-01-03'])
    assets=np.array([f'{i:06d}' for i in range(n)])
    panel=tmp_path/'panel'; panel.mkdir()
    values={'dates':dates,'asset_ids':assets,'alpha_tensor':np.ones((2,3,n),np.float32),
            'forward_returns':np.full((2,n),np.nan,np.float32),
            'eligible_mask':np.ones((2,n),bool)}
    values['eligible_mask'][0,0]=False
    values['alpha_tensor'][0,:,1]=np.nan
    for k,v in values.items(): np.save(panel/f'{k}.npy',v)
    mp=tmp_path/'members.npz'
    np.savez(mp,dates=dates,asset_ids=assets,members=np.ones((2,n),bool),index_name='csi300')
    cfg=ExperimentConfig(); cfg.data.path=str(panel); cfg.data.index_universe='csi300'
    cfg.data.membership_path=str(mp); cfg.model.num_factors=3; cfg.model.horizons=[1]
    cfg.validate()
    ds=AlphaPanelDataset(load_alpha_panel(panel,cfg.data),[1],cfg.data)
    assert len(ds[0]['asset_ids'])==303
    assert len(ds[1]['asset_ids'])==305
    assert not ds[0]['return_mask'].any()  # Unknown outcomes never filter candidates.
    cfg.data.max_assets_per_date=200
    with pytest.raises(ValueError): cfg.validate()


def test_turnover_tracks_entries_and_exits():
    t=turnover_from_asset_weights([{'A':0.5,'B':0.5},{'B':0.25,'C':0.75}])
    assert t == {'gross_turnover':1.5,'one_way_turnover':0.75}


def test_formal_config_is_uncapped():
    cfg=load_config('configs/paper_index_universe_no_cap.yaml')
    assert cfg.model.portfolio.allocation_mode=='long_only_softmax'
    assert cfg.model.portfolio.min_selected_stocks==1
    assert cfg.data.max_assets_per_date is None
    assert cfg.model.horizons==[3,5,10,15,20]
