"""Synthetic correctness fixtures only. No clinical/performance assertions."""
from dataclasses import replace
import numpy as np
import pytest
import torch
from src.data_loaders.schema import Trajectory
from src.preprocessing.v6_pipeline import transform,FEATURES,wavelet_band
from src.augmentation.v6_augmentations import Augmenter,DEFAULT_PROBABILITIES,dtw_map
from src.models.v6_tremorformer import TremorFormer
from src.pretraining.v6_mae_pretrain import MaskedAutoencoder
from src.training.v6_train import MultiTaskLoss,validate_partition
from src.evaluation.v6_eval import aggregate,metrics,bootstrap,paired_mcnemar

@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(2); torch.manual_seed(42)


def trial(subject='p0',label=0,domain='touch',rate=100):
    t=np.arange(0,7,1/rate)
    x=np.cos(2*t)+.01*np.sin(2*np.pi*5*t); y=np.sin(2*t)
    return Trajectory(subject,'s0','spiral',domain,
           np.column_stack([x,y,t,.5+.1*np.sin(t),np.full_like(t,np.nan)]),label).validate()


def tiny(): return TremorFormer(d_model=32,blocks=1,heads=4,expansion=2,dropout=0)

@pytest.mark.parametrize('rate',[60,100,133,200])
def test_pipeline_rates(rate):
    x,q=transform(trial(rate=rate))
    assert x.shape[1:]==(19,512) and x.dtype==np.float32
    assert np.isfinite(x).all()
    assert np.all(x[:,17]==1) and np.all(x[:,18]==0)
    np.testing.assert_allclose(x[:,:17].mean(-1),0,atol=1e-5)


def test_wavelet_bands():
    t=np.arange(1024)/100
    a=np.sin(2*np.pi*6*t)
    b=wavelet_band(a,4,8); c=wavelet_band(a,8,12)
    assert np.std(b)>4*np.std(c)
    with pytest.raises(ValueError): transform(replace(trial(),samples=trial().samples[:100]))


def test_full_model_size_forward_backward():
    model=TremorFormer(); count=sum(p.numel() for p in model.parameters())
    assert 15_000_000<=count<=40_000_000
    out=model(torch.randn(1,19,512),torch.tensor([1]))
    assert out['pd_logits'].shape==(1,) and out['device_logits'].shape==(1,5)
    out['pd_logits'].sum().backward()
    assert torch.isfinite(model.stems[0].weight.grad).all()


def test_masked_pretrain_and_masked_multitask():
    model=tiny(); loss=MaskedAutoencoder(model)(torch.randn(2,19,512),torch.tensor([0,1]))
    assert torch.isfinite(loss); loss.backward()
    out=model(torch.randn(2,19,512),torch.tensor([0,1]))
    batch=dict(label=torch.tensor([0.,1.]),updrs=torch.full((2,),float('nan')),
               synthetic=torch.zeros(2,dtype=torch.bool),task=torch.full((2,),-1),
               age=torch.full((2,),-1),domain=torch.tensor([0,1]))
    objective=MultiTaskLoss()(out,batch); objective.backward(); assert torch.isfinite(objective)


def test_augmentation_and_parent_guards():
    donor=trial('donor',1,'wacom'); recipient=trial('hc',0,'touch')
    probabilities={k:1. for k in DEFAULT_PROBABILITIES}
    aug=Augmenter({'donor','hc'},[donor],probabilities)
    result,parents=aug(recipient)
    assert result.synthetic and result.label==1 and set(parents)=={'hc','donor'}
    assert np.isfinite(transform(result)[0]).all()
    with pytest.raises(ValueError): Augmenter({'hc'},[donor],probabilities)
    with pytest.raises(ValueError): aug(trial('outsider'))
    mapped=dtw_map(np.sin(np.arange(300)/30),np.random.default_rng(42))
    assert np.all(np.diff(mapped)>=0)


def test_partition_and_metrics():
    with pytest.raises(ValueError):
        validate_partition([trial()],dict(development_subjects=['p0'],holdout_subjects=['p0']))
    ids,y,p=aggregate([0,0,1,1],[.1,.3,.8,.9],['a','a','b','b'])
    assert metrics(y,p)['roc_auc']==1
    assert bootstrap(y,p,n=20)['roc_auc']['value']==1
    assert paired_mcnemar((ids,y,p),(ids,y,p))['exact_p']==1
    with pytest.raises(ValueError): aggregate([0,1],[.2,.8],['same','same'])


def test_calibration_and_ssl_reject_parent_leakage(tmp_path):
    from scripts.v6_export_onnx import fit_calibration
    from src.pretraining.v6_mae_pretrain import pretrain
    from src.training.v6_train import TrajectoryDataset
    from torch.utils.data import DataLoader
    model=tiny()
    calibration=dict(subject=np.array(['train']),X=np.zeros((1,19,512)),domain=np.array([0]),label=np.array([0]))
    with pytest.raises(ValueError):
        fit_calibration(model,dict(validation_subjects=['val'],training_subjects=['train']),calibration)
    ds=TrajectoryDataset([trial('val')],{'val'})
    with pytest.raises(ValueError):
        pretrain(model,DataLoader(ds,batch_size=1),{'train'},tmp_path/'never.pth',epochs=1)


def test_subject_report_bootstrap_and_probe(tmp_path):
    from src.evaluation.v6_eval import report,device_probe
    prediction=dict(probability=np.array([.1,.2,.8,.9]),label=np.array([0,0,1,1]),
                    subject=np.array(['a','b','c','d']),domain=np.array(['touch','mouse','touch','mouse']))
    values=report(prediction,['a','b','c','d'],tmp_path/'report.md',replicates=20,plot_dir=tmp_path/'plots')
    assert values['all']['subjects']==4
    assert (tmp_path/'plots/roc_calibration.png').exists()
    assert device_probe([[0],[1]],['touch','mouse'],['a','b'])['status']=='INCONCLUSIVE'


def test_device_probe_detects_obvious_domain_leakage():
    from src.evaluation.v6_eval import device_probe
    features=[]; domains=[]; subjects=[]
    for i in range(10):
        for domain in ('mouse','touch'):
            features.append([float(domain=='touch')*10, i*.001])
            domains.append(domain); subjects.append(f's{i}')
    result=device_probe(features,domains,subjects)
    assert result['status']=='MEASURED' and result['mean_auc']>.99
    assert result['gate_passed'] is False
