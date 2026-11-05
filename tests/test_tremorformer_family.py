"""Engineering correctness,not distillation/holdout evidence or production weights.

Random model weights are explicitly scoped to export mechanics. Recorded input
fixtures test the unchanged feature path; clinical metrics are NEVER fabricated.
"""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from src.models.tremorformer_family import TremorFormer,parameter_counts,load_student
from src.models.v6_tremorformer import TremorFormer as Original
from src.training.distill import distillation_loss,FrozenONNXTeacher,teacher_targets
from src.training.family_data import validate_scaler,validate_teacher_groups,windows
from src.preprocessing.v6_pipeline import feature_names,DWT_CONTRACT,CUTOFF
from src.data_loaders.schema import DOMAINS
from src.data_loaders.uci395_loader import load
from src.export.export_family import export_fp32,compress,parity,session,check_signature,LIMITS
from src.eval.benchmark_family import latency,benchmark


@pytest.fixture(scope='module')
def recorded_windows():
    records=load(Path(__file__).parent/'fixtures/uci395')
    return windows(records,{r.subject_id for r in records})


def test_exact_family_counts_and_pro_checkpoint_compatibility():
    assert parameter_counts()==dict(nano=718404,base=3648400,pro=15779536)
    torch.manual_seed(17); original=Original()
    torch.manual_seed(17); pro=TremorFormer('pro')
    assert original.config==pro.config
    assert original.state_dict().keys()==pro.state_dict().keys()
    for key,value in original.state_dict().items(): assert torch.equal(value,pro.state_dict()[key])


@pytest.mark.parametrize('tier',['nano','base'])
def test_students_keep_grl_and_nano_separable_stems(tier):
    model=TremorFormer(tier)
    h=torch.randn(2,model.config['d_model'],requires_grad=True)
    reverse=torch.autograd.grad(model.device_head(model.grl(h)).sum(),h)[0]
    forward=torch.autograd.grad(model.device_head(h).sum(),h)[0]
    torch.testing.assert_close(reverse,-.1*forward)
    if tier=='nano':
        assert len(model.blocks)==2
        assert all(stem[0].groups==19 and stem[1].kernel_size==(1,) for stem in model.stems)
    else: assert len(model.blocks)==4


def test_binary_kl_is_nonzero_and_teacher_detached():
    s=torch.zeros(2,requires_grad=True); t=torch.tensor([2.,-2.],requires_grad=True); y=torch.tensor([1.,0.])
    loss=distillation_loss(s,t,y,temperature=3.,alpha=1.)
    assert loss.item()>0; loss.backward()
    assert s.grad.abs().sum()>0 and t.grad is None
    same=distillation_loss(t,t,y,temperature=3.,alpha=1.)
    assert abs(same.item())<1e-5
    bce=distillation_loss(t,t,y,alpha=0.)
    torch.testing.assert_close(bce,torch.nn.functional.binary_cross_entropy_with_logits(t,y))


@pytest.mark.parametrize('T,alpha',[(0,.5),(-1,.5),(3,-.1),(3,1.1)])
def test_invalid_kd_settings_fail(T,alpha):
    with pytest.raises(ValueError): distillation_loss(torch.zeros(2),torch.ones(2),torch.zeros(2),T,alpha)


def test_original_teacher_groups_reused_and_leakage_rejected():
    # Abstract group sets exercise guards only,not a generated clinical cohort.
    meta=dict(training_subjects=['train_A','train_B'],validation_subjects=['valid_A'],ssl_subjects=['train_A'])
    dev={'train_A','train_B','valid_A'}; holdout={'LOCKED_test'}
    train,valid=validate_teacher_groups(meta,dev,holdout)
    assert train=={'train_A','train_B'} and valid=={'valid_A'}
    for update in (dict(ssl_subjects=['valid_A']),dict(ssl_subjects=['LOCKED_test']),
                   dict(training_subjects=['train_A','valid_A']),dict(training_subjects=['train_A'])):
        with pytest.raises(ValueError): validate_teacher_groups({**meta,**update},dev,holdout)


def test_shared_scaler_requires_exact_existing_normalization(tmp_path):
    # Structural config,not fitted statistics; never written as a release scaler.
    config=dict(feature_contract=DWT_CONTRACT,feature_names=list(feature_names(DWT_CONTRACT)),
        normalization=dict(type='per_window_population_zscore',epsilon=1e-6,exempt_channels=[]),
        apply_global_standardization=False,canonical_rate_hz=100,device_filter_cutoff_hz=CUTOFF,
        device_embedding_lookup={d:i for i,d in enumerate(DOMAINS)},output_semantics='raw_logits')
    path=tmp_path/'test_config.json';path.write_text(json.dumps(config));validate_scaler(path)
    config['apply_global_standardization']=True;path.write_text(json.dumps(config))
    with pytest.raises(ValueError): validate_scaler(path)


@pytest.fixture(scope='module')
def engineering_exports(tmp_path_factory,recorded_windows):
    root=tmp_path_factory.mktemp('UNTRAINED_engineering_only'); results={}
    for i,(tier,precision) in enumerate([('nano','int8'),('base','fp16')]):
        torch.manual_seed(42+i);model=TremorFormer(tier).eval()
        raw=root/(tier+'_fp32.onnx');path=root/(tier+'.onnx')
        export_fp32(model,raw);compress(raw,path,precision)
        fp32=parity(model,raw,'fp32',recorded_windows)
        compressed=parity(model,path,precision,recorded_windows)
        results[tier]=dict(model=model,path=path,fp32=fp32,compressed=compressed)
    return results


@pytest.mark.parametrize('tier',['nano','base'])
def test_compressed_export_dynamic_signatures_size_and_parity(engineering_exports,tier):
    value=engineering_exports[tier]
    assert value['path'].stat().st_size<=LIMITS[tier]
    assert value['fp32']['passed'] and value['compressed']['passed']
    assert value['compressed']['tensors']==100 and value['compressed']['recorded_validation_tensors']>0
    check_signature(session(value['path']))


def test_onnx_teacher_is_always_eval_frozen_and_caches_same_targets(engineering_exports,recorded_windows):
    teacher=FrozenONNXTeacher(engineering_exports['base']['path'])
    teacher.train(True);assert not teacher.training and not list(teacher.parameters())
    x=torch.from_numpy(recorded_windows['X']).requires_grad_(True);d=torch.from_numpy(recorded_windows['device_id'])
    out=teacher(x,d)
    assert not out.requires_grad
    np.testing.assert_allclose(teacher_targets(teacher,recorded_windows),out.numpy(),atol=1e-6)


def test_latency_is_single_thread_and_no_placeholder_holdout_report(engineering_exports,recorded_windows,tmp_path):
    runtime=session(engineering_exports['nano']['path'])
    assert runtime.get_session_options().intra_op_num_threads==1
    measured=latency(runtime,recorded_windows['X'][0],recorded_windows['device_id'][0],repeats=3,warmup=1)
    assert measured['median_ms']>0 and measured['shape']==[1,19,512]
    (tmp_path/'family_manifest.json').write_text('{"accepted":false}')
    report=tmp_path/'SHOULD_NOT_EXIST.md'
    with pytest.raises(ValueError): benchmark(SimpleNamespace(family=tmp_path,output=report))
    assert not report.exists()


def test_incomplete_student_cannot_be_promoted(tmp_path):
    ck=tmp_path/'incomplete.pth'
    torch.save(dict(family_tier='nano',feature_contract=DWT_CONTRACT,complete=False,epochs_requested=40,epochs_completed=1),ck)
    with pytest.raises(ValueError,match='incomplete'): load_student(ck)


def test_new_family_notebook_valid_clean_and_python_parseable():
    import nbformat
    notebook=nbformat.read('notebooks/v6_family_distill.ipynb',4);nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type=='code':
            assert not cell.outputs and cell.execution_count is None
            ast.parse(cell.source)
    text='\n'.join(c.source for c in notebook.cells)
    assert 'USE_SYNTHETIC=False' in text and '40,8,3.,.5' in text and 'partition_subjects' not in text
