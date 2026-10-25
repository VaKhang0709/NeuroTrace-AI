"""Mock-only acquisition/generation/export checks. No patient/clinical evidence."""
import ast
import hashlib
import io
import json
import urllib.error
import zipfile
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
import torch
from src.data_ingestion.auto_downloader import acquire,extract_zip,download
from src.data_loaders.public_trajectory_parser import load_uci
from src.augmentation.tremor_injector import mock_sources,generate,eligible_rows,materialize
from src.preprocessing.v6_pipeline import transform,DWT_CONTRACT,BASE_CONTRACT,feature_names
from src.data_loaders.synthetic_manifest_loader import SyntheticManifestDataset,CombinedTrainingDataset
from src.models.v6_tremorformer import TremorFormer

@pytest.fixture(autouse=True)
def threads(): torch.set_num_threads(2)


def test_local_archive_and_strict_uci_parser(tmp_path):
    t=np.arange(700)*10
    a=np.column_stack([np.cos(t/1000),np.sin(t/1000),t*0,.5+.1*np.sin(t/200),t*0,t,t*0])
    stream=io.StringIO(); np.savetxt(stream,a,delimiter=';')
    archive=tmp_path/'fixture.zip'
    with zipfile.ZipFile(archive,'w') as z: z.writestr('P0001.txt',stream.getvalue())
    outcomes=acquire(tmp_path/'raw',archive,report=tmp_path/'summary.md')
    assert outcomes[0]['status']=='EXTRACTED_NOT_CLINICALLY_VALIDATED'
    trials=list(load_uci(tmp_path/'raw/uci_395'))
    assert trials[0].subject_id=='uci::P0001' and trials[0].label==1
    assert trials[0].samples[-1,2]==6.99 and np.isnan(trials[0].samples[:,4]).all()
    evil=tmp_path/'evil.zip'
    with zipfile.ZipFile(evil,'w') as z: z.writestr('../escape.txt','bad')
    with pytest.raises(ValueError): extract_zip(evil,tmp_path/'unsafe')
    assert not (tmp_path/'escape.txt').exists()


def test_network_failure_is_blocked_not_success(tmp_path,monkeypatch):
    def blocked(*args,**kwargs): raise urllib.error.URLError('network blocked')
    monkeypatch.setattr('urllib.request.urlopen',blocked)
    rows=acquire(tmp_path/'raw',report=tmp_path/'summary.md')
    assert rows[0]['status']=='BLOCKED'
    assert 'BLOCKED' in (tmp_path/'summary.md').read_text()
    assert not (tmp_path/'raw/uci_395').exists()


def test_http_checksum_and_partial_cleanup(tmp_path,monkeypatch):
    class Response(io.BytesIO):
        def geturl(self): return 'https://zenodo.org/file.zip'
    monkeypatch.setattr('urllib.request.urlopen',lambda *a,**kw:Response(b'fixture bytes'))
    path=tmp_path/'file.zip'
    with pytest.raises(ValueError): download('https://zenodo.org/file.zip',path,'0'*64)
    assert not path.exists() and not path.with_suffix('.zip.partial').exists()
    assert download('https://zenodo.org/file.zip',path)==hashlib.sha256(b'fixture bytes').hexdigest()


def test_generation_two_parent_filters_and_mock_rejection(tmp_path):
    records=mock_sources(); parents={r.subject_id for r in records}
    manifest=tmp_path/'manifest.csv'
    result=generate(records,parents,tmp_path/'samples',manifest,per_domain=3,mock=True)
    assert result['trajectories']==6 and result['parent_subjects']==3
    with pytest.raises(ValueError): eligible_rows(manifest,parents)
    assert eligible_rows(manifest,{'mock_touch'},allow_mock=True)==[] # donor absent!
    rows=eligible_rows(manifest,{'mock_touch','mock_donor'},allow_mock=True)
    assert len(rows)==3 and {r['domain'] for r in rows}=={'touch'}
    assert materialize(rows[0]).synthetic
    ds=SyntheticManifestDataset(manifest,{'mock_touch','mock_donor'},contract=DWT_CONTRACT,allow_mock=True)
    assert ds[0]['x'].shape==(19,512) and ds[0]['synthetic'].item()
    assert ds.provenance_subjects==['mock_donor','mock_touch']
    with pytest.raises(ValueError): SyntheticManifestDataset(manifest,parents)
    with pytest.raises(ValueError): generate(records,parents,tmp_path/'bad',tmp_path/'bad.csv',per_domain=1)


def test_versioned_channel_contracts():
    source=mock_sources()[1]
    original,_=transform(source,contract=BASE_CONTRACT)
    requested,_=transform(source,contract=DWT_CONTRACT)
    assert original.shape==requested.shape
    assert feature_names(DWT_CONTRACT)[-6:]==('wavelet_db4_l1','wavelet_db4_l2','wavelet_db4_l3','wavelet_db4_l4','wavelet_db4_l5','v5_proxy_p')
    np.testing.assert_allclose(requested.mean(-1),0,atol=1e-5)
    assert not np.array_equal(original,requested)
    with pytest.raises(ValueError): transform(source,contract='unversioned')


def test_notebook_is_valid_and_cells_parse():
    nbformat=pytest.importorskip('nbformat')
    notebook=nbformat.read('notebooks/v6_gpu_beast_training.ipynb',as_version=4); nbformat.validate(notebook)
    assert all(not c.get('outputs') for c in notebook.cells if c.cell_type=='code')
    for c in notebook.cells:
        if c.cell_type=='code': ast.parse(c.source)


def test_dynamic_export_pipeline_on_mock_checkpoint(tmp_path):
    from scripts.export_v6_onnx_pipeline import export_pipeline
    records=[]
    for i in range(8):
        records.append(replace(mock_sources()[1],subject_id=f'fixture_{i}',label=i%2,source='unit_test_mock'))
    model=TremorFormer(d_model=32,blocks=1,heads=4,expansion=2,dropout=0.)
    path=tmp_path/'MOCK_UNTRAINED.pth'
    partition=dict(development_subjects=[r.subject_id for r in records[:6]],holdout_subjects=[r.subject_id for r in records[6:]])
    torch.save(dict(model_config=model.config,model_state_dict=model.state_dict(),val_auc=.5,
                    feature_contract=DWT_CONTRACT,training_subjects=partition['development_subjects'][:4],
                    validation_subjects=partition['development_subjects'][4:],ssl_subjects=[]),path)
    result=export_pipeline([path],records,partition,tmp_path/'exports',threshold_policy='fixed_0.5')
    assert result['parity']['passed'] and result['parity']['dynamic_sequence']
    assert result['feature_contract']==DWT_CONTRACT
    assert result['classification_threshold']==.5 and not result['apply_global_standardization']
    assert len(result['mean'])==len(result['scale'])==19
    import onnxruntime as ort
    session=ort.InferenceSession(str(tmp_path/'exports/neurotrace_v6_tremorformer.onnx'))
    assert session.get_inputs()[0].shape==['batch',19,'sequence_length']
    assert result['parity']['max_logit_error']<1e-5


def test_fold_training_consumes_synthetic_only_with_explicit_mock_permission(tmp_path):
    from torch.utils.data import DataLoader
    from src.training.v6_train import TrajectoryDataset,train_fold
    records=mock_sources(); train_parents={'mock_donor','mock_touch'}
    manifest=tmp_path/'train_mock.csv'
    generate(records,train_parents,tmp_path/'synth',manifest,per_domain=1,mock=True)
    synthetic=SyntheticManifestDataset(manifest,train_parents,DWT_CONTRACT,allow_mock=True)
    real=TrajectoryDataset(records[:2],train_parents,contract=DWT_CONTRACT)
    combined=CombinedTrainingDataset(real,synthetic)
    val=[replace(records[0],subject_id='val_pd'),replace(records[1],subject_id='val_hc')]
    validation=TrajectoryDataset(val,{'val_pd','val_hc'},contract=DWT_CONTRACT)
    model=TremorFormer(d_model=32,blocks=1,heads=4,expansion=2,dropout=0.)
    model.feature_contract=DWT_CONTRACT
    path=train_fold(model,DataLoader(combined,batch_size=4),DataLoader(validation,batch_size=4),tmp_path/'fold',epochs=1)
    saved=torch.load(path,weights_only=True)
    assert saved['feature_contract']==DWT_CONTRACT
    assert set(saved['synthetic_parent_subjects'])==train_parents
    assert not set(saved['training_subjects'])&set(saved['validation_subjects'])
    import csv
    row=next(csv.DictReader((tmp_path/'fold/epochs.csv').open()))
    assert int(row['synthetic_windows'])>0
    assert {'val_subject_auc','val_pr_auc','val_ece','val_sensitivity','val_specificity'}<=set(row)


def test_gpu_runner_fails_closed_without_cuda(monkeypatch):
    from scripts.train_v6_gpu_standalone import gpu_info
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError): gpu_info(require=True)
    assert gpu_info(require=False) is None
