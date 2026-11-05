"""Read-only family inputs. NEVER regenerates a partition or links new identities."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from src.data_loaders.schema import Trajectory,DOMAINS
from src.preprocessing.v6_pipeline import transform,feature_names,DWT_CONTRACT,CUTOFF


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''): h.update(block)
    return h.hexdigest()


def validate_scaler(path):
    config=json.loads(Path(path).read_text())
    expected=dict(type='per_window_population_zscore',epsilon=1e-6,exempt_channels=[])
    if (config.get('feature_contract')!=DWT_CONTRACT or config.get('feature_names')!=list(feature_names(DWT_CONTRACT))
        or config.get('apply_global_standardization') is not False or config.get('normalization')!=expected
        or config.get('canonical_rate_hz')!=100 or config.get('device_filter_cutoff_hz')!=CUTOFF
        or config.get('device_embedding_lookup')!={d:i for i,d in enumerate(DOMAINS)}):
        raise ValueError('Scaler incompatible with the unchanged DWT19 extractor; do not silently adapt it')
    if config.get('output_semantics')!='raw_logits': raise ValueError('Teacher must return raw logits')
    return config


def validate_teacher_groups(metadata,dev,holdout,unlabeled=()):
    dev,holdout,unlabeled=map(set,(dev,holdout,unlabeled))
    train=set(metadata['training_subjects']); valid=set(metadata['validation_subjects']); ssl=set(metadata.get('ssl_subjects',[]))
    if (not train or not valid or train&valid or (train|valid)!=dev or (train|valid|ssl)&holdout
        or valid&ssl or not ssl<=(train|unlabeled)):
        raise ValueError('Teacher/student parent leakage or incomplete development provenance')
    return train,valid


def load_context(records_path,partition_path,scaler_path,metadata_path,teacher_onnx):
    # Missing inputs intentionally fail; no UCI downloader, split constructor or mock fallback here.
    config=validate_scaler(scaler_path)
    if sha256(teacher_onnx)!=config.get('parity',{}).get('onnx_sha256'):
        raise ValueError('Teacher ONNX does not match the shared production scaler hash')
    partition=json.loads(Path(partition_path).read_text())
    parts=[partition.get(k,[]) for k in ('development_subjects','holdout_subjects','unlabeled_subjects')]
    if any(len(p)!=len(set(p)) for p in parts): raise ValueError('Duplicate partition identities')
    dev,holdout,unlabeled=map(set,parts)
    if (len(dev),len(holdout),len(unlabeled))!=(62,15,0) or dev&holdout:
        raise ValueError('Require the ORIGINAL locked 62 development / 15 holdout partition, no regeneration')
    raw=json.loads(Path(records_path).read_text())
    records=[Trajectory.from_dict(t) for t in raw]
    if {t.subject_id for t in records}!=dev|holdout: raise ValueError('Missing/extra cohort parents; no silent exclusions')
    labels={}
    for tr in records:
        if tr.synthetic or 'mock' in tr.source.lower() or tr.label not in (0,1): raise ValueError('Family requires real known diagnoses; synthetic disabled')
        if tr.subject_id in labels and labels[tr.subject_id]!=tr.label: raise ValueError('Conflicting parent labels')
        labels[tr.subject_id]=tr.label
    path=Path(metadata_path)
    if path.suffix.lower()=='.json':
        metadata=json.loads(path.read_text())
        if metadata.get('teacher_onnx_sha256')!=sha256(teacher_onnx) or metadata.get('partition_sha256')!=sha256(partition_path):
            raise ValueError('ONNX-only teacher provenance must bind BOTH original partition bytes and teacher hash')
    else: metadata=torch.load(path,map_location='cpu',weights_only=True)
    if metadata.get('feature_contract')!=DWT_CONTRACT: raise ValueError('Teacher provenance needs DWT19 contract')
    train,valid=validate_teacher_groups(metadata,dev,holdout,unlabeled)
    if len(valid)!=config.get('calibration',{}).get('validation_subject_count'):
        raise ValueError('Teacher validation count differs from production calibration provenance')
    for ids in (train,valid,holdout):
        if {labels[p] for p in ids}!={0,1}: raise ValueError('Each clinical subset needs both diagnoses')
    context=dict(partition_sha256=sha256(partition_path),scaler_sha256=sha256(scaler_path),
                 teacher_onnx_sha256=sha256(teacher_onnx),teacher_metadata_sha256=sha256(metadata_path))
    return records,partition,metadata,context


def windows(records,subjects):
    X=[]; y=[]; domain=[]; parents=[]
    for tr in records:
        if tr.subject_id not in set(subjects): continue
        x,_=transform(tr,contract=DWT_CONTRACT) # EXACT existing normalization; no global scaler pass.
        X.extend(x); y.extend([tr.label]*len(x)); domain.extend([DOMAINS.index(tr.domain)]*len(x)); parents.extend([tr.subject_id]*len(x))
    if set(parents)!=set(subjects): raise ValueError('Missing requested parents in feature data')
    return dict(X=np.asarray(X,dtype=np.float32),label=np.asarray(y,dtype=np.float32),
                device_id=np.asarray(domain,dtype=np.int64),subject=np.asarray(parents))
