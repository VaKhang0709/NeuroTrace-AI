"""Selected-member export, measured train-only channel statistics, validation-only
isotonic map and validation-specific threshold. NEVER modifies V5 artifacts.

Dynamic time is engineering support; training/reference window length is512.
The DWT19 and bands19 contracts have equal widths but DIFFERENT semantics.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
from src.data_loaders.schema import Trajectory,DOMAINS
from src.preprocessing.v6_pipeline import transform,feature_names,BASE_CONTRACT,DWT_CONTRACT,CUTOFF
from src.inference.v6_ensemble_inference import load_model
from src.evaluation.v6_eval import aggregate
from src.training.v6_train import TrajectoryDataset
from scripts.v6_export_onnx import export_member,fit_calibration


def training_statistics(records,training_subjects,contract):
    total=np.zeros(19); squared=np.zeros(19); count=0
    for tr in records:
        if tr.subject_id not in set(training_subjects): continue
        if tr.synthetic: raise ValueError('Statistics require real training sources')
        windows,_=transform(tr,contract=contract)
        v=windows.astype(np.float64)
        total+=v.sum(axis=(0,2)); squared+=(v*v).sum(axis=(0,2)); count+=len(windows)*512
    if not count: raise ValueError('No training samples for measured statistics')
    mean=total/count; std=np.sqrt(np.maximum(squared/count-mean*mean,0))
    return mean,std,count


def validation_threshold(y,p):
    if set(np.unique(y))!={0,1}: raise ValueError('Threshold selection needs both validation diagnoses')
    candidates=np.unique(np.r_[.5,p])
    return float(max(candidates,key=lambda t:((p[y==1]>=t).mean()+(p[y==0]<t).mean()-1,-abs(t-.5))))


def export_pipeline(checkpoints,records,partition,output='models/v6',threshold_policy='validation_youden'):
    loaded=[load_model(p) for p in checkpoints]
    if not loaded: raise ValueError('Missing trained checkpoints')
    if not np.isfinite([c['val_auc'] for _,c in loaded]).all(): raise ValueError('Invalid member validation AUC')
    best=int(np.argmax([c['val_auc'] for _,c in loaded])); model,metadata=loaded[best]
    contract=metadata.get('feature_contract',BASE_CONTRACT)
    holdout=set(partition['holdout_subjects']); train=set(metadata['training_subjects'])
    valid=set(metadata['validation_subjects']); ssl=set(metadata.get('ssl_subjects',[]))
    if train&valid or (train|valid|ssl)&holdout or valid&ssl: raise ValueError('Export provenance overlap')
    if not (train|valid)<=set(partition['development_subjects']): raise ValueError('Checkpoint outside development partition')
    val_records=[tr for tr in records if tr.subject_id in valid]
    if {tr.subject_id for tr in val_records}!=valid: raise ValueError('Incomplete member-validation cohort')
    if any(tr.synthetic for tr in val_records): raise ValueError('Synthetic calibration forbidden')
    dataset=TrajectoryDataset(val_records,valid,contract=contract)
    samples=[dataset[i] for i in range(len(dataset))]
    calibration=dict(X=np.stack([s['x'].numpy() for s in samples]),domain=np.array([int(s['domain']) for s in samples]),
                     label=np.array([int(s['label']) for s in samples]),subject=np.array([s['subject'] for s in samples]))
    mapping=fit_calibration(model,metadata,calibration)
    logits=[]
    with torch.no_grad():
        for i in range(0,len(samples),4):
            x=torch.from_numpy(calibration['X'][i:i+4]); d=torch.from_numpy(calibration['domain'][i:i+4])
            logits.extend(model(x,d)['pd_logits'].numpy())
    _,y,raw_probability=aggregate(calibration['label'],expit(logits),calibration['subject'])
    calibrated=np.interp(raw_probability,mapping['x'],mapping['y'])
    if threshold_policy not in ('fixed_0.5','validation_youden'): raise ValueError('Unknown threshold policy')
    threshold=.5 if threshold_policy=='fixed_0.5' else validation_threshold(y,calibrated)
    if {tr.subject_id for tr in records if tr.subject_id in train}!=train:
        raise ValueError('Incomplete training-parent records for channel statistics')
    mean,std,n=training_statistics(records,train,contract)
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    onnx_path=output/'neurotrace_v6_tremorformer.onnx'
    parity=export_member(checkpoints[best],onnx_path,(calibration['X'],calibration['domain']),dynamic_sequence=True)
    config=dict(version='v6-gpu-export-1',feature_contract=contract,feature_names=feature_names(contract),
                shape=['batch',19,'sequence_length'],reference_sequence_length=512,minimum_engineering_sequence_length=64,
                parity_test_sequence_lengths=[256,384,512,768],canonical_rate_hz=100,
                mean=mean.tolist(),scale=np.where(std>1e-6,std,1).tolist(),measured_std=std.tolist(),
                statistics_samples=n,statistics_role='TRAIN_ONLY_DIAGNOSTICS_NOT_AN_ADDITIONAL_INPUT_TRANSFORM',
                apply_global_standardization=False,
                normalization=dict(type='per_window_population_zscore',epsilon=1e-6,
                                   exempt_channels=[17,18] if contract==BASE_CONTRACT else []),
                device_embedding_lookup={d:i for i,d in enumerate(DOMAINS)},device_filter_cutoff_hz=CUTOFF,
                classification_threshold=threshold,threshold_policy=threshold_policy,
                threshold_scope='selected-member validation only; not independently validated',
                calibration=mapping,calibration_scope='after subject-mean sigmoid probability; selected member only',
                selected_member=best,output_semantics='raw_logits',parity=parity,
                clinical_validation='NOT_ESTABLISHED',dynamic_time_scope='512 training contract; other lengths engineering-tested only')
    (output/'scaler_config_v6.json').write_text(json.dumps(config,indent=2,allow_nan=False)+'\n')
    return config


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--checkpoints',nargs='+',required=True)
    p.add_argument('--records',required=True); p.add_argument('--partition',required=True); p.add_argument('--output',default='models/v6')
    p.add_argument('--threshold-policy',choices=['fixed_0.5','validation_youden'],default='validation_youden')
    a=p.parse_args(); records=[Trajectory.from_dict(r) for r in json.loads(Path(a.records).read_text())]
    result=export_pipeline(a.checkpoints,records,json.loads(Path(a.partition).read_text()),a.output,a.threshold_policy)
    print(json.dumps(dict(output=a.output,contract=result['feature_contract'],parity=result['parity']),indent=2))

if __name__=='__main__': main()
