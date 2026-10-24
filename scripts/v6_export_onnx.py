"""Export five fold models or one validation-selected member, plus honest calibration.

Isotonic map is fitted to chosen member's SUBJECT-mean validation probabilities,
never test data or the full ensemble's OOF-incompatible development predictions.
All artifacts remain experimental until externally validated; V5 is untouched.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from src.inference.v6_ensemble_inference import load_model
from src.preprocessing.v6_pipeline import FEATURES,CUTOFF,BINARY_CHANNELS
from src.data_loaders.schema import DOMAINS
from src.evaluation.v6_eval import aggregate

class ClinicalONNX(nn.Module):
    def __init__(self,model): super().__init__(); self.model=model
    def forward(self,x,domain):
        return self.model(x,domain)['pd_logits']


def export_member(checkpoint,output,real_windows=None,seed=42):
    import onnx
    import onnxruntime as ort
    model,metadata=load_model(checkpoint); model.eval(); wrapper=ClinicalONNX(model).eval()
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    c=model.config['in_channels']
    fastpath=torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with torch.no_grad():
            torch.onnx.export(wrapper,(torch.zeros(1,c,512),torch.tensor([4])),str(output),
                  opset_version=17,dynamo=False,do_constant_folding=True,
                  input_names=['trajectory','device_id'],output_names=['logits'],
                  dynamic_axes={'trajectory':{0:'batch'},'device_id':{0:'batch'},'logits':{0:'batch'}})
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)
        wrapper.eval(); model.eval()
    onnx.checker.check_model(onnx.load(str(output)))
    session=ort.InferenceSession(str(output),providers=['CPUExecutionProvider'])
    rng=np.random.default_rng(seed); maximum=0.; real_count=0
    with torch.no_grad():
        for i in range(100):
            if real_windows is not None and i>=50 and i-50<len(real_windows[0]):
                x=real_windows[0][i-50:i-49].astype(np.float32)
                domain=real_windows[1][i-50:i-49].astype(np.int64); real_count+=1
            else:
                bs=[1,2,4][i%3]; x=rng.normal(size=(bs,c,512)).astype(np.float32)
                domain=rng.integers(0,5,bs,dtype=np.int64)
            pt=wrapper(torch.from_numpy(x),torch.from_numpy(domain)).numpy()
            runtime=session.run(['logits'],{'trajectory':x,'device_id':domain})[0]
            if pt.shape!=runtime.shape or not np.isfinite(runtime).all(): raise ValueError('Invalid ONNX output')
            maximum=max(maximum,float(np.abs(pt.astype(float)-runtime.astype(float)).max()))
    parity=dict(tensors=100,supplied_validation_tensors=real_count,max_logit_error=maximum,
                tolerance=1e-5,passed=maximum<1e-5,
                onnx_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    output.with_suffix('.parity.json').write_text(json.dumps(parity,indent=2))
    if not parity['passed']: raise AssertionError(f'Parity failed: {maximum}')
    return parity


def fit_calibration(model,metadata,calibration):
    subjects=np.asarray(calibration['subject']); known=set(metadata['validation_subjects'])
    train=set(metadata['training_subjects'])|set(metadata.get('ssl_subjects',[]))
    if set(subjects)&train or not set(subjects)<=known: raise ValueError('Calibration parent leakage')
    x=np.asarray(calibration['X'],dtype=np.float32); domain=np.asarray(calibration['domain'],dtype=np.int64)
    if len(x)!=len(subjects) or x.shape[1:]!=(model.config['in_channels'],512): raise ValueError('Bad calibration tensors')
    if 'synthetic' in calibration and np.asarray(calibration['synthetic']).any(): raise ValueError('Synthetic calibration prohibited')
    logits=[]
    with torch.no_grad():
        for i in range(0,len(x),4):
            logits.extend(model(torch.from_numpy(x[i:i+4]),torch.from_numpy(domain[i:i+4]))['pd_logits'].numpy())
    _,y,p=aggregate(calibration['label'],expit(logits),subjects)
    if len(np.unique(y))!=2: raise ValueError('Need both diagnoses for calibration')
    iso=IsotonicRegression(out_of_bounds='clip').fit(p,y)
    return dict(method='isotonic_after_subject_mean_probability',x=iso.X_thresholds_.tolist(),
                y=iso.y_thresholds_.tolist(),validation_subject_count=len(y),
                warning='Small validation-set isotonic calibration may overfit; not independently validated')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints',nargs='+',required=True)
    parser.add_argument('--calibration',required=True,help='Private NPZ from selected member validation only')
    parser.add_argument('--output',default='models/v6_exports')
    args=parser.parse_args(); output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    loaded=[load_model(path) for path in args.checkpoints]
    best=int(np.argmax([c['val_auc'] for _,c in loaded]))
    model,metadata=loaded[best]
    with np.load(args.calibration,allow_pickle=False) as z: calibration={k:z[k] for k in z.files}
    mapping=fit_calibration(model,metadata,calibration)
    files=[]
    for i,path in enumerate(args.checkpoints):
        real=(calibration['X'],calibration['domain']) if i==best else None
        filename=output/f'tremorformer_fold_{i}.onnx'; parity=export_member(path,filename,real)
        files.append(dict(file=filename.name,size_bytes=filename.stat().st_size,parity=parity))
    config=dict(version='v6-research-1',feature_names=FEATURES,shape=['batch',len(FEATURES),512],
       canonical_rate_hz=100,normalization='per_window_population_zscore_epsilon_1e-6',
       binary_channels_unstandardized=BINARY_CHANNELS,mean=[0.]*len(FEATURES),scale=[1.]*len(FEATURES),
       statistics_semantics='identity AFTER per-window normalization; not fitted global raw statistics',
       device_lookup={name:i for i,name in enumerate(DOMAINS)},device_filter_cutoff_hz=CUTOFF,
       classification_threshold=.5,output_semantics='raw_logits_apply_sigmoid_then_subject_mean',
       calibration=mapping,calibrated_member=best,models=files,
       ensemble_calibration='NOT_AVAILABLE: requires separate all-member-unseen calibration cohort')
    (output/'scaler_config_v6.json').write_text(json.dumps(config,indent=2)+'\n')

if __name__=='__main__': main()
