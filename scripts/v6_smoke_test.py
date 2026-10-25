"""CPU correctness only: synthetic paths, tiny five-fold loop, full-width export.

Produces no clinical performance claim or production model. All weights/arrays
remain in an ignored/private output directory. Run with --output /tmp/v6_smoke.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from src.data_loaders.schema import Trajectory
from src.models.v6_tremorformer import TremorFormer
from src.training.v6_train import run,TrajectoryDataset,predict
from src.inference.v6_ensemble_inference import Ensemble,load_model
from src.evaluation.v6_eval import report
from scripts.v6_export_onnx import export_member,fit_calibration


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',default='/tmp/v6_smoke')
    parser.add_argument('--full-export',action='store_true'); args=parser.parse_args()
    torch.set_num_threads(2); torch.manual_seed(42); np.random.seed(42)
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    records=[]
    for i in range(14):
        t=np.arange(700)/100
        x=np.cos(2*t)+.01*np.sin(2*np.pi*(4+i%3)*t); y=np.sin(2*t)
        samples=np.column_stack([x,y,t,.5+.1*np.cos(2*t),np.full_like(t,np.nan)])
        sid=f'dev{i:02}' if i<10 else f'holdout{i:02}'
        records.append(Trajectory(sid,'session','spiral',['wacom','touch','mouse'][i%3],samples,i%2).validate())
    partition=dict(development_subjects=[tr.subject_id for tr in records[:10]],holdout_subjects=[tr.subject_id for tr in records[10:]])
    config=dict(d_model=32,blocks=1,heads=4,expansion=2,dropout=0.)
    manifest=run(records,partition,output/'folds',epochs=1,ssl_epochs=1,batch_size=4,
                 device='cpu',model_config=config)
    paths=[v['best_checkpoint'] for v in manifest]
    ensemble=Ensemble(paths)
    pred=dict(probability=[],label=[],subject=[],domain=[],synthetic=[])
    uncertainty=[]
    for tr in records[10:]:
        result=ensemble.predict_trial(tr)
        p=result['window_probability']; uncertainty.append(result['uncertainty'])
        pred['probability'].extend(p); pred['label'].extend([tr.label]*len(p))
        pred['subject'].extend([tr.subject_id]*len(p)); pred['domain'].extend([tr.domain]*len(p)); pred['synthetic'].extend([False]*len(p))
    pred={k:np.asarray(v) for k,v in pred.items()}
    report(pred,partition['holdout_subjects'],output/'SYNTHETIC_ONLY_evaluation.md',replicates=20,plot_dir=output/'plots')
    model,ckpt=load_model(paths[0])
    val=[tr for tr in records if tr.subject_id in ckpt['validation_subjects']]
    ds=TrajectoryDataset(val,ckpt['validation_subjects'])
    items=[ds[i] for i in range(len(ds))]
    calibration=dict(X=np.stack([b['x'].numpy() for b in items]),domain=np.array([int(b['domain']) for b in items]),
                     subject=np.array([b['subject'] for b in items]),label=np.array([float(b['label']) for b in items]))
    np.savez_compressed(output/'synthetic_validation.npz',**calibration)
    mapping=fit_calibration(model,ckpt,calibration)
    small_parity=export_member(paths[0],output/'tiny_SMOKE_ONLY.onnx',
                              (calibration['X'],calibration['domain']))
    summary=dict(status='SYNTHETIC_CORRECTNESS_ONLY',cuda_available=torch.cuda.is_available(),
                 fold_count=len(manifest),epochs_per_fold=1,ssl_epochs_per_fold=1,
                 synthetic_holdout_trials=4,ensemble_predictions_per_trial=50,
                 tiny_export_parity=small_parity,clinical_metrics='NOT_MEASURED',
                 calibration_map_points=len(mapping['x']))
    if args.full_export:
        full=TremorFormer().eval()
        checkpoint=output/'full_RANDOM_INITIALIZATION.pth'
        torch.save(dict(model_state_dict=full.state_dict(),model_config=full.config),checkpoint)
        summary['full_parameters']=sum(p.numel() for p in full.parameters())
        summary['full_random_export_parity']=export_member(checkpoint,output/'full_RANDOM_INITIALIZATION.onnx')
    (output/'correctness_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
