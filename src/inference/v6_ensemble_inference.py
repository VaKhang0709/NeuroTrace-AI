"""Raw-trajectory 10-TTA x 5-fold inference, never trained in this module."""
from dataclasses import replace
import numpy as np
import torch
from src.models.v6_tremorformer import TremorFormer
from src.preprocessing.v6_pipeline import transform


def load_model(path,device='cpu'):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    model=TremorFormer(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    return model.to(device).eval(),checkpoint


def variants(trial,count=10):
    if count!=10: raise ValueError('Reference protocol fixes ten TTA variants')
    # First is identity; angles/time-shifts fixed before test inspection.
    for angle,offset in [(0,0),(-15,0),(15,0),(-10,.02),(10,-.02),
                         (-5,.04),(5,-.04),(0,.02),(0,-.02),(0,.04)]:
        a=trial.samples.copy(); t=a[:,2]
        query=np.clip(t+offset,t[0],t[-1])
        for c in (0,1,3,4):
            finite=np.isfinite(a[:,c])
            if finite.sum()>1:
                v=np.unwrap(a[finite,c]) if c==4 else a[finite,c]
                a[:,c]=np.interp(query,t[finite],v)
        theta=np.deg2rad(angle); matrix=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        center=np.median(a[:,:2],axis=0); a[:,:2]=(a[:,:2]-center)@matrix.T+center
        yield replace(trial,samples=a)

class Ensemble:
    def __init__(self,paths,device='cpu',require_five=True):
        if require_five and len(paths)!=5: raise ValueError('Need exactly five fold models')
        loaded=[load_model(p,device) for p in paths]
        self.models=[m for m,_ in loaded]; self.checkpoints=[c for _,c in loaded]; self.device=device

    @torch.no_grad()
    def predict_trial(self,trial,require_unseen=True):
        if require_unseen:
            for c in self.checkpoints:
                known=set(c.get('training_subjects',[]))|set(c.get('validation_subjects',[]))|set(c.get('ssl_subjects',[]))
                if trial.subject_id in known: raise ValueError('Ensemble test parent appeared in development')
                if not c.get('training_subjects') or not c.get('validation_subjects'):
                    raise ValueError('Missing checkpoint provenance')
        all_prob=[]; identity_features=[]
        from src.data_loaders.schema import DOMAINS
        for variant_index,augmented in enumerate(variants(trial)):
            x,_=transform(augmented); x=torch.from_numpy(x).to(self.device)
            domain=torch.full((len(x),),DOMAINS.index(trial.domain),device=self.device,dtype=torch.long)
            for model in self.models:
                # Bound inference memory, even for long private drawings.
                chunks=[]; features=[]
                for i in range(0,len(x),4):
                    out=model(x[i:i+4],domain[i:i+4])
                    chunks.append(torch.sigmoid(out['pd_logits']).cpu().numpy())
                    if variant_index==0: features.append(out['features'].cpu().numpy())
                all_prob.append(np.concatenate(chunks))
                if variant_index==0: identity_features.append(np.concatenate(features))
        values=np.stack(all_prob)  # (50, windows)
        return dict(window_probability=values.mean(0),window_uncertainty=values.std(0),
                    trial_probability=float(values.mean()),
                    uncertainty=float(values.mean(1).std()),predictions=int(values.shape[0]),
                    features=np.concatenate(identity_features,axis=1))


def main():
    import argparse,json
    from pathlib import Path
    from src.data_loaders.schema import Trajectory
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records',required=True); p.add_argument('--partition',required=True)
    p.add_argument('--checkpoints',nargs=5,required=True); p.add_argument('--output',default='reports/private/v6_predictions.npz')
    a=p.parse_args(); partition=json.loads(Path(a.partition).read_text())
    allowed=set(partition['holdout_subjects'])
    if allowed & (set(partition['development_subjects'])|set(partition.get('unlabeled_subjects',[]))):
        raise ValueError('Partition overlap')
    ensemble=Ensemble(a.checkpoints,device='cuda' if torch.cuda.is_available() else 'cpu')
    collected={k:[] for k in ['probability','label','subject','domain','synthetic','features','uncertainty']}
    for raw in json.loads(Path(a.records).read_text()):
        tr=Trajectory.from_dict(raw)
        if tr.subject_id not in allowed: continue
        if tr.synthetic or tr.label not in (0,1): raise ValueError('Clinical holdout must be real, labeled')
        result=ensemble.predict_trial(tr); n=len(result['window_probability'])
        collected['probability'].extend(result['window_probability']); collected['features'].extend(result['features'])
        collected['uncertainty'].extend(result['window_uncertainty'])
        for key,value in [('label',tr.label),('subject',tr.subject_id),('domain',tr.domain),('synthetic',False)]:
            collected[key].extend([value]*n)
    if set(collected['subject'])!=allowed:
        raise ValueError('Missing locked holdout parents; no silent exclusions')
    output=Path(a.output); output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,**{k:np.asarray(v) for k,v in collected.items()})

if __name__=='__main__': main()
