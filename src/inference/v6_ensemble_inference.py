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
        all_prob=[]
        from src.data_loaders.schema import DOMAINS
        for augmented in variants(trial):
            x,_=transform(augmented); x=torch.from_numpy(x).to(self.device)
            domain=torch.full((len(x),),DOMAINS.index(trial.domain),device=self.device,dtype=torch.long)
            for model in self.models:
                # Bound inference memory, even for long private drawings.
                chunks=[]
                for i in range(0,len(x),4):
                    chunks.append(model.predict_proba(x[i:i+4],domain[i:i+4]).cpu().numpy())
                all_prob.append(np.concatenate(chunks))
        values=np.stack(all_prob)  # (50, windows)
        return dict(window_probability=values.mean(0),window_uncertainty=values.std(0),
                    trial_probability=float(values.mean()),
                    uncertainty=float(values.mean(1).std()),predictions=int(values.shape[0]))
