"""Fold-local augmentation. Raw-coordinate transforms precede feature extraction.

Tremor injection is opt-in synthetic stress training, NOT a verified diagnosis.
No donors or recipients from validation/test may enter the augmenter.
"""
from dataclasses import replace
import numpy as np
from scipy.interpolate import CubicSpline

DEFAULT_PROBABILITIES=dict(time_warp=.2,magnitude=.2,jitter=.2,slice=.2,rotation=.3,
                          elastic=.2,pressure_dropout=.2,device_simulation=.2,
                          tremor_injection=0.,mixup=.2,cutmix=.2)


def dtw_map(values,rng,anchors=64):
    """Monotone random stretching from DTW of speed vs a perturbed speed template.

    Align at <=64 anchors to bound O(K^2) memory/time; linear map full points.
    No diagnostic labels or other subjects enter this time transform.
    """
    n=len(values); k=min(anchors,n)
    a=np.interp(np.linspace(0,n-1,k),np.arange(n),values)
    b=a+rng.normal(0,max(np.std(a),1e-4)*.15,k)
    cost=np.full((k+1,k+1),np.inf); cost[0,0]=0
    for i in range(1,k+1):
        for j in range(1,k+1):
            cost[i,j]=(a[i-1]-b[j-1])**2+min(cost[i-1,j],cost[i,j-1],cost[i-1,j-1])
    i=j=k; pairs=[]
    while i and j:
        pairs.append((i-1,j-1))
        move=np.argmin([cost[i-1,j],cost[i,j-1],cost[i-1,j-1]])
        if move==0: i-=1
        elif move==1: j-=1
        else: i-=1; j-=1
    mapping=np.array([np.mean([i for i,j in pairs if j==q]) for q in range(k)])
    mapping[0]=0; mapping[-1]=k-1
    return np.interp(np.linspace(0,k-1,n),np.arange(k),mapping)*(n-1)/(k-1)


def tremor_component(trial):
    """4-6 Hz donor component, derived only from a training-fold real PD path."""
    a=trial.samples; t=a[:,2]-a[0,2]
    grid=np.arange(0,t[-1],.01)
    if len(grid)<100: raise ValueError('Donor too short')
    xy=np.stack([np.interp(grid,t,a[:,i]) for i in (0,1)])
    xy-=xy.mean(-1,keepdims=True)
    scale=np.percentile(np.hypot(*xy),95)
    if scale<=1e-8: raise ValueError('Stationary donor')
    xy/=scale
    f=np.fft.rfftfreq(len(grid),.01); fft=np.fft.rfft(xy,axis=-1)
    fft[:,(f<4)|(f>6)]=0
    return np.fft.irfft(fft,n=len(grid),axis=-1)

class Augmenter:
    def __init__(self,allowed_subjects,donors=(),probabilities=None,seed=42):
        self.allowed=set(allowed_subjects); self.rng=np.random.default_rng(seed)
        self.p=dict(DEFAULT_PROBABILITIES); self.p.update(probabilities or {})
        if set(self.p)!=set(DEFAULT_PROBABILITIES) or any(not 0<=v<=1 for v in self.p.values()):
            raise ValueError('Bad augmentation probabilities')
        self.donors=[]
        for tr in donors:
            if tr.subject_id not in self.allowed or tr.label!=1 or tr.synthetic or tr.domain!='wacom':
                raise ValueError('Tremor donors must be real training-fold Wacom PD')
            self.donors.append((tr.subject_id,tremor_component(tr)))
        if self.p['tremor_injection'] and not self.donors:
            raise ValueError('Injection enabled without permitted real PD donors')

    def __call__(self,trial):
        if trial.subject_id not in self.allowed: raise ValueError('Non-training recipient')
        r=self.rng; a=trial.samples.copy(); n=len(a); t=a[:,2]-a[0,2]
        spread=max(np.percentile(np.hypot(a[:,0]-np.median(a[:,0]),a[:,1]-np.median(a[:,1])),95),1e-6)
        hit=lambda name:r.random()<self.p[name]
        if hit('time_warp'):
            speed=np.hypot(np.gradient(a[:,0]),np.gradient(a[:,1]))
            mapping=dtw_map(speed,r)
            for c in (0,1,3,4):
                valid=np.isfinite(a[:,c])
                if valid.sum()>1: a[:,c]=np.interp(mapping,np.arange(n)[valid],a[valid,c])
        if hit('magnitude'):
            curve=CubicSpline(np.linspace(0,1,6),r.normal(1,.05,6))(np.linspace(0,1,n))
            center=np.median(a[:,:2],axis=0); a[:,:2]=(a[:,:2]-center)*curve[:,None]+center
        if hit('jitter'): a[:,:2]+=r.normal(0,.002*spread,(n,2))
        if hit('slice'):
            size=max(24,int(n*r.uniform(.85,1))); start=int(r.integers(0,n-size+1))
            for c in (0,1,3,4):
                valid=np.isfinite(a[start:start+size,c])
                if valid.sum()>1:
                    a[:,c]=np.interp(np.linspace(0,size-1,n),np.arange(size)[valid],a[start:start+size,c][valid])
        if hit('rotation'):
            angle=np.deg2rad(r.uniform(-15,15)); matrix=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
            center=np.median(a[:,:2],axis=0); a[:,:2]=(a[:,:2]-center)@matrix.T+center
        if hit('elastic'):
            a[:,0]+=.005*spread*np.sin(2*np.pi*r.uniform(.2,1)*t+r.uniform(0,6.28))
            a[:,1]+=.005*spread*np.sin(2*np.pi*r.uniform(.2,1)*t+r.uniform(0,6.28))
        if hit('pressure_dropout'): a[:,3]=np.nan
        domain=trial.domain
        if trial.domain=='wacom' and hit('device_simulation'):
            grid=np.arange(0,t[-1],1/60)
            a=np.column_stack([np.interp(grid,t,a[:,c]) if np.isfinite(a[:,c]).all() else np.full(len(grid),np.nan) for c in range(5)])
            a[:,2]=grid+trial.samples[0,2]
            a[:,:2]+=r.normal(0,.004*spread,(len(a),2)); a[:,3:]=np.nan
            domain='touch'
        label=trial.label; synthetic=trial.synthetic; provenance=[]
        if label==0 and domain in ('touch','mouse') and hit('tremor_injection'):
            donor_id,component=self.donors[int(r.integers(len(self.donors)))]
            # Repeat donor time samples at 100 Hz on recipient seconds (preserve Hz).
            donor_t=np.arange(component.shape[-1])*.01
            local=np.mod(a[:,2]-a[0,2],donor_t[-1])
            strength=r.uniform(.1,.4)
            for c in (0,1): a[:,c]+=spread*strength*np.interp(local,donor_t,component[c])
            label=1; synthetic=True; provenance=[trial.subject_id,donor_id]
        return replace(trial,samples=a,domain=domain,label=label,synthetic=synthetic),provenance


def mix_batch(batch,rng,probabilities=None,alpha=.4):
    """Same-class/domain feature-space Mixup and 1D CutMix; mask mixed auxiliaries.

    Donor tensors are detached originals to avoid cascading mix dependencies.
    Derivative channels aren't recomputed: these are latent regularizers, not new
    physical trajectories or clinical evidence. Evaluation never uses this function.
    """
    import torch
    p=dict(DEFAULT_PROBABILITIES); p.update(probabilities or {})
    x=batch['x'].clone(); originals=x.clone(); changed=torch.zeros(len(x),dtype=torch.bool,device=x.device)
    for i in range(len(x)):
        eligible=torch.where((batch['label']==batch['label'][i]) & (batch['domain']==batch['domain'][i]))[0].tolist()
        eligible=[j for j in eligible if j!=i]
        if not eligible: continue
        j=int(rng.choice(eligible))
        if rng.random()<p['mixup']:
            w=float(rng.beta(alpha,alpha)); x[i]=w*x[i]+(1-w)*originals[j]; changed[i]=True
        if rng.random()<p['cutmix']:
            length=int(rng.integers(16,257)); start=int(rng.integers(0,513-length))
            x[i,:,start:start+length]=originals[j,:,start:start+length]; changed[i]=True
    result=dict(batch,x=x)
    for key in ('task','age'): result[key]=batch[key].masked_fill(changed,-1)
    result['updrs']=batch['updrs'].masked_fill(changed,float('nan'))
    return result
