"""Anchored, TWO-PARENT synthetic tremor recipes; not clinical Mobile-PD data.

Production mode needs real recorded UCI-PD donors AND recorded touch/mouse HC
recipients. Mock mode is explicit, tagged, and rejected by production training.
Default 5001/domain =10002 variants, NOT10002 independent subjects.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.signal import butter,sosfiltfilt
from src.data_loaders.schema import Trajectory


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(trial):
    trial.validate(); a=trial.samples; t=a[:,2]-a[0,2]
    grid=np.arange(int(np.floor(t[-1]*100))+1)/100
    if len(grid)<512: raise ValueError('Generator requires >=5.11s real trajectories')
    result=np.column_stack([np.interp(grid,t,a[:,c]) if np.isfinite(a[:,c]).all()
                            else np.full(len(grid),np.nan) for c in range(5)])
    result[:,2]=grid
    return result


def extract_tremor(trial):
    """4–8Hz component is tremor-associated, NOT proven pure pathology."""
    a=canonical(trial); xy=a[:,:2]-np.median(a[:,:2],axis=0)
    radius=np.percentile(np.hypot(xy[:,0],xy[:,1]),95)
    if radius<=1e-8: raise ValueError('Stationary donor')
    return sosfiltfilt(butter(4,[4,8],btype='bandpass',fs=100,output='sos'),xy/radius,axis=0)


def overlay(baseline,tremor,rng,target_domain):
    a=baseline.copy(); n=len(a); t=a[:,2]
    center=np.median(a[:,:2],axis=0)
    radius=np.percentile(np.hypot(*(a[:,:2]-center).T),95)
    if radius<=1e-8: raise ValueError('Stationary recipient')
    smooth=sosfiltfilt(butter(4,3,fs=100,output='sos'),a[:,:2],axis=0)
    # Monotone slower progress near a random micro-hesitation; preserve total time.
    duration=max(t[-1],.01); location=rng.uniform(.2,.8)*duration
    progress_rate=1-rng.uniform(.05,.25)*np.exp(-((t-location)/.15)**2)
    progress=np.cumsum(progress_rate); progress=(progress-progress[0])/(progress[-1]-progress[0])*t[-1]
    smooth=np.column_stack([np.interp(progress,t,smooth[:,c]) for c in (0,1)])
    phase=np.cumsum(1+rng.uniform(.01,.05)*np.sin(2*np.pi*.3*t+rng.uniform(0,6.28)))
    phase=(phase-phase[0])*.01
    donor_t=np.arange(len(tremor))*.01
    # Mirror extension avoids repeating a discontinuous FFT/bandpass boundary.
    period=2*donor_t[-1]; q=np.mod(phase,period); q=np.where(q>donor_t[-1],period-q,q)
    amplitude=rng.uniform(.3,1.2)
    envelope=1+.1*np.sin(2*np.pi*.2*t+rng.uniform(0,6.28))
    injected=np.column_stack([np.interp(q,donor_t,tremor[:,c]) for c in (0,1)])
    a[:,:2]=smooth+radius*amplitude*envelope[:,None]*injected
    # Actual target domains are simulated acquisition, not recorded phone PD.
    rate=60 if target_domain=='touch' else 100
    grid=np.arange(0,t[-1]+1e-9,1/rate)
    xy=np.column_stack([np.interp(grid,t,a[:,c]) for c in (0,1)])
    xy+=rng.normal(0,radius*(.0015 if rate==60 else .0005),xy.shape)
    out=np.column_stack([xy,grid,np.full((len(grid),2),np.nan)])
    return out,dict(amplitude_multiplier=float(amplitude),hesitation_center_seconds=float(location),
                    target_hz=rate,source_band_hz=[4,8],frequency_modulation_max_fraction=.05)


def generate(records,allowed_subjects,output='data/processed/v6_synthetic_10k',
             manifest='data/processed/v6_expanded_10k_manifest.csv',per_domain=5001,seed=42,mock=False):
    allowed=set(allowed_subjects)
    if not allowed or per_domain<1: raise ValueError('Explicit permitted parent pool/count required')
    if Path(manifest).exists(): raise FileExistsError('Refuse overwrite of an existing synthetic manifest')
    donors=[tr for tr in records if tr.subject_id in allowed and tr.label==1 and tr.domain=='wacom' and not tr.synthetic]
    recipients=[tr for tr in records if tr.subject_id in allowed and tr.label==0 and tr.domain in ('touch','mouse') and not tr.synthetic]
    if not donors or not recipients: raise ValueError('Missing recorded Wacom-PD donors or touch/mouse HC baselines; no data invented')
    if not mock and any(tr.source!='uci395' for tr in donors): raise ValueError('Production donor must be reviewed UCI395 source')
    if not mock and any(tr.source.startswith('mock') for tr in records): raise ValueError('Mock sources forbidden in production generation')
    prepared=[(d,extract_tremor(d)) for d in donors]
    baselines=[(r,canonical(r)) for r in recipients]
    root=Path(output); root.mkdir(parents=True,exist_ok=True)
    path=Path(manifest); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix('.partial.csv')
    rng=np.random.default_rng(seed); count=0
    fields=['sample_id','subject_id','donor_subject_id','recipient_session_id','donor_session_id',
            'domain','baseline_domain','task_id','label','synthetic','label_status','evidence_kind',
            'point_count','duration_seconds','path','sha256','seed','parameters_json']
    try:
        with tmp.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
            for domain in ('touch','mouse'):
                eligible=[b for b in baselines if b[0].domain==domain] or baselines
                for i in range(per_domain):
                    donor,component=prepared[int(rng.integers(len(prepared)))]
                    recipient,base=eligible[int(rng.integers(len(eligible)))]
                    samples,parameters=overlay(base,component,rng,domain)
                    sid=f'{domain}_s{seed}_{i:06d}'; target=root/(sid+'.npz')
                    if target.exists(): raise FileExistsError('Refuse synthetic file overwrite')
                    np.savez_compressed(target,samples=samples)
                    writer.writerow(dict(sample_id=sid,subject_id=recipient.subject_id,donor_subject_id=donor.subject_id,
                       recipient_session_id=recipient.session_id,donor_session_id=donor.session_id,
                       domain=domain,baseline_domain=recipient.domain,task_id=recipient.task_id,label=1,synthetic=True,
                       label_status='unvalidated_synthetic_pd_hypothesis',evidence_kind='MOCK_SOURCE' if mock else 'RECORDED_SOURCE_SYNTHESIS',
                       point_count=len(samples),duration_seconds=float(samples[-1,2]),path=str(target.resolve()),
                       sha256=digest(target),seed=seed,parameters_json=json.dumps(parameters,sort_keys=True)))
                    count+=1
        tmp.replace(path)
    finally: tmp.unlink(missing_ok=True)
    return dict(trajectories=count,per_domain=per_domain,parent_subjects=len({t.subject_id for t in donors+recipients}),
                manifest=str(path),evidence='MOCK_SOURCE' if mock else 'RECORDED_SOURCE_SYNTHESIS',
                caveat='Variant count is not independent participants or real mobile PD diagnoses')


def eligible_rows(manifest,train_subjects,allow_mock=False):
    allowed=set(train_subjects); rows=[]
    with Path(manifest).open(newline='') as f:
        for row in csv.DictReader(f):
            if row['evidence_kind']=='MOCK_SOURCE' and not allow_mock: raise ValueError('Mock manifest cannot train production')
            if row['subject_id'] in allowed and row['donor_subject_id'] in allowed:
                rows.append(row)
    return rows


def materialize(row):
    path=Path(row['path'])
    if digest(path)!=row['sha256']: raise ValueError('Synthetic file checksum mismatch')
    with np.load(path,allow_pickle=False) as z: samples=z['samples']
    return Trajectory(row['subject_id'],row['sample_id'],row['task_id'],row['domain'],samples,
                      label=1,synthetic=True,source='tremor_synthesis').validate()


def mock_sources():
    """Test-only formula paths. Never label them real or export clinical metrics."""
    result=[]
    for sid,label,domain in [('mock_donor',1,'wacom'),('mock_touch',0,'touch'),('mock_mouse',0,'mouse')]:
        t=np.arange(700)/100
        x=np.cos(t*2)+(.02*np.sin(2*np.pi*5*t) if label else 0)
        a=np.column_stack([x,np.sin(t*2),t,np.full((len(t),2),np.nan)])
        result.append(Trajectory(sid,'mock_session','spiral',domain,a,label,source='mock_fixture').validate())
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--records'); p.add_argument('--partition')
    p.add_argument('--output',default='data/processed/v6_synthetic_10k'); p.add_argument('--manifest',default='data/processed/v6_expanded_10k_manifest.csv')
    p.add_argument('--per-domain',type=int,default=5001); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--mock-fixtures',action='store_true',help='TEST ONLY; downstream production loader rejects this evidence')
    a=p.parse_args()
    if a.mock_fixtures:
        if not str(Path(a.output).resolve()).startswith('/tmp/') or not str(Path(a.manifest).resolve()).startswith('/tmp/'):
            p.error('Mock generation must stay under /tmp, not production data paths')
        records=mock_sources(); allowed={r.subject_id for r in records}
    else:
        if not a.records or not a.partition: p.error('Authorized records and locked partition required')
        records=[Trajectory.from_dict(r) for r in json.loads(Path(a.records).read_text())]
        part=json.loads(Path(a.partition).read_text()); allowed=set(part['development_subjects'])
        if allowed & set(part['holdout_subjects']): raise ValueError('Partition overlap')
    print(json.dumps(generate(records,allowed,a.output,a.manifest,a.per_domain,a.seed,a.mock_fixtures),indent=2))

if __name__=='__main__': main()
