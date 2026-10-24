"""Subject-primary evaluation; no test-driven deployment threshold selection.

Private NPZ: probability,label,subject,domain, optional features, synthetic.
Reports never contain subject IDs. Frozen-feature probe is diagnostic, not a
claim that all neural representations are domain-invariant.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import (roc_auc_score,average_precision_score,roc_curve,
                             confusion_matrix)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold


def aggregate(y,p,subjects):
    y=np.asarray(y); p=np.asarray(p); subjects=np.asarray(subjects)
    ids=np.unique(subjects); yy=[]; pp=[]
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)): raise ValueError('Invalid probabilities')
    if not set(np.unique(y))<=set((0,1)): raise ValueError('Clinical labels must be known')
    for sid in ids:
        mask=subjects==sid; labels=np.unique(y[mask])
        if len(labels)!=1: raise ValueError('Conflicting subject labels')
        yy.append(labels[0]); pp.append(np.mean(p[mask]))
    return ids,np.array(yy,dtype=int),np.array(pp)


def metrics(y,p):
    y=np.asarray(y); p=np.asarray(p)
    tn,fp,fn,tp=confusion_matrix(y,p>=.5,labels=[0,1]).ravel()
    both=len(np.unique(y))==2
    sens95=None
    if both:
        fpr,tpr,_=roc_curve(y,p); sens95=float(tpr[fpr<=.05].max())
    ece=0.
    for i in range(10):
        mask=(p>=i/10)&(p<(i+1)/10 if i<9 else p<=1)
        if mask.any(): ece+=mask.mean()*abs(y[mask].mean()-p[mask].mean())
    return dict(roc_auc=float(roc_auc_score(y,p)) if both else None,
       pr_auc=float(average_precision_score(y,p)) if both else None,
       sensitivity=float(tp/(tp+fn)) if tp+fn else None,
       specificity=float(tn/(tn+fp)) if tn+fp else None,
       sensitivity_at_95_specificity=sens95,ece=float(ece),
       tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))


def bootstrap(y,p,n=1000,seed=42):
    """Input must already have one row/subject; resample subjects with replacement."""
    if n<1: raise ValueError('Bootstrap replicates must be positive')
    rng=np.random.default_rng(seed); original=metrics(y,p); draws={k:[] for k in original}
    for _ in range(n):
        index=rng.integers(0,len(y),len(y)); values=metrics(y[index],p[index])
        for k,v in values.items():
            if v is not None: draws[k].append(v)
    return {k:dict(value=v,ci95=np.percentile(draws[k],[2.5,97.5]).tolist() if draws[k] else None,
                   valid_replicates=len(draws[k])) for k,v in original.items()}


def device_probe(features,domain,subjects):
    features=np.asarray(features); domain=np.asarray(domain); subjects=np.asarray(subjects)
    classes=np.unique(domain)
    if len(classes)<2 or len(np.unique(subjects))<5:
        return dict(status='INCONCLUSIVE',reason='Need multiple devices and >=5 held-out parents')
    aucs=[]
    for tr,va in GroupKFold(5).split(features,domain,subjects):
        assert not set(subjects[tr])&set(subjects[va])
        if not np.array_equal(np.unique(domain[tr]),classes) or not np.array_equal(np.unique(domain[va]),classes):
            return dict(status='INCONCLUSIVE',reason='A probe fold lacks at least one domain')
        estimator=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced'))
        estimator.fit(features[tr],domain[tr]); prob=estimator.predict_proba(features[va])
        auc=roc_auc_score(domain[va],prob[:,1]) if len(classes)==2 else roc_auc_score(domain[va],prob,labels=classes,multi_class='ovr',average='macro')
        aucs.append(float(auc))
    return dict(status='MEASURED',mean_auc=float(np.mean(aucs)),fold_aucs=aucs,
                gate_passed=bool(np.mean(aucs)<.70),ceiling=.70)


def paired_mcnemar(current,baseline):
    ids,y,p=current; other,yy,pp=baseline
    if not np.array_equal(ids,other) or not np.array_equal(y,yy):
        raise ValueError('V5 comparison requires identical paired held-out subjects and diagnoses')
    a=(p>=.5)==y; b=(pp>=.5)==y
    n01=int((~a & b).sum()); n10=int((a & ~b).sum())
    return dict(v6_wrong_v5_right=n01,v6_right_v5_wrong=n10,
                exact_p=float(binomtest(n01,n01+n10,.5).pvalue) if n01+n10 else 1.)


def report(prediction,holdout_subjects,output='reports/v6_final_evaluation.md',
           v5=None,replicates=1000,plot_dir='reports/private/v6_plots'):
    required=('probability','label','subject','domain')
    if any(k not in prediction for k in required): raise ValueError('Missing prediction columns')
    if not len(prediction['subject']): raise ValueError('No predictions')
    if not set(prediction['subject'])<=set(holdout_subjects): raise ValueError('Non-holdout parent in clinical report')
    if 'synthetic' in prediction and np.asarray(prediction['synthetic']).any():
        raise ValueError('Synthetic samples forbidden in clinical evaluation')
    result={}; overall=None
    for domain in ['all']+list(np.unique(prediction['domain'])):
        mask=np.ones(len(prediction['subject']),dtype=bool) if domain=='all' else prediction['domain']==domain
        group=aggregate(prediction['label'][mask],prediction['probability'][mask],prediction['subject'][mask])
        if domain=='all': overall=group
        result[str(domain)]=dict(subjects=len(group[0]),metrics=bootstrap(group[1],group[2],replicates))
    probe=dict(status='INCONCLUSIVE',reason='No held-out encoder features supplied')
    if 'features' in prediction:
        # One feature vector per subject-domain; HC-only isolates device from label.
        keys=sorted(set(zip(prediction['subject'],prediction['domain'])))
        feat=[]; dom=[]; groups=[]
        for sid,d in keys:
            mask=(prediction['subject']==sid)&(prediction['domain']==d)
            if np.unique(prediction['label'][mask]).tolist()==[0]:
                feat.append(prediction['features'][mask].mean(0)); dom.append(d); groups.append(sid)
        probe=device_probe(feat,dom,groups) if feat else dict(status='INCONCLUSIVE',reason='No HC features')
    comparison=None
    if v5 is not None:
        baseline=aggregate(v5['label'],v5['probability'],v5['subject'])
        comparison=paired_mcnemar(overall,baseline)
    plot_dir=Path(plot_dir); plot_dir.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for d in ['all']+list(np.unique(prediction['domain'])):
        mask=np.ones(len(prediction['subject']),bool) if d=='all' else prediction['domain']==d
        _,y,p=aggregate(prediction['label'][mask],prediction['probability'][mask],prediction['subject'][mask])
        if len(set(y))==2:
            fpr,tpr,_=roc_curve(y,p); axes[0].plot(fpr,tpr,label=str(d))
        true,estimated=calibration_curve(y,p,n_bins=10); axes[1].plot(estimated,true,'o-',label=str(d))
    axes[0].set(xlabel='FPR',ylabel='TPR'); axes[1].set(xlabel='Mean prediction',ylabel='Observed fraction')
    for ax in axes: ax.legend(); ax.plot([0,1],[0,1],'--',color='grey')
    fig.tight_layout(); fig.savefig(plot_dir/'roc_calibration.png'); plt.close(fig)
    lines=['# V6 held-out subject evaluation','',
           'Threshold fixed at 0.50. Subject-cluster bootstrap, 1000 replicates by default.',
           'Sensitivity@95% specificity is a descriptive held-out ROC statistic, NOT a selected deployment threshold.',
           'PR-AUC here means average precision. Frozen-feature HC device probe is a separate engineering diagnostic.',
           '| Domain | Subjects | Metric | Estimate | 95% CI | Valid bootstrap replicates |',
           '|---|---:|---|---:|---|---:|']
    for d,values in result.items():
        for metric,stat in values['metrics'].items():
            lines.append(f"| {d} | {values['subjects']} | {metric} | {stat['value']} | {stat['ci95']} | {stat['valid_replicates']} |")
    lines+=['','Device probe: `'+json.dumps(probe)+'`','',
            'Paired V5 McNemar: `'+json.dumps(comparison)+'`' if comparison else 'V5 paired significance: INCONCLUSIVE (no paired held-out predictions).',
            '',f'Aggregate plots saved locally to `{plot_dir}/roc_calibration.png`.',
            'No raw trajectories or subject IDs are included. Not clinical validation.']
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True); output.write_text('\n'.join(lines)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--predictions',required=True)
    p.add_argument('--partition',required=True); p.add_argument('--v5-predictions'); p.add_argument('--output',default='reports/v6_final_evaluation.md')
    a=p.parse_args(); part=json.loads(Path(a.partition).read_text())
    with np.load(a.predictions,allow_pickle=False) as z: pred={k:z[k] for k in z.files}
    v5=None
    if a.v5_predictions:
        with np.load(a.v5_predictions,allow_pickle=False) as z: v5={k:z[k] for k in z.files}
    report(pred,part['holdout_subjects'],a.output,v5)

if __name__=='__main__': main()
