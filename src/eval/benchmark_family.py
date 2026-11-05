"""Locked subject-level REAL holdout + single-thread host CPU window latency.

No train/validation threshold search,no teacher isotonic applied to students.
Writes a report ONLY after all verified tiers/parents finish. No placeholder rows.
"""
import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np
from scipy.special import expit
from src.training.family_data import load_context,windows,sha256
from src.export.export_family import session,check_signature
from src.evaluation.v6_eval import aggregate,metrics


def latency(runtime,x,d,repeats=100,warmup=20):
    if repeats<1 or warmup<1: raise ValueError('Need positive latency repeats/warmup')
    feed=dict(trajectory=np.asarray(x,dtype=np.float32).reshape(1,19,512),device_id=np.asarray(d,dtype=np.int64).reshape(1))
    for _ in range(warmup): runtime.run(['logits'],feed)
    measurements=[]
    for _ in range(repeats):
        start=time.perf_counter_ns(); runtime.run(['logits'],feed); measurements.append((time.perf_counter_ns()-start)/1e6)
    return dict(median_ms=float(np.median(measurements)),p95_ms=float(np.percentile(measurements,95)),
                mean_ms=float(np.mean(measurements)),repeats=repeats,warmup=warmup,threads=1,shape=[1,19,512])


def benchmark(args):
    root=Path(args.family); manifest=json.loads((root/'family_manifest.json').read_text())
    if not manifest.get('accepted'): raise ValueError('All three tier size/signature/parity gates must pass FIRST')
    records,partition,metadata,context=load_context(args.records,args.partition,root/'scaler_config_v6.json',args.teacher_metadata,args.teacher_onnx)
    context['records_sha256']=sha256(args.records)
    if manifest['context']!=context: raise ValueError('Benchmark cohort/provenance/teacher/scaler differs from export')
    if set(manifest['tiers'])!={'nano','base','pro'}: raise ValueError('Require all three tiers, no partial report')
    holdout=windows(records,partition['holdout_subjects']); rows=[]; private={}
    for tier in ('nano','base','pro'):
        info=manifest['tiers'][tier]; path=root/info['file']
        if sha256(path)!=info['sha256']: raise ValueError(f'{tier} artifact checksum changed')
        runtime=session(path); check_signature(runtime); logits=[]
        for i in range(0,len(holdout['X']),8):
            logits.extend(runtime.run(['logits'],dict(trajectory=holdout['X'][i:i+8],device_id=holdout['device_id'][i:i+8]))[0])
        if not np.isfinite(logits).all(): raise ValueError('Nonfinite holdout logits')
        ids,y,p=aggregate(holdout['label'],expit(logits),holdout['subject'])
        if set(ids)!=set(partition['holdout_subjects']): raise ValueError('Missing/extra locked holdout subjects')
        result=metrics(y,p); timed=latency(runtime,holdout['X'][0:1],holdout['device_id'][0:1],args.repeats)
        rows.append(dict(tier=tier,subjects=len(ids),metrics=result,latency=timed,bytes=info['bytes'],sha256=info['sha256']))
        private[tier+'_subject']=ids; private[tier+'_label']=y; private[tier+'_probability']=p
    lines=['# V6 family — locked REAL Wacom holdout','',
        'Subject-mean probabilities; fixed 0.50, no threshold search or shared teacher isotonic. Single-window inference without TTA.',
        f'Partition SHA256: `{context["partition_sha256"]}`; exact original 62 development / 15 holdout, unchanged.',
        '', '| Tier | Subjects | ROC-AUC | PR-AUC | Sensitivity @0.50 | Specificity @0.50 | CPU median ms | CPU p95 ms | Bytes |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        m=row['metrics']; t=row['latency']
        lines.append(f"| {row['tier']} | {row['subjects']} | {m['roc_auc']:.6f} | {m['pr_auc']:.6f} | {m['sensitivity']:.6f} | {m['specificity']:.6f} | {t['median_ms']:.3f} | {t['p95_ms']:.3f} | {row['bytes']} |")
    lines+=['',f'Host: {platform.platform()}, {platform.machine()}, {platform.processor()}. ORT CPU, sequential, intra/inter threads=1; 20warmups; {args.repeats}timed calls.',
        'Latency excludes loading, preprocessing and network. Host CPU is only a proxy, NOT a mobile hardware benchmark.',
        '', '## Artifact SHA-256']
    for row in rows: lines.append(f"- {row['tier']}: `{row['sha256']}`")
    lines+=['','All tiers Wacom-trained. Touchscreen PD sensitivity is NOT clinically established. Fifteen research subjects are not independent clinical deployment validation.',
            'The 0.30–0.70 cascade is a UX/resource heuristic,not a validated abstention or medical-risk boundary. PRO upload requires explicit consent.']
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text('\n'.join(lines)+'\n')
    output.with_suffix('.json').write_text(json.dumps(dict(context=context,results=rows),indent=2,allow_nan=False))
    np.savez_compressed(output.with_suffix('.predictions.npz'),**private)
    print(output); return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('family','records','partition','teacher-metadata','teacher-onnx'): p.add_argument('--'+name,required=True)
    p.add_argument('--output',default='reports/private/v6_family_holdout.md'); p.add_argument('--repeats',type=int,default=100)
    benchmark(p.parse_args())

if __name__=='__main__': main()
