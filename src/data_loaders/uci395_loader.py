"""UCI395 -> canonical [x,y,seconds,pressure,azimuth_radians], never pixels.

Verified mirror README: X;Y;Z;Pressure;GripAngle;Timestamp;Test ID (headerless).
Named six-column X/Y/Pressure/Azimuth/Tilt/Time is also supported ONLY by header.
Timestamp unit is explicit configuration: default milliseconds is an ASSUMPTION
when the source header/README does not declare it. No sampling-rate inference.
Keep LAST per timestamp in acquisition order, then reject remaining reversals.
"""
import argparse
import ast
import csv
import json
import math
import re
from collections import Counter
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
import numpy as np
from .schema import Trajectory

TIME_SCALE={'seconds':1.,'milliseconds':1e-3,'microseconds':1e-6}

@lru_cache(maxsize=1)
def _safe_subject_normalizer():
    # Reuse V5's safe suffix-only helper WITHOUT importing notebook magic, torch,
    # or V4's numeric-block truncation. No production V5 source is edited.
    path=Path(__file__).resolve().parents[1]/'ai_model/scripts/train_pipeline_v5_diagnostic.py'
    tree=ast.parse(path.read_text().split('\n',1)[1])
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='normalise_subject_code']
    if len(nodes)!=1: raise RuntimeError('Expected unique audited V5 normalizer')
    scope={'re':re}; exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
    return scope['normalise_subject_code']


def subject_code(stem):
    # Remove only EXPLICIT task suffixes (optional task ordinal), never generic
    # numeric suffixes: H_P000-0001 and H_P000-0002 must remain distinct.
    stem=re.sub(r'(?i)[_-](spiral|meander|sst|dst|stcp|circle|line)(?:[_-]\d+)?$','',stem)
    return _safe_subject_normalizer()(stem)


def identity(path,root,mapping=None):
    supplied=(mapping or {}).get(path.relative_to(root).as_posix())
    if supplied: return str(supplied['subject_id']),int(supplied['label'])
    code=subject_code(path.stem)
    if not re.fullmatch(r'(?:H_)?[PC][_-]?\d+(?:[-_]\d+)*',code,re.I):
        raise ValueError('unrecognized_subject_id_use_subject_map')
    prefix=re.match(r'(?:H_)?([PC])',code,re.I)[1].upper()
    label=int(prefix=='P')
    parents={p.lower() for p in path.relative_to(root).parts[:-1]}
    if (label and parents & {'control','controls','healthy','hc'}) or (not label and parents & {'parkinson','patients','pd'}):
        raise ValueError('folder_filename_label_conflict')
    return 'uci::'+code.upper(),label


def _token(value): return re.sub(r'[^a-z0-9]','',value.lower())


def _layout(rows,time_unit):
    header=None; body=[]
    aliases={'x':{'x','xpos'},'y':{'y','ypos'},'pressure':{'pressure','p','force'},
             'time':{'time','timestamp','timems','timestampms','timeus','timestampus','times','timestamps'},
             'azimuth':{'azimuth','azimuthdeg','azimuthrad'},'test':{'testid','test'}}
    for row in rows:
        if any(_token(v) in aliases['x'] for v in row) and any(_token(v) in aliases['time'] for v in row):
            if header is not None: continue # repeated header, not numeric samples
            header=row; continue
        body.append(row)
    if header is None:
        if not body or any(len(r)!=7 for r in body): raise ValueError('unsupported_headerless_layout_expected_verified_7_columns')
        return body,{'x':0,'y':1,'pressure':3,'time':5,'test':6},time_unit,'configured_assumption_headerless',7,False
    columns={}
    for key,names in aliases.items():
        for i,name in enumerate(header):
            if _token(name) in names: columns[key]=i; break
    if not {'x','y','pressure','time'}<=set(columns): raise ValueError('incomplete_named_header')
    token=_token(header[columns['time']]); declared=None
    if token.endswith('ms'): declared='milliseconds'
    elif token.endswith('us'): declared='microseconds'
    elif token in ('times','timestamps'): declared='seconds'
    if declared and declared!=time_unit: raise ValueError('configured_time_unit_conflicts_with_header')
    return body,columns,time_unit,'header' if declared else 'configured_assumption_named_header',len(header),header[0]==''


def parse_file(path,subject_id,label,time_unit='milliseconds',azimuth_unit='unknown',audit=None):
    if time_unit not in TIME_SCALE: raise ValueError('Explicit supported timestamp unit required')
    if azimuth_unit not in ('unknown','degrees','radians'): raise ValueError('Unsupported azimuth unit')
    audit=audit if audit is not None else Counter()
    rows=[]
    for row in csv.reader(Path(path).read_text(encoding='utf-8-sig',errors='strict').splitlines(),delimiter=';'):
        row=[v.strip() for v in row]
        while row and not row[-1]: row.pop()
        if not row or row[0].startswith(('#','//')): continue
        # Skip prose safely, not numeric malformed rows or unknown layouts.
        if len(row)==1 and not re.match(r'^[+\-\d.]',row[0]): continue
        rows.append(row)
    body,col,unit,origin,width,leading_blank=_layout(rows,time_unit)
    audit['layout_'+('named_header' if 'headerless' not in origin else 'verified_7_headerless')]+=1
    audit['time_'+origin]+=1
    values=[]
    for row in body:
        indices=col
        if leading_blank and len(row)==width-1:
            indices={k:i-1 for k,i in col.items()}
        elif len(row)!=width:
            audit['malformed_rows_removed']+=1; continue
        try:
            numbers=np.array([float(v) for v in (row[1:] if leading_blank and len(row)==width and row[0]=='' else row)])
            # A blank leading index shifts only the header positions.
            if leading_blank and len(row)==width and row[0]=='': indices={k:i-1 for k,i in col.items()}
            if not np.isfinite(numbers).all(): audit['nonfinite_rows_removed']+=1; continue
            az=float('nan')
            if 'azimuth' in indices:
                if azimuth_unit=='degrees': az=numbers[indices['azimuth']]*np.pi/180
                elif azimuth_unit=='radians': az=numbers[indices['azimuth']]
                else: audit['azimuth_unknown_unit_rows_marked_missing']+=1
            values.append([numbers[indices['x']],numbers[indices['y']],numbers[indices['time']],
                           numbers[indices['pressure']],az,
                           numbers[indices['test']] if 'test' in indices else -1])
        except (ValueError,IndexError): audit['malformed_rows_removed']+=1
    if not values: raise ValueError('no_finite_numeric_samples')
    a=np.asarray(values,dtype=float)
    cuts=np.r_[0,np.flatnonzero(np.diff(a[:,5])!=0)+1,len(a)]
    result=[]
    explicit_task=next((task for task in ('meander','spiral','line','circle') if task in Path(path).stem.lower() or task in [p.lower() for p in Path(path).parts[:-1]]),None)
    for start,end in zip(cuts[:-1],cuts[1:]):
        block=a[start:end]
        last={float(t):i for i,t in enumerate(block[:,2])}
        keep=sorted(last.values()) # ACQUISITION order; do NOT sort by clock.
        audit['duplicate_timestamp_rows_removed']+=len(block)-len(keep)
        block=block[keep]
        if len(block)<24: audit['short_trajectories_skipped']+=1; continue
        t=(block[:,2]-block[0,2])*TIME_SCALE[unit]
        if np.any(np.diff(t)<=0): audit['nonmonotonic_trajectories_skipped']+=1; continue
        if t[-1]<.4: audit['short_duration_trajectories_skipped']+=1; continue
        test=int(block[0,5])
        task=explicit_task or ({0:'spiral',1:'spiral',2:'circular'}.get(test,'unknown'))
        samples=np.column_stack([block[:,0],block[:,1],t,block[:,3],block[:,4]])
        # README does not calibrate x/y to mm/pixels nor GripAngle to azimuth.
        tr=Trajectory(subject_id,subject_id+'::session_unknown',task,'wacom',samples,label,
                      updrs=float('nan'),age_bucket=-1,synthetic=False,source='uci395',coordinate_units='unknown')
        result.append(tr.validate())
    return result


def load(root,time_unit='milliseconds',azimuth_unit='unknown',subject_map=None,audit=None):
    root=Path(root); audit=audit if audit is not None else Counter()
    if not root.is_dir(): raise FileNotFoundError(f'Real UCI directory not found: {root}')
    mapping={}
    if subject_map:
        with Path(subject_map).open(newline='') as f: mapping={r['relative_path']:r for r in csv.DictReader(f)}
    records=[]
    for path in sorted(p for p in root.rglob('*') if p.is_file()):
        if path.suffix.lower()!='.txt': audit['non_txt_files_ignored']+=1; continue
        audit['total_txt_files']+=1
        if path.stem.lower() in ('readme','license','licence'): audit['metadata_txt_files_ignored']+=1; continue
        try:
            sid,label=identity(path,root,mapping)
            parsed=parse_file(path,sid,label,time_unit,azimuth_unit,audit)
            records.extend(parsed); audit['txt_files_parsed']+=1
            if not parsed: audit['files_without_usable_trajectories']+=1
        except (ValueError,OSError,UnicodeError) as exc:
            audit['files_skipped_'+str(exc).split(':')[0]]+=1
    if not records: raise ValueError('No real schema-valid UCI trajectories. See parsing counters: '+json.dumps(dict(audit)))
    return records


def to_record(trial):
    record=asdict(trial)
    record['samples']=[[float(v) if math.isfinite(v) else None for v in row] for row in trial.samples]
    record['updrs']=float(trial.updrs) if math.isfinite(trial.updrs) else None
    record.update(schema_version='neurotrace-trajectory-1',time_units='seconds')
    Trajectory.from_dict(record) # Serialization round-trip is a hard boundary.
    return record


def partition_subjects(records,seed=42,holdout_fraction=.2):
    if not 0<holdout_fraction<1: raise ValueError('holdout_fraction must be in (0,1)')
    subjects={}
    for tr in records:
        if tr.subject_id in subjects and subjects[tr.subject_id]!=tr.label: raise ValueError('Conflicting parent diagnosis')
        subjects[tr.subject_id]=tr.label
    rng=np.random.default_rng(seed); development=[]; holdout=[]; unlabeled=[]
    for label in sorted(set(subjects.values())):
        ids=sorted(s for s,y in subjects.items() if y==label)
        if label==-1: unlabeled.extend(ids); continue
        if len(ids)<2: raise ValueError('Need >=2 genuine subjects/label for stratified development/holdout; never clone subjects')
        order=rng.permutation(ids).tolist(); n=max(1,min(len(ids)-1,int(round(len(ids)*holdout_fraction))))
        holdout.extend(order[:n]); development.extend(order[n:])
    assert not set(development)&set(holdout)
    assert set(subjects)==set(development)|set(holdout)|set(unlabeled)
    return dict(development_subjects=sorted(development),holdout_subjects=sorted(holdout),unlabeled_subjects=sorted(unlabeled))


def build(root,records_path='data/processed/v6_unified_records.json',partition_path='data/processed/partition.json',
          time_unit='milliseconds',azimuth_unit='unknown',seed=42,subject_map=None,feature_contract=None):
    audit=Counter(); records=load(root,time_unit,azimuth_unit,subject_map,audit)
    audit['schema_valid_trajectories_before_feature_qc']=len(records)
    # Optional feature-readiness QC BEFORE creating a new locked partition.
    # No resampling occurs in the loader; the existing feature pipeline owns it.
    if feature_contract:
        from src.preprocessing.v6_pipeline import transform
        eligible=[]
        for tr in records:
            try: transform(tr,contract=feature_contract); eligible.append(tr)
            except ValueError as exc: audit['feature_qc_skipped_'+str(exc)]+=1
        records=eligible
    partition=partition_subjects(records,seed)
    if Path(partition_path).exists():
        existing=json.loads(Path(partition_path).read_text())
        sets=[set(existing[k]) for k in ('development_subjects','holdout_subjects','unlabeled_subjects')]
        if any(sets[i]&sets[j] for i in range(3) for j in range(i)) or set.union(*sets)!={tr.subject_id for tr in records}:
            raise ValueError('Existing locked partition conflicts with prepared cohort; choose a reviewed new run, never silently overwrite it')
        partition=existing
    counts=Counter(tr.label for tr in {t.subject_id:t for t in records}.values())
    output=[to_record(tr) for tr in records]
    for path,value in [(Path(records_path),output),(Path(partition_path),partition)]:
        path.parent.mkdir(parents=True,exist_ok=True); temp=path.with_suffix(path.suffix+'.partial')
        temp.write_text(json.dumps(value,allow_nan=False)); temp.replace(path)
    summary=dict(raw_layout='verified headerless X,Y,Z,Pressure,GripAngle,Timestamp,TestID; named headers explicitly mapped',
        time_unit=time_unit,time_scale=TIME_SCALE[time_unit],
        time_unit_evidence='CONFIGURED ASSUMPTION: mirrored README does not declare timestamp units; no rate inference',
        azimuth_unit=azimuth_unit,coordinate_units='unknown (source README does not calibrate native coordinates)',
        counters=dict(audit),records_written=len(output),subjects_per_label=dict(counts),
        partitions={k:len(v) for k,v in partition.items()},seed=seed)
    Path(records_path).with_suffix('.ingestion.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2)); return records,partition,summary


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',required=True)
    p.add_argument('--records',default='data/processed/v6_unified_records.json'); p.add_argument('--partition',default='data/processed/partition.json')
    p.add_argument('--time-unit',choices=list(TIME_SCALE),default='milliseconds')
    p.add_argument('--azimuth-unit',choices=['unknown','degrees','radians'],default='unknown')
    p.add_argument('--seed',type=int,default=42); p.add_argument('--subject-map')
    p.add_argument('--feature-contract',choices=['v6-bands19-v1','v6-dwt19-v1'])
    a=p.parse_args(); build(a.root,a.records,a.partition,a.time_unit,a.azimuth_unit,a.seed,a.subject_map,a.feature_contract)

if __name__=='__main__': main()
