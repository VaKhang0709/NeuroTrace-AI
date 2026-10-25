"""Strict raw-to-unified adapters: never reconstruct movement from an image.

UCI seven headerless semicolon columns: x;y;z;pressure;grip;timestamp_ms;test_id
(source: original audited V5 resolve_uci_columns). Grip is NOT assumed azimuth.
For nonstandard filenames provide explicit relative-path subject/diagnosis mapping.
"""
import csv
import re
from pathlib import Path
import numpy as np
from .schema import Trajectory,load_unified


def load_uci(root,subject_map=None):
    root=Path(root)
    mapping={}
    if subject_map:
        with Path(subject_map).open(newline='') as f:
            mapping={r['relative_path']:r for r in csv.DictReader(f)}
    accepted=0
    for path in sorted(root.rglob('*.txt')):
        record=mapping.get(path.relative_to(root).as_posix())
        if record:
            subject=str(record['subject_id']); label=int(record['label'])
        else:
            match=re.fullmatch(r'(?:H_)?([PC])[_-]?(\d+)(?:[_-]?(?:sst|dst|stcp|spiral|trial\d+))?',path.stem,re.I)
            if not match: continue
            label=int(match[1].upper()=='P'); subject='uci::'+match[1].upper()+match[2]
        try:
            a=np.genfromtxt(path,delimiter=';',dtype=float,ndmin=2)
            if a.shape[1]!=7: continue
            a=a[np.isfinite(a[:,[0,1,5,6]]).all(1)]
            cuts=np.r_[0,np.where(np.diff(a[:,6])!=0)[0]+1,len(a)]
            for k,(start,end) in enumerate(zip(cuts[:-1],cuts[1:])):
                block=a[start:end]
                if len(block)<24: continue
                t=(block[:,5]-block[0,5])*1e-3  # Explicit source ms, NOT median-unit heuristic.
                if np.any(np.diff(t)<=0): continue  # No silent timing repair.
                samples=np.column_stack([block[:,0],block[:,1],t,block[:,3],np.full(len(t),np.nan)])
                tr=Trajectory(subject,path.stem+f'_run{k}','spiral','wacom',samples,
                              label,source='uci395',coordinate_units='native_tablet_UNKNOWN').validate()
                accepted+=1; yield tr
        except (ValueError,OSError): continue
    if not accepted: raise ValueError('No valid strict UCI streams. Verify seven-column format, ms units and subject mapping; do not infer clinical labels.')


def load_authorized_mirror(root):
    """Open mirrors must supply reviewed neurotrace-trajectory-1 JSON conversion."""
    yield from load_unified(root)
