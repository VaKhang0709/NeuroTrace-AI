"""Compatibility raw-to-unified adapters: never reconstruct movement from an image.

UCI seven headerless semicolon columns: x;y;z;pressure;grip;timestamp_ms;test_id
(source: original audited V5 resolve_uci_columns). Grip is NOT assumed azimuth.
For nonstandard filenames provide explicit relative-path subject/diagnosis mapping.
"""
import csv
import re
from pathlib import Path
import numpy as np
from .schema import Trajectory,load_unified


def load_uci(root,subject_map=None,time_unit='milliseconds',azimuth_unit='unknown'):
    """Canonical UCI395 adapter; explicit clock units, named headers supported."""
    from .uci395_loader import load
    yield from load(root,time_unit=time_unit,azimuth_unit=azimuth_unit,subject_map=subject_map)


def load_authorized_mirror(root):
    """Open mirrors must supply reviewed neurotrace-trajectory-1 JSON conversion."""
    yield from load_unified(root)


def append_metadata_summary(records,report):
    """Aggregate supplied metadata, never raw trajectories or parent IDs."""
    from collections import defaultdict
    groups=defaultdict(list)
    for tr in records: groups[(tr.source,tr.domain,tr.label)].append(tr)
    lines=['','## Parsed unified metadata (operator-supplied labels, not clinical verification)','',
           '| Source | Domain | Label (-1 unknown) | Trajectories | Parent IDs | Points | Median sampling Hz |',
           '|---|---|---:|---:|---:|---:|---:|']
    for (source,domain,label),items in sorted(groups.items()):
        rates=[1/np.median(np.diff(t.samples[:,2])) for t in items]
        lines.append(f'| {source} | {domain} | {label} | {len(items)} | {len({t.subject_id for t in items})} | {sum(len(t.samples) for t in items)} | {np.median(rates):.3f} |')
    path=Path(report); path.write_text(path.read_text()+'\n'.join(lines)+'\n')
