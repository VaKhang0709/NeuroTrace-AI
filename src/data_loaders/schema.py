"""Versioned, explicit-unit trajectory schema. Missing labels are never HC."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np

DOMAINS = ('wacom', 'touch', 'mouse', 'stylus', 'unknown')
TASKS = ('spiral', 'line', 'free_write')

@dataclass
class Trajectory:
    subject_id: str
    session_id: str
    task_id: str
    domain: str
    samples: np.ndarray  # columns x,y,t_seconds,pressure,azimuth_radians; missing sensors NaN
    label: int = -1
    updrs: float = float('nan')
    age_bucket: int = -1
    synthetic: bool = False
    source: str = 'unknown'
    coordinate_units: str = 'unknown'

    def validate(self):
        a = np.asarray(self.samples, dtype=float)
        if not self.subject_id or self.domain not in DOMAINS or self.label not in (-1, 0, 1):
            raise ValueError('Invalid identity, domain or label')
        if a.ndim != 2 or a.shape[1] != 5 or len(a) < 24:
            raise ValueError('Expected >=24 samples with x,y,seconds,pressure,azimuth columns')
        if not np.isfinite(a[:, :3]).all() or np.any(np.diff(a[:, 2]) <= 0):
            raise ValueError('Finite x/y and strictly increasing seconds required')
        if a[-1, 2] - a[0, 2] < .4:
            raise ValueError('Duration below 0.4 seconds')
        if np.isfinite(self.updrs) and not 0<=self.updrs<=132:
            raise ValueError('UPDRS Part III must be in [0,132] or missing')
        if self.age_bucket not in (-1,0,1,2,3,4):
            raise ValueError('Age bucket must be -1 (unknown) or 0..4')
        self.samples = a
        return self

    @classmethod
    def from_dict(cls, record):
        if record.get('schema_version') != 'neurotrace-trajectory-1' or record.get('time_units') != 'seconds':
            raise ValueError('Explicit unified schema and second timestamps required')
        fields = {k: v for k, v in record.items() if k in cls.__dataclass_fields__}
        fields['samples'] = np.asarray(fields['samples'], dtype=float)
        return cls(**fields).validate()


def load_unified(root):
    """Authorized harmonized records only. No automatic patient labels/units."""
    for path in sorted(Path(root).rglob('*.json')):
        yield Trajectory.from_dict(json.loads(path.read_text()))
