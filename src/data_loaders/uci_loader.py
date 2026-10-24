"""UCI #395: https://archive.ics.uci.edu/dataset/395/parkinson+disease+spiral+drawings+using+digitized+graphics+tablet

Reuse original audited loader (no rasterization). Timestamp heuristic limitations
remain; review raw acquisition units before training. Requires authorized local raw data.
"""
import sys
import types
from pathlib import Path
from collections import Counter
import numpy as np
from .schema import Trajectory


def load(root):
    root=Path(root)
    if not root.exists():
        raise FileNotFoundError('UCI raw directory unavailable; no data fabricated')
    script=Path(__file__).resolve().parents[1]/'ai_model/scripts/train_pipeline_v5_diagnostic.py'
    module=types.ModuleType('_v6_uci_source'); module.__file__=str(script)
    sys.modules[module.__name__]=module
    exec(compile(script.read_text().split('\n',1)[1],str(script),'exec'),module.__dict__)
    trials=module.load_uci_trials(root,Counter())
    if not trials:
        raise ValueError('No valid UCI trials; review parser and units')
    for tr in trials:
        a=np.column_stack([tr.x,tr.y,tr.t,tr.pressure,np.full(len(tr.x),np.nan)])
        yield Trajectory(tr.parent_subject_id,tr.subject_id,tr.task,'wacom',a,
                         tr.label,source='uci395',coordinate_units='tablet-native-UNKNOWN').validate()
