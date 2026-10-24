"""Source-only audit checks. Fixtures are synthetic, never participant records."""
import ast
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy import signal
from sklearn.model_selection import GroupKFold

SOURCE = Path('src/ai_model/scripts/train_pipeline_v5_diagnostic.py')


def original_functions(*names):
    tree = ast.parse(SOURCE.read_text().split('\n', 1)[1])
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = dict(math=math, np=np, signal=signal, random=random, defaultdict=defaultdict,
              Any=Any, Dict=Dict, List=List, Optional=Optional, Sequence=Sequence,
              Tuple=Tuple, GroupKFold=GroupKFold)
    # Execute selected pure functions, not the training module or model code.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<original-v5>', 'exec'), ns)
    return ns


def test_timestamp_heuristic_seconds_and_ms():
    f = original_functions('repair_time_axis')['repair_time_axis']
    np.testing.assert_allclose(f(np.arange(30)*10), np.arange(30)*.01)
    np.testing.assert_allclose(f(np.arange(30)*.01), np.arange(30)*.01)
    # Illustrates ambiguity: microsecond 10000 increments interpreted as ms!
    np.testing.assert_allclose(f(np.arange(30)*10000), np.arange(30)*10)


def test_nested_strokes_flattened_without_penup_marker():
    ns = original_functions('_to_float', 'repair_time_axis', 'extract_json_stroke_arrays')
    record = {'strokes': [{'points': [{'x': i, 'y': 1, 't': i*10} for i in range(12)]},
                          {'points': [{'x': i+100, 'y': 2, 't': (i+12)*10} for i in range(12)]}]}
    x, y, p, t = ns['extract_json_stroke_arrays'](record)
    assert x.shape == t.shape == (24,)
    assert x[12]-x[11] == 89
    assert np.isnan(p).all()
    np.testing.assert_allclose(np.diff(t), .01)


def test_nominal_100hz_is_not_exact():
    duration = .4
    n = max(48, int(round(duration*100))+1)
    assert n == 48
    assert not np.isclose(np.diff(np.linspace(0, duration, n))[0], .01)
    source = SOURCE.read_text()
    assert 'n_grid = max(48, int(round(dur * 100.0)) + 1)' in source
    assert 'np.gradient((xg - cx0) / sc0, 0.01)' in source


def test_parent_split_masks_keep_virtual_siblings_together():
    ns = original_functions('make_holdout_subjects', 'make_repeated_group_folds')
    table = {f's{i}': {'label': i % 2, 'source': i % 3} for i in range(30)}
    holdout = set(ns['make_holdout_subjects'](table, .1, 42))
    cv = sorted(set(table)-holdout)
    parents = np.repeat(sorted(table), 3)  # real and 2 virtual windows per parent
    virtual = np.tile([False, True, True], 30)
    for train, valid in ns['make_repeated_group_folds'](table, cv, 5, 42):
        train_mask = np.isin(parents, train)
        valid_mask = np.isin(parents, valid) & ~virtual
        assert not (set(parents[train_mask]) & set(parents[valid_mask]))
        assert not (set(parents[train_mask]) & holdout)
        for parent in train:
            assert train_mask[parents == parent].all()
        assert not virtual[valid_mask].any()
    # This fixture validates algorithm masks, NOT actual parent identities.


def test_contract_evidence_hashes_and_unknown_values():
    import hashlib
    cfg = json.loads(Path('docs/v5_input_contract.json').read_text())
    assert cfg['schema_version'] == '1.0.0'
    assert all(v is None for v in cfg['production_values'].values())
    assert all(f['evidence'] for f in cfg['findings'])
    for source in cfg['sources']:
        assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest() == source['sha256']


def test_window_feature_normalization_not_raw_dt_or_pixels():
    from types import SimpleNamespace
    from src.data.preprocessor import extract_trial_windows
    t = np.arange(240) * .01
    trial = SimpleNamespace(x=np.sin(3*t)*80+500, y=np.cos(3*t)*80+300,
                            t=t, pressure=np.linspace(.2, .8, 240),
                            z=np.full_like(t, np.nan), has_real_pressure=True)
    cfg = SimpleNamespace(window=256, stride=64, seq_len=256)
    window = extract_trial_windows(trial, cfg)[0]
    assert window.shape == (6, 256)
    np.testing.assert_allclose(window[:2].mean(axis=1), 0, atol=1e-6)
    np.testing.assert_allclose(np.mean(window[0]**2+window[1]**2), 1, atol=1e-6)
    np.testing.assert_allclose(window[3].mean(), 0, atol=1e-6)
    assert abs(np.median(window[5])) < 1e-6
    assert window[4].mean() > 0
