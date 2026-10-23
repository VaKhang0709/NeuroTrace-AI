%%writefile /content/drive/MyDrive/NEUROTRACE/train_pipeline_v4_augmented.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_pipeline_v4_augmented.py — NeuroTrace v4, augmented subject-grouped trainer
===============================================================================

v3 already splits strictly by SUBJECT so no person is ever on both sides of a
split. v4 keeps that guarantee and adds physics-preserving AUGMENTATION, which
is the honest way to get more training signal out of 92 people instead of
inventing subjects.

What is new versus v3
  1. SLIDING-WINDOW SLICING. Each raw recording is cut into overlapping
     windows of ``--window`` samples with ``--stride`` (256 / 64 by default).
     A 700-sample spiral becomes ~7 windows instead of 1 trial.
  2. ROTATION AUGMENTATION (TRAINING FOLD ONLY). Every training window is also
     emitted rotated by -30, -15, +15 and +30 degrees about the drawing centre,
     the two-dimensional rotation:

         x_rot = x*cos(theta) - y*sin(theta)
         y_rot = x*sin(theta) + y*cos(theta)

     dt, velocity, pressure and Z are scalars, so the same rotation leaves them
     untouched — which is exactly what the spec asks for ("preserve raw
     continuous kinematics intact"). Speed is stored as a magnitude
     (hypot(vx, vy)), which is rotation-invariant, so nothing needs rewriting.
  3. Augmented copies inherit their parent's ``subject_id``, so GroupKFold
     keeps every variant of a person in the same fold. Augmentation is applied
     AFTER the split and NEVER to validation data.

===============================================================================
 MEDICAL DISCLAIMER (keep this in every artefact derived from this file)
===============================================================================
NeuroTrace AI is a RESEARCH PROTOTYPE, not a medical device. It is not approved
by any regulator (FDA / EMA / Vietnam MoH). Its output is a statistical risk
indicator for research and education only — NOT a diagnosis, and never a reason
to start, stop or change treatment. Anyone noticing tremor, stiffness, slowness
or handwriting change must see a qualified neurologist.

ĐÂY LÀ NGUYÊN MẪU NGHIÊN CỨU, KHÔNG PHẢI THIẾT BỊ Y TẾ — không phải chẩn đoán.

===============================================================================
 FOUR THINGS THIS SCRIPT REFUSES TO PRETEND ABOUT
===============================================================================
(A) AUGMENTATION DOES NOT CREATE SUBJECTS. 1,500 windows from 92 people is
    still 92 people. Every metric below is computed at SUBJECT level (window
    probabilities averaged within a person), and the "n" that governs the
    confidence interval is 92, not the window count. Any number quoted without
    that caveat is misleading.

(B) DEVICE / LABEL CONFOUND. Every Parkinson subject comes from the Wacom
    tablet; every app subject is healthy AND comes from a touch screen, so a
    model can score well by detecting the DEVICE. Section [6] measures it.

(C) CROSS-DEVICE SENSITIVITY IS UNDEFINED — the app holdout is healthy only.

(D) ROTATION IS ONLY SAFE BECAUSE THE LABEL IS ROTATION-INVARIANT. Tremor
    frequency, bradykinesia and micrographia do not depend on how the tablet
    was oriented. Scale augmentation would NOT be safe (micrographia is itself
    a sign), which is why none is applied here.

===============================================================================
 SPEC FIX APPLIED / SỬA ĐIỂM MÂU THUẪN TRONG ĐẶC TẢ
===============================================================================
The spec asks for BCEWithLogitsLoss and reports probabilities. Resolved as:
    * ``forward()`` returns RAW LOGITS  -> feed BCEWithLogitsLoss directly.
    * ``predict_proba()`` applies sigmoid -> used for every metric.
Applying a sigmoid inside forward() as well would double-squash and destroy
training, so the sigmoid lives in inference, not in the graph.

===============================================================================
 USAGE / CÁCH DÙNG (Google Colab)
===============================================================================
  Cell 1: paste this file (the %%writefile line creates it on Drive)
  Cell 2: !python3 /content/drive/MyDrive/NEUROTRACE/train_pipeline_v4_augmented.py \
            --uci-root /content/drive/MyDrive/NEUROTRACE/external_test_data/uci_395_raw \
            --json-root /content/drive/MyDrive/NEUROTRACE/healthy

  Data check only (no torch needed):   ... --audit-only
  Smoke run:                           ... --epochs 3
  Custom rotations (note the "="):     ... --rotation-angles=-45,-20,20,45
  No augmentation (v3 behaviour):      ... --no-augment

Inputs   <project>/data/uci_raw/**/*.txt     UCI #395 (Test ID 0 and 1 only)
         <project>/data/healthy/**/*.json    NeuroTrace app captures
Outputs  <project>/models/neurotrace_v4_augmented.pth
         <project>/models/scaler_v4.pkl
         <project>/models/train_report_v4.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import random
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# sklearn — required everywhere (splits, scaling, metrics)
# ---------------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# torch — required for training, optional for --audit-only
# ---------------------------------------------------------------------------
try:  # pragma: no cover - environment dependent
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = TensorDataset = None
    HAS_TORCH = False

# ---------------------------------------------------------------------------
# scipy CubicSpline is a quality upgrade; np.interp is a fine fallback.
# ---------------------------------------------------------------------------
try:  # pragma: no cover
    from scipy.interpolate import CubicSpline

    HAS_SCIPY = True
except Exception:  # pragma: no cover
    CubicSpline = None
    HAS_SCIPY = False


# ===========================================================================
# Configuration / Cấu hình
# ===========================================================================
PROJECT_ROOT = Path("/content/drive/MyDrive/NEUROTRACE")
UCI_REL = Path("data") / "uci_raw"          # v4: was external_test_data/uci_395_raw
JSON_REL = Path("data") / "healthy"         # v4: was healthy/
MODEL_REL = Path("models")

# ---- artefact names (v4) ----
MODEL_FILENAME = "neurotrace_v4_augmented.pth"
SCALER_FILENAME = "scaler_v4.pkl"
REPORT_FILENAME = "train_report_v4.json"

# ---- audit expectation: 92 unique subjects, 27 PD + 65 HC ----
EXPECTED_SUBJECTS = 92
EXPECTED_PD = 27
EXPECTED_HC = 65

# ---- augmentation defaults (spec) ----
DEFAULT_WINDOW = 256
DEFAULT_STRIDE = 64
DEFAULT_ROTATION_ANGLES: Tuple[float, ...] = (-30.0, -15.0, 15.0, 30.0)
MIN_AUGMENTED_SAMPLES = 600

FEATURE_NAMES: Tuple[str, ...] = ("x", "y", "z", "pressure", "velocity", "dt")
N_CHANNELS = len(FEATURE_NAMES)

# UCI-395 test ids we train on (0 = Static Spiral, 1 = Dynamic Spiral).
UCI_TRAIN_TEST_IDS = (0, 1)
# 2 = Stability Test on Certain Point ("Circular") — excluded by spec.

# Positional layout of the HEADERLESS UCI #395 text dumps, per the
# dataset's own variable table. Most files are raw 7-column semicolon
# dumps with NO header row at all, so name-based lookup has nothing to
# match and every one of them used to be dropped as uci_missing_xy.
UCI_POSITIONAL_ORDER: Tuple[str, ...] = (
    "x", "y", "z", "pressure", "gripangle", "timestamp", "testid",
)

# Legacy locations the project used before v4 (they were the v3 defaults).
# Kept as fallbacks so the script still finds the data if a folder was
# never renamed, instead of silently training on zero UCI subjects.
LEGACY_UCI_RELS: Tuple[Path, ...] = (
    Path("external_test_data") / "uci_395_raw",
    Path("external_test_data") / "uci_395",
    Path("uci_raw"),
)
LEGACY_JSON_RELS: Tuple[Path, ...] = (Path("healthy"),)

CONTROL_TOKENS = {"control", "controls", "healthy", "health", "hc", "normal", "normals", "ctl", "con"}
PD_TOKENS = {"parkinson", "parkinsons", "pd", "pwp", "patient", "patients", "pdp", "disease", "hasta"}

ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
CANDIDATE_DELIMITERS = (";", ",", "\t", "|")

COLUMN_ALIASES: Dict[str, Tuple[str, ...]] = {
    "x": ("x", "x_coord", "xcoordinate", "xposition", "pos_x", "xpos"),
    "y": ("y", "y_coord", "ycoordinate", "yposition", "pos_y", "ypos"),
    "z": ("z", "z_coord", "zcoordinate"),
    "pressure": ("pressure", "press", "force", "pen_pressure"),
    "gripangle": ("gripangle", "grip_angle", "angle", "grip", "tilt"),
    "timestamp": ("timestamp", "time", "t", "timestamp_ms", "time_ms", "time_sec",
                  "systemtime", "system_time", "time_stamp", "datetime"),
    "testid": ("testid", "test_id", "test", "task", "taskid", "task_id", "testtype", "test_type"),
}
TEST_ID_TEXT_HINTS: List[Tuple[str, int]] = [
    ("static", 0), ("sst", 0),
    ("dynamic", 1), ("dst", 1), ("blink", 1),
    ("stability", 2), ("stcp", 2), ("circular", 2), ("circle", 2), ("point", 2),
]
WRAPPER_KEYS = ("data", "records", "recordings", "items", "drawings", "sessions",
                "results", "entries", "rows", "captures", "samples", "files", "list",
                "content", "payload", "export", "exports")
POINT_KEYS = ("strokes", "points", "path", "samples", "trajectory", "coordinates", "data")
PARTICIPANT_STRONG = {
    "participantid", "participant", "subjectid", "subject", "patientid", "patient",
    "userid", "user", "username", "uid", "uuid", "personid", "person",
    "volunteerid", "volunteer", "email", "respondentid", "respondent",
}
PARTICIPANT_PROXY = {"deviceid", "deviceuuid", "device", "installationid", "clientid", "anonymousid"}

MIN_SAMPLES_PER_TRIAL = 32


def _resolve_data_dir(root: Path, explicit: str, primary: Path,
                      fallbacks: Sequence[Path], what: str,
                      notes: Optional[List[str]] = None) -> Path:
    """Pick a data directory: explicit flag > v4 path > any legacy path."""
    if explicit:
        return Path(explicit)
    p = root / primary
    if p.exists():
        return p
    for fb in fallbacks:
        cand = root / fb
        if cand.exists():
            if notes is not None:
                notes.append(f"{what}: {primary} not found - using legacy path {fb}")
            return cand
    return p


@dataclass
class TrainConfig:
    """Every knob that changes the numbers. Shipped inside the model bundle."""

    # ---- paths ----
    project_root: str = str(PROJECT_ROOT)
    uci_root: str = ""
    json_root: str = ""
    model_dir: str = ""
    cache_path: str = ""

    # ---- features ----
    seq_len: int = 256
    resample_by: str = "time"             # time | arclength
    geometry: str = "center_scale"        # center_scale | none
    z_fill: str = "median"                # median | zero
    pressure_mapping: str = "quantile"    # quantile | none
    append_duration_channel: bool = False
    app_tasks: str = "auto"               # auto | spiral | all
    max_trials_per_subject: int = 0       # 0 = no cap

    # ---- v4 augmentation ----
    window: int = DEFAULT_WINDOW          # sliding-window length in samples
    stride: int = DEFAULT_STRIDE          # sliding-window stride in samples
    rotation_angles: Tuple[float, ...] = DEFAULT_ROTATION_ANGLES
    augment_train: bool = True            # rotate TRAINING windows only
    min_samples: int = MIN_AUGMENTED_SAMPLES   # audit warning threshold

    # ---- model ----
    conv1: int = 64
    conv2: int = 128
    lstm_hidden: int = 64
    lstm_layers: int = 2
    dropout: float = 0.3

    # ---- optimisation ----
    batch_size: int = 32
    epochs: int = 40                      # v4 spec
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    early_stop_patience: int = 10         # v4 spec
    min_epochs: int = 5
    use_scheduler: bool = True            # CosineAnnealingLR in v4

    # ---- validation ----
    n_splits: int = 5
    holdout_frac: float = 0.10
    threshold: float = 0.5

    # ---- audit expectation: 92 subjects = 27 PD + 65 HC ----
    expected_subjects: int = EXPECTED_SUBJECTS
    expected_pd: int = EXPECTED_PD
    expected_hc: int = EXPECTED_HC

    # ---- misc ----
    seed: int = 42
    num_workers: int = 0
    limit_subjects: int = 0               # 0 = no limit (smoke tests)
    audit_only: bool = False
    device: str = "auto"
    verbose: bool = True

    def resolved_paths(self, notes: Optional[List[str]] = None
                       ) -> Tuple[Path, Path, Path]:
        root = Path(self.project_root)
        uci = _resolve_data_dir(root, self.uci_root, UCI_REL,
                                LEGACY_UCI_RELS, "UCI", notes)
        js = _resolve_data_dir(root, self.json_root, JSON_REL,
                               LEGACY_JSON_RELS, "JSON", notes)
        models = Path(self.model_dir) if self.model_dir else (root / MODEL_REL)
        return uci, js, models


# ===========================================================================
# Small helpers / Hàm phụ trợ
# ===========================================================================
def rule(char: str = "=", width: int = 88) -> str:
    return char * width


def section(title: str, width: int = 88) -> None:
    print("\n" + rule("=", width))
    print(f" {title}")
    print(rule("=", width))


def kv(label: str, value: Any, indent: int = 2, width: int = 36) -> None:
    print(f"{' ' * indent}{label:<{width}}: {value}")


def human(n: Any) -> str:
    return f"{n:,}" if isinstance(n, (int, np.integer)) else ("n/a" if n is None else str(n))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(pref: str = "auto"):
    if not HAS_TORCH:
        return None
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_text_safely(path: Path) -> str:
    for enc in ENCODINGS:
        try:
            with open(path, "r", encoding=enc, errors="strict") as fh:
                return fh.read()
        except UnicodeDecodeError:
            continue
        except Exception:
            break
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def normalise_key(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip().lower()).strip("_").replace("_", "")


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    try:
        return float(str(v).strip().replace(",", "."))
    except Exception:
        return None


# ===========================================================================
# 1. UCI #395 loader / Bộ nạp UCI
# ===========================================================================
@dataclass
class RawTrial:
    """One contiguous recording block from one device, before feature building."""

    subject_id: str
    label: int                 # 1 = PD, 0 = healthy
    source: str                # "uci" | "app"
    device: str                # "wacom_cintiq" | "touchscreen"
    test_id: int
    task: str
    origin: str                # file path for traceability
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray              # may be all-NaN for app data
    pressure: np.ndarray       # may be all-NaN for app data
    t: np.ndarray              # seconds, monotonically increasing
    has_real_pressure: bool = False
    synthetic_time: bool = False


def normalise_subject_code(stem: str) -> str:
    """'H_P000-0025' -> 'H_P000'; 'C_0009' stays (see the audit script)."""
    blocks = list(re.finditer(r"\d+", stem))
    if len(blocks) >= 2:
        last = blocks[-1]
        if last.end() == len(stem):
            head = stem[: last.start()].rstrip("-_ .")
            if head:
                return head
    return stem


def infer_group_from_path(rel: Path) -> Optional[int]:
    """1 = PD, 0 = control, None = unknown. Directory first, then filename."""
    for part in reversed(rel.parts[:-1]):
        token = re.sub(r"[^a-z]", "", part.lower())
        if token in PD_TOKENS:
            return 1
        if token in CONTROL_TOKENS:
            return 0
    m = re.match(r"^[A-Za-z][-_ ]?([CPcp])[-_ ]?\d", rel.stem)
    if m:
        letter = m.group(1).upper()
        return 0 if letter == "C" else 1
    return None


def normalise_test_id(raw: str) -> Optional[int]:
    token = str(raw).strip()
    if not token:
        return None
    val = _to_float(token)
    if val is not None and float(val).is_integer():
        iv = int(val)
        if iv in (0, 1, 2):
            return iv
    low = token.lower()
    for hint, tid in TEST_ID_TEXT_HINTS:
        if hint in low:
            return tid
    return None


def read_uci_table(path: Path) -> Tuple[List[str], Dict[str, List[str]]]:
    """Semicolon-first delimited reader with header auto-detection."""
    text = read_text_safely(path)
    if not text.strip():
        return [], {}
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return [], {}

    counts = {d: "\n".join(lines[:5]).count(d) for d in CANDIDATE_DELIMITERS}
    delim = ";" if counts.get(";", 0) > 0 else max(counts, key=lambda d: counts[d])

    rows = [r for r in csv.reader(lines, delimiter=delim, skipinitialspace=True)
            if any(c.strip() for c in r)]
    if not rows:
        return [], {}

    def numeric(cell: str) -> bool:
        return bool(re.match(r"^[+-]?(\d+([.,]\d*)?|[.,]\d+)([eE][+-]?\d+)?$", cell.strip()))

    lead = [c for c in rows[0][:4] if c.strip()]
    if len(lead) >= 4 and all(numeric(c) for c in lead):
        has_header = False
    else:
        has_header = any((not numeric(c)) and c.strip() != "" for c in rows[0])
    columns = [c.strip() for c in rows[0]] if has_header else []
    body = rows[1:] if has_header else rows
    if not columns:
        width = max(len(r) for r in rows)
        columns = [f"col_{i}" for i in range(width)]

    data: Dict[str, List[str]] = {c: [] for c in columns}
    for row in body:
        for i, col in enumerate(columns):
            data[col].append(row[i].strip() if i < len(row) else "")
    return columns, data


def find_col(columns: Sequence[str], alias_key: str) -> Optional[str]:
    wanted = {normalise_key(a) for a in COLUMN_ALIASES[alias_key]}
    for c in columns:
        if normalise_key(c) in wanted:
            return c
    return None


def _columns_look_numeric(columns: Sequence[str], data: Dict[str, List[str]],
                          sample: int = 25, min_frac: float = 0.8) -> bool:
    """True when the leading columns of a headerless table parse as numbers.

    This guards the positional fallback below: a file whose first row is
    prose is a genuinely different (and unreadable) file, not a headerless
    numeric dump, and treating it as one would map words onto the x channel.
    """
    checked = 0
    numeric = 0
    for col in list(columns)[:4]:
        for cell in list(data.get(col, []))[:sample]:
            cell = (cell or "").strip()
            if not cell:
                continue
            checked += 1
            if _to_float(cell) is not None:
                numeric += 1
    if checked == 0:
        return False
    return (numeric / checked) >= min_frac


def resolve_uci_columns(columns: Sequence[str], data: Dict[str, List[str]],
                        stats: Optional[Counter] = None
                        ) -> Optional[Dict[str, Optional[str]]]:
    """Resolve the UCI columns by NAME first, then by POSITION.

    The UCI #395 files are raw semicolon dumps with no header row, so the
    reader names their fields ``col_0 ... col_6`` and no alias ever matches.
    That is what dropped 77 of 79 files as ``uci_missing_xy``: only the two
    files that happen to carry a header were ever parsed. When the name
    lookup for x/y fails we fall back to the documented positional layout
    (x, y, z, pressure, gripangle, timestamp, testid) - but only after
    checking there are at least four columns and that they really are
    numeric, so a text file can never be silently misread as data.
    """
    x_col = find_col(columns, "x")
    y_col = find_col(columns, "y")
    if x_col is not None and y_col is not None:
        return {"x": x_col, "y": y_col,
                "z": find_col(columns, "z"),
                "pressure": find_col(columns, "pressure"),
                "gripangle": find_col(columns, "gripangle"),
                "timestamp": find_col(columns, "timestamp"),
                "testid": find_col(columns, "testid")}

    # ---- headerless fallback: map by position ----
    if len(columns) < 4 or not _columns_look_numeric(columns, data):
        return None

    named: Dict[str, Optional[str]] = {role: None for role in UCI_POSITIONAL_ORDER}
    for i, role in enumerate(UCI_POSITIONAL_ORDER):
        if i < len(columns):
            named[role] = columns[i]
    if stats is not None:
        stats["uci_positional_fallback"] += 1
        if len(columns) < len(UCI_POSITIONAL_ORDER):
            stats["uci_no_test_id_column"] += 1
    return named


def repair_time_axis(t: np.ndarray) -> Tuple[Optional[np.ndarray], int, float]:
    """Force a strictly increasing time axis / Ép trục thời gian tăng dần.

    Handwriting capture glitches produce three common defects:
      1. a **clock restart** (each test block or each stroke re-zeroes t),
      2. **duplicate timestamps** (two samples with the same t),
      3. occasional **backwards jumps**.

    The naive fix ``np.maximum.accumulate(t)`` is a trap: as soon as a clock
    restarts, it flattens every later sample onto the same timestamp, so
    ``dt`` collapses to 0 and the velocity channel explodes by orders of
    magnitude (a 0.001 s floor on a 0.008 s step inflates speed 8x; worse for
    larger steps). The model then reads that artefact as a real signal.

    Instead we keep the *observed* intervals and substitute the median positive
    step only where the clock failed to advance, which preserves every real
    interval and degrades gracefully.

    Returns ``(t_fixed, n_repaired, median_step)``. ``t_fixed`` is ``None`` when
    the clock never advances at all - the caller must then synthesise a time
    axis rather than trust it.
    """
    t = np.asarray(t, dtype=float)
    if t.size < 2:
        return t, 0, 0.0
    d = np.diff(t)
    pos = d[d > 0]
    if pos.size == 0:
        return None, int(d.size), 0.0
    step = float(np.median(pos))
    bad = ~(d > 0)
    n_bad = int(bad.sum())
    if n_bad:
        fixed = np.empty_like(t)
        fixed[0] = t[0]
        fixed[1:] = t[0] + np.cumsum(np.where(d > 0, d, step))
        return fixed, n_bad, step
    return t, 0, step


def continuous_time_across_strokes(t: np.ndarray, seg: Sequence[int],
                                   hz: float = 60.0) -> np.ndarray:
    """Chain per-stroke clocks into one continuous time axis.

    Each stroke is repaired on its own (see :func:`repair_time_axis`) and then
    shifted so that it begins one sampling step after the previous stroke ends.
    Stroke order and the within-stroke velocity profile are both preserved;
    absolute pen-up gaps are not recoverable from per-stroke timers, so a
    single sampling step is used as the join - honest and stable.
    """
    t = np.asarray(t, dtype=float)
    seg = np.asarray(seg)
    if seg.shape[0] != t.shape[0]:
        seg = np.zeros(t.shape[0], dtype=int)
    out = np.empty_like(t)
    offset = 0.0
    for s in np.unique(seg):
        m = seg == s
        ts = t[m]
        if not np.isfinite(ts).any() or ts.size < 2:
            out[m] = np.arange(ts.size, dtype=float) / hz + offset
            offset = out[m][-1] + 1.0 / hz
            continue
        ts = np.nan_to_num(ts, nan=float(np.nanmedian(ts[np.isfinite(ts)])))
        fixed, _, step = repair_time_axis(ts)
        if fixed is None or step <= 0:
            fixed = np.arange(ts.size, dtype=float) / hz
            step = 1.0 / hz
        fixed = fixed - fixed[0] + offset
        out[m] = fixed
        offset = fixed[-1] + step
    return out - out[0]


def is_degenerate_channel(a: np.ndarray, min_unique: int = 2,
                          rtol: float = 1e-9) -> bool:
    """True when a channel carries no information / Kênh không mang thông tin.

    Many capture apps always write a placeholder value (e.g. ``"p": 0.5`` on
    every point) even when the device reports no real pressure. Treating that
    as real pressure silently disables the virtual-pressure proxy and hands the
    network a constant channel - which for this dataset is a perfect
    device fingerprint (constant on touch, varying on the Wacom tablet).
    """
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return True
    if np.unique(a).size < min_unique:
        return True
    scale = max(abs(float(np.mean(a))), 1e-12)
    return bool(np.std(a) / scale < rtol)


def contiguous_runs(values: Sequence[Optional[int]]) -> List[Tuple[Optional[int], int, int]]:
    """[(value, start, end_exclusive)] over contiguous equal values."""
    runs: List[Tuple[Optional[int], int, int]] = []
    if not values:
        return runs
    start = 0
    for i in range(1, len(values)):
        if values[i] != values[start]:
            runs.append((values[start], start, i))
            start = i
    runs.append((values[start], start, len(values)))
    return runs


def load_uci_trials(root: Path, cfg: TrainConfig, stats: Counter) -> List[RawTrial]:
    """Every Static/Dynamic Spiral block in every .txt under ``root``."""
    trials: List[RawTrial] = []
    if not root.exists():
        stats["uci_root_missing"] += 1
        return trials

    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".txt")
    if cfg.limit_subjects:
        paths = paths[: cfg.limit_subjects * 4]
    stats["uci_files"] = len(paths)

    for path in paths:
        try:
            rel = path.relative_to(root) if str(path).startswith(str(root)) else path
            label = infer_group_from_path(Path(rel))
            if label is None:
                stats["uci_unlabelled_files"] += 1
                continue

            columns, data = read_uci_table(path)
            if not columns or not data:
                stats["uci_unreadable_files"] += 1
                continue

            # Name lookup first; headerless files fall back to the
            # positional UCI #395 layout (see resolve_uci_columns).
            resolved = resolve_uci_columns(columns, data, stats)
            if resolved is None:
                stats["uci_missing_xy"] += 1
                continue
            x_col, y_col = resolved["x"], resolved["y"]
            t_col, test_col = resolved["timestamp"], resolved["testid"]

            n_rows = len(data[x_col])
            if n_rows < MIN_SAMPLES_PER_TRIAL:
                stats["uci_too_short"] += 1
                continue

            def col_to_array(key: str) -> np.ndarray:
                c = resolved.get(key)
                if c is None or c not in data:
                    return np.full(n_rows, np.nan)
                return np.array([_to_float(v) for v in data[c]], dtype=float)

            xs, ys = col_to_array("x"), col_to_array("y")
            zs = col_to_array("z")
            ps = col_to_array("pressure")
            ts = col_to_array("timestamp") if t_col else np.arange(n_rows, dtype=float)

            # --- timestamps -> seconds ---
            synthetic_time = False
            if t_col is None or not np.isfinite(ts).any():
                ts = np.arange(n_rows, dtype=float) / 60.0   # assume 60 Hz
                synthetic_time = True
                stats["uci_synthetic_time"] += 1
            else:
                ts = np.nan_to_num(ts, nan=np.nanmedian(ts))
                diffs = np.diff(ts)
                step = np.median(diffs[diffs > 0]) if np.any(diffs > 0) else 0.0
                if step >= 1.0:                 # milliseconds
                    ts = ts / 1000.0
                ts = ts - ts[0]
                # NOTE: do NOT use np.maximum.accumulate here. Files that hold
                # several test blocks re-zero the clock per block; the running
                # maximum would flatten every later block onto one timestamp.
                # The repair is done per trial (after block splitting) instead.

            # --- split into contiguous test blocks ---
            if test_col is not None:
                raw_ids = [normalise_test_id(v) for v in data[test_col]]
            else:
                inferred = normalise_test_id(path.stem) or normalise_test_id(str(rel))
                raw_ids = [inferred] * n_rows
                if inferred is None:
                    stats["uci_no_test_id"] += 1
                    continue

            subject = f"uci::{normalise_subject_code(path.stem)}"
            for tid, i0, i1 in contiguous_runs(raw_ids):
                if tid not in UCI_TRAIN_TEST_IDS:
                    continue
                if (i1 - i0) < MIN_SAMPLES_PER_TRIAL:
                    stats["uci_short_blocks"] += 1
                    continue
                sl = slice(i0, i1)
                blk_t = ts[sl]
                blk_t, n_rep, _ = repair_time_axis(blk_t)
                if blk_t is None:                      # clock never advances
                    blk_t = np.arange(i1 - i0, dtype=float) / 60.0
                    synthetic_time_blk = True
                    stats["uci_synthetic_time"] += 1
                else:
                    synthetic_time_blk = synthetic_time
                    if n_rep:
                        stats["uci_timestamp_repairs"] += n_rep
                trials.append(RawTrial(
                    subject_id=subject, label=label, source="uci",
                    device="wacom_cintiq", test_id=int(tid),
                    task="static_spiral" if tid == 0 else "dynamic_spiral",
                    origin=str(rel),
                    x=xs[sl], y=ys[sl], z=zs[sl], pressure=ps[sl], t=blk_t,
                    has_real_pressure=bool(np.isfinite(ps[sl]).any()),
                    synthetic_time=synthetic_time_blk,
                ))
                stats["uci_trials"] += 1
        except Exception as exc:  # noqa: BLE001 - one bad file must not kill the run
            stats["uci_errors"] += 1
            stats[f"uci_error::{type(exc).__name__}"] += 1
            continue
    return trials


# ===========================================================================
# 2. App JSON loader / Bộ nạp JSON ứng dụng
# ===========================================================================
def load_json_records(path: Path) -> List[Any]:
    text = read_text_safely(path)
    if not text.strip():
        return []
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            for key in WRAPPER_KEYS:
                v = payload.get(key)
                if isinstance(v, list):
                    return list(v)
            return [payload]
        if isinstance(payload, list):
            return list(payload)
        return [payload]
    except Exception:
        out = []
        for line in text.splitlines():
            line = line.strip().rstrip(",")
            if not line or line in ("[", "]", "{", "}"):
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
        return out


def _first_present(d: Dict[str, Any], names: Sequence[str]) -> Optional[Any]:
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    lowered = {normalise_key(k): k for k in d.keys()}
    for n in names:
        k = lowered.get(normalise_key(n))
        if k is not None and d[k] is not None:
            return d[k]
    return None


def extract_strokes(record: Any) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Return (x, y, t, pressure) for one drawing record, or None.

    Understands: {"strokes": [[{x,y,t,p}, ...], ...]}, flat point lists,
    {"x": [...], "y": [...]} layouts, and record-level pressure/radius fields.
    Missing timestamps are synthesised at 60 Hz (flagged by the caller).
    """
    if not isinstance(record, dict):
        return None

    xs: List[float] = []
    ys: List[float] = []
    ts: List[float] = []
    ps: List[float] = []
    seg: List[int] = []            # which stroke each point belongs to
    have_t = False

    raw_strokes = _first_present(record, POINT_KEYS)
    fallback_pressure = _first_present(record, ("pressure", "force", "p", "radius", "r"))

    if isinstance(raw_strokes, list) and raw_strokes:
        for si, stroke in enumerate(raw_strokes):
            if isinstance(stroke, list):
                for pt in stroke:
                    if not isinstance(pt, dict):
                        continue
                    xv, yv = _to_float(_first_present(pt, ("x", "X"))), _to_float(_first_present(pt, ("y", "Y")))
                    if xv is None or yv is None:
                        continue
                    xs.append(xv)
                    ys.append(yv)
                    tv = _to_float(_first_present(pt, ("t", "timestamp", "time", "ts")))
                    if tv is not None:
                        ts.append(tv)
                        have_t = True
                    else:
                        ts.append(np.nan)
                    pv = _to_float(_first_present(pt, ("pressure", "force", "p", "radius", "r")))
                    ps.append(pv if pv is not None else np.nan)
                    seg.append(si)
            elif isinstance(stroke, dict):
                xv, yv = _to_float(_first_present(stroke, ("x", "X"))), _to_float(_first_present(stroke, ("y", "Y")))
                if xv is None or yv is None:
                    continue
                xs.append(xv)
                ys.append(yv)
                tv = _to_float(_first_present(stroke, ("t", "timestamp", "time", "ts")))
                ts.append(tv if tv is not None else np.nan)
                have_t = have_t or tv is not None
                pv = _to_float(_first_present(stroke, ("pressure", "force", "p", "radius", "r")))
                ps.append(pv if pv is not None else np.nan)
                seg.append(si)
    else:
        xa = _first_present(record, ("x", "X"))
        ya = _first_present(record, ("y", "Y"))
        if not (isinstance(xa, list) and isinstance(ya, list) and len(xa) == len(ya) and xa):
            return None
        xs = [float(v) for v in xa]
        ys = [float(v) for v in ya]
        seg = [0] * len(xs)
        ta = _first_present(record, ("t", "timestamp", "time", "timestamps", "ts"))
        if isinstance(ta, list) and len(ta) == len(xs):
            ts = [float(v) for v in ta]
            have_t = True
        else:
            ts = [np.nan] * len(xs)
        pa = _first_present(record, ("pressure", "force", "p", "radius", "r"))
        ps = ([float(v) for v in pa] if isinstance(pa, list) and len(pa) == len(xs)
              else [np.nan] * len(xs))

    if len(xs) < MIN_SAMPLES_PER_TRIAL:
        return None

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    t = np.asarray(ts, dtype=float)
    p = np.asarray(ps, dtype=float)

    if not have_t or not np.isfinite(t).any():
        t = np.arange(len(xs), dtype=float) / 60.0
        have_t = False
    else:
        t = np.nan_to_num(t, nan=np.nanmedian(t[np.isfinite(t)]))
        diffs = np.diff(t)
        step = np.median(diffs[diffs > 0]) if np.any(diffs > 0) else 0.0
        if step >= 1.0:                       # milliseconds -> seconds
            t = t / 1000.0
        # A drawing can hold several strokes and capture code very often
        # re-zeroes the timer at the start of each one. Flattening the series
        # with np.maximum.accumulate would collapse every stroke after the
        # first onto a single timestamp (dt = 0 -> velocity explodes), so the
        # strokes are re-chained onto one continuous clock instead.
        t = continuous_time_across_strokes(t, seg if len(seg) == len(xs) else [0] * len(xs))

    if not np.isfinite(p).any() and fallback_pressure is not None:
        fv = _to_float(fallback_pressure)
        if fv is not None:
            p = np.full(len(x), fv)

    return x, y, t, p


def json_participant(record: Dict[str, Any], rel: Path) -> Optional[str]:
    for key, value in record.items():
        if isinstance(value, (dict, list)):
            continue
        if normalise_key(key) in PARTICIPANT_STRONG and str(value).strip():
            return str(value).strip()
    for key, value in record.items():
        if isinstance(value, (dict, list)):
            continue
        if normalise_key(key) in PARTICIPANT_PROXY and str(value).strip():
            return str(value).strip()
    m = re.search(r"(participant[-_ ]?\w?\d+|subject[-_ ]?\w?\d+|user[-_ ]?\w?\d+)", str(rel).lower())
    return m.group(1) if m else None


def load_app_trials(root: Path, cfg: TrainConfig, stats: Counter) -> List[RawTrial]:
    """Every usable drawing in every .json under ``root`` (all healthy)."""
    trials: List[RawTrial] = []
    if not root.exists():
        stats["json_root_missing"] += 1
        return trials

    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".json")
    stats["json_files"] = len(paths)

    seen_tasks: Counter = Counter()
    by_participant: Dict[str, List[RawTrial]] = defaultdict(list)

    for path in paths:
        try:
            rel = path.relative_to(root) if str(path).startswith(str(root)) else path
            for record in load_json_records(path):
                if not isinstance(record, dict):
                    continue
                task = _first_present(record, ("task", "tasktype", "testtype", "test", "shape"))
                task_name = str(task).lower() if task is not None else "unknown"
                seen_tasks[task_name] += 1

                pid = json_participant(record, Path(rel))
                if pid is None:
                    stats["json_no_participant"] += 1
                    continue

                parsed = extract_strokes(record)
                if parsed is None:
                    stats["json_unusable_record"] += 1
                    continue
                x, y, t, p = parsed

                by_participant[pid].append(RawTrial(
                    subject_id=f"json::{pid}", label=0, source="app",
                    device="touchscreen", test_id=-1, task=task_name,
                    origin=str(rel),
                    x=x, y=y, z=np.full(len(x), np.nan), pressure=p, t=t,
                    has_real_pressure=bool(np.isfinite(p).any()),
                    synthetic_time=False,
                ))
                stats["json_trials"] += 1
        except Exception as exc:  # noqa: BLE001
            stats["json_errors"] += 1
            stats[f"json_error::{type(exc).__name__}"] += 1
            continue

    # Task filtering: prefer spirals so app kinematics match the UCI spirals.
    mode = cfg.app_tasks
    if mode == "auto":
        spiralish = [tname for tname in seen_tasks if "spiral" in tname]
        mode = "spiral" if spiralish else "all"
        stats["task_filter_resolved"] = mode
    if mode == "spiral":
        keep = [tname for tname in seen_tasks if "spiral" in tname]
    else:
        keep = list(seen_tasks)
    keep_set = set(keep)

    for pid, items in by_participant.items():
        selected = [tr for tr in items if tr.task in keep_set]
        stats["json_trials_dropped_by_task"] += len(items) - len(selected)
        if not selected:
            continue
        if cfg.max_trials_per_subject:
            selected = selected[: cfg.max_trials_per_subject]
        trials.extend(selected)

    stats["json_participants"] = len({tr.subject_id for tr in trials})
    stats["json_task_counts"] = dict(seen_tasks)
    return trials


# ===========================================================================
# 3. Feature builder / Xây đặc trưng (6 kênh, L điểm)
# ===========================================================================
def _resample_channel(values: np.ndarray, src_grid: np.ndarray, dst_grid: np.ndarray) -> np.ndarray:
    """Spline (cubic when possible, linear fallback) resampling of one channel."""
    finite = np.isfinite(values) & np.isfinite(src_grid)
    if finite.sum() < 4:
        return np.interp(dst_grid, src_grid, np.nan_to_num(values))
    v = values[finite]
    g = src_grid[finite]
    order = np.argsort(g, kind="stable")
    g, v = g[order], v[order]
    keep = np.concatenate([[True], np.diff(g) > 1e-12])
    g, v = g[keep], v[keep]
    if g.size < 4 or not HAS_SCIPY:
        return np.interp(dst_grid, g, v)
    try:
        return CubicSpline(g, v, extrapolate=True)(dst_grid)
    except Exception:  # noqa: BLE001
        return np.interp(dst_grid, g, v)


def virtual_pressure_proxy(velocity: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """Deterministic stand-in for stylus pressure on touch devices.

        proxy = 0.5 * (1 - v_norm) + 0.5 * r_norm

    Rationale / Lý do: slower strokes and larger excursions from the drawing
    centre are empirically associated with higher contact force. Both terms are
    robustly normalised inside the trial.

    HONEST LIMITATION: this is a pure function of geometry, so it adds NO new
    clinical information. Its only job is to keep channel 4 populated and on a
    comparable scale, so the network does not read "empty channel" as
    "touch device" (i.e. as "healthy").
    """
    def robust_norm(a: np.ndarray) -> np.ndarray:
        hi = np.nanpercentile(np.abs(a[np.isfinite(a)]), 95) if np.isfinite(a).any() else 0.0
        if not np.isfinite(hi) or hi <= 1e-12:
            return np.zeros_like(a)
        return np.clip(a / hi, 0.0, 1.0)

    v_n = robust_norm(velocity)
    r_n = robust_norm(radius)
    proxy = 0.5 * (1.0 - v_n) + 0.5 * r_n
    return np.clip(proxy, 0.0, 1.0)


def build_channels(trial: RawTrial, cfg: TrainConfig,
                   uci_pressure_ref: Optional[np.ndarray] = None,
                   uci_z_ref: Optional[np.ndarray] = None,
                   stats: Optional[Counter] = None) -> Optional[np.ndarray]:
    """Build the NATIVE-resolution (N_raw, 6[+1]) matrix for one trial.

    v4 change: v3 resampled straight to ``seq_len`` here, which made one trial
    equal one row. Slicing has to happen first for a sliding window to mean
    anything, so this now returns the full-length matrix and the windowing /
    resampling happen in :func:`build_dataset`, right after slicing.
    """
    x, y, t = trial.x, trial.y, trial.t

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(t)
    if finite.sum() < MIN_SAMPLES_PER_TRIAL:
        return None
    x, y, t = x[finite], y[finite], t[finite]
    z = trial.z[finite] if trial.z.shape == trial.x.shape else np.full(finite.sum(), np.nan)
    p = trial.pressure[finite] if trial.pressure.shape == trial.x.shape else np.full(finite.sum(), np.nan)

    # ---- dt + velocity on the ORIGINAL sampling grid ----
    dt = np.diff(t, prepend=t[0])
    dt[0] = dt[1] if dt.size > 1 else 1.0 / 60.0
    # A fixed 1e-4 floor is dangerous: on an 8 ms grid a single zero/duplicate
    # timestamp then inflates velocity 80x, and the network happily learns that
    # spike. Repair non-advancing samples with the median step first, then clamp
    # to a sane fraction of that step (a real sampling grid does not vary 20x).
    pos = dt[dt > 0]
    if pos.size == 0:
        return None
    med_dt = float(np.median(pos))
    if med_dt <= 0:
        return None
    n_bad_dt = int((dt <= 0).sum())
    if n_bad_dt and stats is not None:
        stats["dt_repaired_samples"] += n_bad_dt
    dt = np.where(dt > 0, dt, med_dt)
    dt = np.clip(dt, max(1e-6, 0.05 * med_dt), None)

    vx = np.diff(x, prepend=x[0]) / dt
    vy = np.diff(y, prepend=y[0]) / dt
    vx[0], vy[0] = vx[1], vy[1]
    velocity = np.hypot(vx, vy)

    # ---- Z channel: fill BEFORE the geometry step ----
    # Z is the Wacom pen height; touch capture has no equivalent at all. The
    # fill must happen *before* centring/scaling so the constant then passes
    # through the same normalisation as real UCI Z and both sources land on one
    # numeric frame. Filling afterwards would leave app Z at the raw UCI median
    # (~12) while UCI Z sits near 0 - a device flag, not a neutralisation.
    if not np.isfinite(z).any():
        if cfg.z_fill == "median" and uci_z_ref is not None and uci_z_ref.size:
            z = np.full(z.shape, float(np.nanmedian(uci_z_ref)))
            if stats is not None:
                stats["z_filled_from_uci_median"] += 1
        else:
            z = np.zeros(z.shape)
            if stats is not None:
                stats["z_filled_zero"] += 1

    # ---- geometry normalisation (device-invariant scale) ----
    if cfg.geometry == "center_scale":
        cx, cy = float(np.median(x)), float(np.median(y))
        rad = np.hypot(x - cx, y - cy)
        scale = float(np.nanpercentile(rad, 95))
        if not np.isfinite(scale) or scale <= 1e-9:
            scale = 1.0
        x = (x - cx) / scale
        y = (y - cy) / scale
        z = (z - float(np.nanmedian(z) if np.isfinite(z).any() else 0.0)) / scale
        velocity = velocity / scale
        radius = rad / scale
    else:
        radius = np.hypot(x - np.median(x), y - np.median(y))

    # ---- pressure channel ----
    # A present-but-constant pressure column (many capture apps write a
    # placeholder such as "p": 0.5 on every point) must NOT count as real
    # pressure: it would disable the proxy and hand the network a channel that
    # is flat on touch and variable on the tablet - a perfect device flag.
    has_p = bool(trial.has_real_pressure) and np.isfinite(p).any()
    if has_p and is_degenerate_channel(p, min_unique=3):
        has_p = False
        if stats is not None:
            stats["pressure_placeholder_detected"] += 1
    if has_p:
        pressure = np.nan_to_num(p, nan=float(np.nanmedian(p[np.isfinite(p)])))
    else:
        proxy01 = virtual_pressure_proxy(velocity, radius)
        if cfg.pressure_mapping == "quantile" and uci_pressure_ref is not None and uci_pressure_ref.size:
            qs = np.linspace(0.0, 1.0, min(512, max(16, uci_pressure_ref.size)))
            ref_q = np.quantile(uci_pressure_ref, qs)
            pressure = np.interp(proxy01, qs, ref_q)
        else:
            pressure = proxy01
        if stats is not None:
            stats["pressure_proxy_used"] += 1

    # v4: NO resampling here — hand back the native-resolution matrix so the
    # caller can cut several overlapping windows out of it.
    mat = np.stack([x, y, z, pressure, velocity, dt], axis=1).astype(np.float32)
    if cfg.append_duration_channel:
        duration = np.full((mat.shape[0], 1), float(t[-1] - t[0]), dtype=np.float32)
        mat = np.concatenate([mat, duration], axis=1)

    return mat if np.isfinite(mat).all() else None


# ===========================================================================
# 3b. Sliding windows, resampling and rotation / Cửa sổ trượt & xoay
# ===========================================================================
def slice_windows(mat: np.ndarray, window: int, stride: int,
                  min_len: int) -> List[np.ndarray]:
    """Cut overlapping windows of ``window`` samples every ``stride`` samples.

    The last window is always emitted even when it does not land exactly on the
    stride, so the tail of a recording is never silently discarded. A recording
    shorter than the window yields exactly one (whole-trial) window rather than
    nothing — throwing away short trials would bias the sample towards people
    who draw slowly.
    """
    n = int(mat.shape[0])
    if n < min_len:
        return []
    if window <= 0 or stride <= 0 or n <= window:
        return [mat]
    starts = list(range(0, n - window + 1, stride))
    tail = n - window
    if not starts or starts[-1] != tail:
        starts.append(tail)
    return [mat[s:s + window] for s in starts]


def resample_matrix(mat: np.ndarray, L: int, cfg: TrainConfig) -> np.ndarray:
    """Resample one window onto a uniform grid of ``L`` points."""
    n, c = mat.shape
    if n == L:
        return mat.astype(np.float32)
    if cfg.resample_by == "arclength":
        seg = np.hypot(np.diff(mat[:, 0], prepend=mat[0, 0]),
                       np.diff(mat[:, 1], prepend=mat[0, 1]))
        seg[0] = 0.0
        grid = np.cumsum(seg)
        if grid[-1] <= 1e-12:
            grid = np.arange(n, dtype=float)
    else:
        grid = np.arange(n, dtype=float)
    grid = grid / (grid[-1] if grid[-1] > 0 else 1.0)
    dst = np.linspace(0.0, 1.0, L)
    out = np.stack([_resample_channel(mat[:, j], grid, dst) for j in range(c)], axis=1)
    return out.astype(np.float32)


def rotate_xy(X: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate the (x, y) channels by ``angle_deg`` about the drawing centre.

        x_rot = x*cos(theta) - y*sin(theta)
        y_rot = x*sin(theta) + y*cos(theta)

    Only channels 0 and 1 are touched. dt, velocity, pressure and Z are
    scalars, so the rotation leaves them exactly as they are — which is both
    what the spec asks for and what is physically correct: the stored velocity
    is a speed magnitude ``hypot(vx, vy)``, and a magnitude is invariant under
    rotation, so there is nothing to correct.
    """
    theta = math.radians(float(angle_deg))
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    out = X.copy()
    # NOTE: these MUST be copies. ``out[:, 0, :]`` is a view, so writing
    # the rotated x first would otherwise make the y line below read the
    # ALREADY-ROTATED x and shear the drawing instead of rotating it.
    x = out[:, 0, :].copy()
    y = out[:, 1, :].copy()
    out[:, 0, :] = x * cos_t - y * sin_t
    out[:, 1, :] = x * sin_t + y * cos_t
    return out


def augment_with_rotations(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                           angles: Sequence[float]
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Append rotated copies of every training row.

    The copy inherits its parent's ``subject_id``. That is the whole leakage
    argument: because GroupKFold groups on ``subject_id``, a rotated twin can
    never land in a different fold from its original. Called on the TRAINING
    split only — validation and holdout data are never rotated.
    """
    if not angles:
        return X, y, groups
    parts = [X]
    for ang in angles:
        parts.append(rotate_xy(X, float(ang)))
    X_out = np.concatenate(parts, axis=0)
    y_out = np.tile(np.asarray(y), len(parts))
    g_out = np.tile(np.asarray(groups), len(parts))
    return X_out, y_out, g_out


def build_dataset(cfg: TrainConfig) -> Tuple[Dict[str, Any], Counter]:
    """Load both sources, build channel matrices, return the tensor bundle."""
    stats: Counter = Counter()
    uci_root, json_root, _ = cfg.resolved_paths()

    uci_trials = load_uci_trials(uci_root, cfg, stats)
    app_trials = load_app_trials(json_root, cfg, stats)
    all_trials = uci_trials + app_trials

    if not all_trials:
        raise RuntimeError("No usable trials found — check --uci-root / --json-root.")

    # ---- cross-device calibration references (fit on UCI only) ----
    uci_pressures = np.concatenate(
        [tr.pressure[np.isfinite(tr.pressure)] for tr in uci_trials
         if tr.has_real_pressure and np.isfinite(tr.pressure).any()]
    ) if any(tr.has_real_pressure for tr in uci_trials) else np.array([])
    uci_zs = np.concatenate(
        [tr.z[np.isfinite(tr.z)] for tr in uci_trials if np.isfinite(tr.z).any()]
    ) if any(np.isfinite(tr.z).any() for tr in uci_trials) else np.array([])
    stats["uci_pressure_ref_n"] = int(uci_pressures.size)
    stats["uci_z_ref_n"] = int(uci_zs.size)

    # ---- per-subject cap (keeps one prolific person from dominating) ----
    if cfg.max_trials_per_subject:
        by_sub: Dict[str, List[RawTrial]] = defaultdict(list)
        for tr in all_trials:
            by_sub[tr.subject_id].append(tr)
        all_trials = [tr for items in by_sub.values() for tr in items[: cfg.max_trials_per_subject]]

    subjects = sorted({tr.subject_id for tr in all_trials})
    if cfg.limit_subjects:
        subjects = subjects[: cfg.limit_subjects]
        keep = set(subjects)
        all_trials = [tr for tr in all_trials if tr.subject_id in keep]

    n_ch = N_CHANNELS + (1 if cfg.append_duration_channel else 0)
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    groups: List[str] = []
    sources: List[int] = []
    meta: List[Dict[str, Any]] = []

    n_windows_per_subject: Counter = Counter()

    for tr in all_trials:
        mat = build_channels(tr, cfg, uci_pressures, uci_zs, stats)
        if mat is None:
            stats["trials_dropped_in_features"] += 1
            continue
        stats["trials_kept"] += 1

        # ---- v4: one recording -> many overlapping windows ----
        windows = slice_windows(mat, cfg.window, cfg.stride, MIN_SAMPLES_PER_TRIAL)
        if not windows:
            stats["windows_dropped_short"] += 1
            continue

        trial_duration = float(tr.t[-1] - tr.t[0]) if tr.t.size > 1 else 0.0
        total_raw = int(mat.shape[0])
        for wi, win in enumerate(windows):
            win_L = resample_matrix(win, cfg.seq_len, cfg)
            if win_L is None or not np.isfinite(win_L).all():
                stats["windows_dropped_bad"] += 1
                continue
            # A window covers part of the recording; scale its duration by the
            # fraction of samples it spans so the timing channel stays honest.
            frac = (win.shape[0] / total_raw) if total_raw else 1.0
            X_list.append(win_L)
            y_list.append(tr.label)
            groups.append(tr.subject_id)
            sources.append(0 if tr.source == "uci" else 1)
            meta.append({
                "subject_id": tr.subject_id, "source": tr.source, "device": tr.device,
                "test_id": tr.test_id, "task": tr.task, "origin": tr.origin,
                "n_samples": int(win.shape[0]),
                "duration_s": float(trial_duration * frac),
                "synthetic_time": tr.synthetic_time,
                "window_index": int(wi),
                "n_windows_in_trial": int(len(windows)),
            })
            n_windows_per_subject[tr.subject_id] += 1
            stats["windows_kept"] += 1

    if not X_list:
        raise RuntimeError("Feature building produced zero trials.")

    X = np.stack(X_list).astype(np.float32)              # (N, L, C)
    X = np.transpose(X, (0, 2, 1))                       # (N, C, L) for Conv1d
    y = np.asarray(y_list, dtype=np.float32)
    groups_arr = np.asarray(groups)
    sources_arr = np.asarray(sources, dtype=int)

    duration_arr = np.asarray([float(md.get("duration_s", 0.0)) for md in meta],
                              dtype=np.float64)
    dt_mean_arr = np.asarray([float(np.mean(mt[FEATURE_NAMES.index("dt")]))
                              for mt in X], dtype=np.float64)

    bundle: Dict[str, Any] = {
        "X": X, "y": y, "groups": groups_arr, "sources": sources_arr,
        "meta": meta, "n_channels": n_ch,
        "feature_names": list(FEATURE_NAMES) + (["duration"] if cfg.append_duration_channel else []),
        "uci_pressure_ref": uci_pressures,
        "uci_z_ref": uci_zs,
        "duration": duration_arr,
        "dt_mean": dt_mean_arr,
    }
    stats["subjects"] = len(set(groups))
    stats["windows_per_subject_min"] = int(min(n_windows_per_subject.values())) \
        if n_windows_per_subject else 0
    stats["windows_per_subject_max"] = int(max(n_windows_per_subject.values())) \
        if n_windows_per_subject else 0
    stats["windows_per_subject_median"] = float(np.median(list(n_windows_per_subject.values()))) \
        if n_windows_per_subject else 0.0
    return bundle, stats


# ===========================================================================
# 4. Subject-grouped splits / Chia tập theo người
# ===========================================================================
def subject_table(groups: np.ndarray, y: np.ndarray, sources: np.ndarray) -> Dict[str, Dict[str, Any]]:
    table: Dict[str, Dict[str, Any]] = {}
    for sid in sorted(set(groups)):
        m = groups == sid
        table[sid] = {
            "label": int(round(float(np.mean(y[m])))),
            "source": int(round(float(np.mean(sources[m])))),
            "n_trials": int(m.sum()),
        }
    return table


def make_holdout(table: Dict[str, Dict[str, Any]], frac: float, seed: int) -> List[str]:
    """Remove ~frac of subjects, stratified by (source, label). Deterministic."""
    rng = random.Random(seed)
    strata: Dict[Tuple[int, int], List[str]] = defaultdict(list)
    for sid, info in table.items():
        strata[(info["source"], info["label"])].append(sid)

    holdout: List[str] = []
    for key in sorted(strata):
        members = sorted(strata[key])
        rng.shuffle(members)
        n = len(members)
        if n >= 5:
            k = max(1, min(n - 2, int(round(frac * n))))
        elif n >= 3:
            k = 1
        else:
            k = 0                      # too few to spare; keep them for CV
        holdout.extend(members[:k])
    return sorted(holdout)


def make_folds(table: Dict[str, Dict[str, Any]], subjects: Sequence[str],
               n_splits: int, seed: int) -> List[Tuple[List[str], List[str]]]:
    """GroupKFold on subjects, with a fallback if a class is too thin."""
    subjects = sorted(subjects)
    y = np.array([table[s]["label"] for s in subjects])
    groups = np.array(subjects)

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    effective = min(n_splits, n_pos if n_pos else n_splits, n_neg if n_neg else n_splits,
                    len(subjects))
    if effective < 2:
        return []

    folds: List[Tuple[List[str], List[str]]] = []
    gkf = GroupKFold(n_splits=effective)
    for tr_idx, va_idx in gkf.split(groups, y, groups=groups):
        folds.append((list(groups[tr_idx]), list(groups[va_idx])))
    return folds


# ===========================================================================
# 5. Model / Mô hình
# ===========================================================================
if HAS_TORCH:
    class NeuroTraceHybridNet(nn.Module):
        """Conv1d-BN-Pool-Dropout x2 -> BiLSTM -> MLP -> 1 logit.

        Spec: Conv1d(6,64) -> BN -> MaxPool -> Dropout(0.3)
              Conv1d(64,128) -> BN -> MaxPool -> Dropout(0.3)
              BiLSTM(128, hidden=64, layers=2, bidirectional=True)
              Dense(128 -> 64 -> 1) with sigmoid at INFERENCE time only.
        """

        def __init__(self, in_channels: int = N_CHANNELS, conv1: int = 64, conv2: int = 128,
                     lstm_hidden: int = 64, lstm_layers: int = 2, dropout: float = 0.3,
                     dense1: int = 128, dense2: int = 64):
            super().__init__()
            self.block1 = nn.Sequential(
                nn.Conv1d(in_channels, conv1, kernel_size=5, padding=2),
                nn.BatchNorm1d(conv1), nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2), nn.Dropout(dropout),
            )
            self.block2 = nn.Sequential(
                nn.Conv1d(conv1, conv2, kernel_size=3, padding=1),
                nn.BatchNorm1d(conv2), nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2), nn.Dropout(dropout),
            )
            self.lstm = nn.LSTM(
                input_size=conv2, hidden_size=lstm_hidden, num_layers=lstm_layers,
                batch_first=True, bidirectional=True,
                dropout=dropout if lstm_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.Linear(lstm_hidden * 2, dense1), nn.ReLU(inplace=True), nn.Dropout(dropout),
                nn.Linear(dense1, dense2), nn.ReLU(inplace=True), nn.Dropout(dropout),
                nn.Linear(dense2, 1),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """x: (B, C, L) -> logits: (B,). RAW LOGITS — pair with BCEWithLogitsLoss."""
            h = self.block1(x)                 # (B, 64, L/2)
            h = self.block2(h)                 # (B, 128, L/4)
            h = h.transpose(1, 2)              # (B, L/4, 128) for the LSTM
            out, _ = self.lstm(h)              # (B, L/4, 2*hidden)
            last = out[:, -1, :]               # final timestep
            return self.head(last).squeeze(-1)

        @torch.no_grad()
        def predict_proba(self, x: "torch.Tensor") -> np.ndarray:
            """Sigmoid lives HERE, not in forward() — used for metrics/inference."""
            self.eval()
            logits = self.forward(x)
            return torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64)


# ===========================================================================
# 6. Scaling, torch plumbing / Chuẩn hoá & tiện ích torch
# ===========================================================================
def fit_scaler(X: np.ndarray) -> StandardScaler:
    """Fit on (N, C, L) training trials only -> per-channel StandardScaler."""
    n, c, l = X.shape
    flat = np.transpose(X, (0, 2, 1)).reshape(-1, c)
    scaler = StandardScaler()
    scaler.fit(flat)
    return scaler


def apply_scaler(scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    n, c, l = X.shape
    flat = np.transpose(X, (0, 2, 1)).reshape(-1, c)
    out = scaler.transform(flat).reshape(n, l, c)
    return np.transpose(out, (0, 2, 1)).astype(np.float32)


def to_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, drop_last=False)


@torch.no_grad()
def predict_proba_batched(model: "nn.Module", X: np.ndarray, device, batch_size: int = 128) -> np.ndarray:
    model.eval()
    out: List[np.ndarray] = []
    for i in range(0, len(X), batch_size):
        batch = torch.from_numpy(X[i:i + batch_size]).to(device)
        logits = model(batch)
        out.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(out).astype(np.float64) if out else np.array([])


# ===========================================================================
# 7. Subject-level metrics / Đo lường cấp người
# ===========================================================================
def aggregate_by_subject(probs: np.ndarray, groups: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mean trial probability per subject — the unit of a screening claim."""
    order = sorted(set(groups))
    index = {s: i for i, s in enumerate(order)}
    sums = np.zeros(len(order))
    counts = np.zeros(len(order))
    for p, g in zip(probs, groups):
        sums[index[g]] += p
        counts[index[g]] += 1
    return np.array(order), sums / np.maximum(counts, 1)


def safe_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC that returns NaN (never 0.0) when a class is missing.

    A single-class split has no ranking to score; reporting 0.0 there would
    silently drag the mean AUC down and look like real evidence of failure.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return float("nan")


def fmt_auc(v: float) -> str:
    return "  n/a" if v is None or math.isnan(v) else f"{v:6.3f}"


def subject_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """Accuracy / F1 / sensitivity / specificity / ROC-AUC at subject level."""
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=float)
    out: Dict[str, Any] = {"n_subjects": int(len(y_true)),
                           "n_pos": int((y_true == 1).sum()),
                           "n_neg": int((y_true == 0).sum())}
    if len(y_true) == 0:
        return out

    preds = (probs >= threshold).astype(int)
    out["accuracy"] = float(accuracy_score(y_true, preds))
    out["f1"] = float(f1_score(y_true, preds, zero_division=0))
    out["precision"] = float(accuracy_score(y_true, preds))  # placeholder, replaced below

    tn = fp = fn = tp = 0
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    out["tn"], out["fp"], out["fn"], out["tp"] = int(tn), int(fp), int(fn), int(tp)
    out["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) else float("nan")
    out["specificity"] = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    out["precision"] = float(tp / (tp + fp)) if (tp + fp) else float("nan")

    if out["n_pos"] > 0 and out["n_neg"] > 0:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, probs))
        except Exception:
            out["roc_auc"] = float("nan")
        try:
            out["average_precision"] = float(average_precision_score(y_true, probs))
        except Exception:
            out["average_precision"] = float("nan")
    else:
        out["roc_auc"] = float("nan")       # genuinely undefined, not zero
        out["average_precision"] = float("nan")

    # best-F1 threshold (fitted on THIS evaluation set only — never on test)
    if out["n_pos"] > 0 and out["n_neg"] > 0:
        grid = np.unique(np.round(np.linspace(0.05, 0.95, 91), 3))
        best_t, best_f1 = threshold, -1.0
        for t in grid:
            f1 = f1_score(y_true, (probs >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = float(f1), float(t)
        out["best_f1_threshold"] = best_t
        out["f1_at_best_threshold"] = best_f1
    return out


def fmt_metrics(m: Dict[str, Any]) -> str:
    def f(k: str) -> str:
        v = m.get(k, float("nan"))
        return "  n/a " if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:6.3f}"
    return (f"acc {f('accuracy')} | f1 {f('f1')} | sens {f('sensitivity')} | "
            f"spec {f('specificity')} | auc {f('roc_auc')}")


# ===========================================================================
# 8. Training / Huấn luyện
# ===========================================================================
def train_model(cfg: TrainConfig, X_tr: np.ndarray, y_tr: np.ndarray,
                X_va: Optional[np.ndarray], y_va: Optional[np.ndarray],
                groups_va: Optional[np.ndarray], device,
                tag: str = "", epochs_override: Optional[int] = None,
                scaler: Optional[StandardScaler] = None,
                groups_tr: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Train one model. Returns state, scaler, history and subject-level metrics.

    v4: the training split is expanded with rotated copies AFTER the scaler is
    fitted and applied. The scaler therefore sees only real geometry, while the
    network sees the augmented set. Validation data is never rotated.
    """
    if scaler is None:
        scaler = fit_scaler(X_tr)
    X_tr_s = apply_scaler(scaler, X_tr)
    X_va_s = apply_scaler(scaler, X_va) if X_va is not None else None

    # ---- rotation augmentation (training rows only) ----
    n_before = int(len(X_tr_s))
    if cfg.augment_train and cfg.rotation_angles and groups_tr is not None:
        X_tr_s, y_tr, tr_groups = augment_with_rotations(
            X_tr_s, y_tr, groups_tr, cfg.rotation_angles)
    else:
        tr_groups = groups_tr
    n_after = int(len(X_tr_s))
    if cfg.verbose and n_after != n_before:
        ang_txt = ", ".join("{:+g}\u00b0".format(a) for a in cfg.rotation_angles)
        print("      [{}] augmentation: {} -> {} training rows "
              "(x{:.0f}, rotations {})".format(
                  tag, n_before, n_after, n_after / max(1, n_before), ang_txt))

    n_pos = float((y_tr == 1).sum())
    n_neg = float((y_tr == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32, device=device) \
        if n_pos > 0 else None

    model = NeuroTraceHybridNet(
        in_channels=X_tr.shape[1], conv1=cfg.conv1, conv2=cfg.conv2,
        lstm_hidden=cfg.lstm_hidden, lstm_layers=cfg.lstm_layers, dropout=cfg.dropout,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # v4 spec: AdamW + cosine annealing (v3 used Adam + ReduceLROnPlateau).
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(epochs_override or cfg.epochs))
    ) if cfg.use_scheduler else None

    loader = to_loader(X_tr_s, y_tr, cfg.batch_size, shuffle=True)

    epochs = int(epochs_override if epochs_override else cfg.epochs)
    history: List[Dict[str, float]] = []
    best = {"auc": -np.inf, "epoch": 0, "state": None, "metrics": {}}
    since_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            running += float(loss.item()) * len(yb)
            seen += len(yb)
        train_loss = running / max(seen, 1)

        row: Dict[str, float] = {"epoch": epoch, "train_loss": train_loss,
                                 "lr": float(optimizer.param_groups[0]["lr"])}
        if X_va_s is not None and len(X_va_s):
            probs = predict_proba_batched(model, X_va_s, device)
            _, subj_probs = aggregate_by_subject(probs, groups_va)
            va_subjects = np.array(sorted(set(groups_va)))
            va_labels = np.array([int(round(float(np.mean(
                y_va[groups_va == s])))) for s in va_subjects])
            met = subject_metrics(va_labels, subj_probs, cfg.threshold)
            auc = met.get("roc_auc", float("nan"))
            score = -1.0 if (auc is None or math.isnan(auc)) else auc
            row["val_auc"] = score
            row["val_acc"] = met.get("accuracy", float("nan"))
            if score > best["auc"]:
                best.update({"auc": score, "epoch": epoch,
                             "state": {k: v.detach().cpu().clone()
                                       for k, v in model.state_dict().items()},
                             "metrics": met})
                since_improve = 0
            else:
                since_improve += 1
        else:
            row["val_auc"] = float("nan")
            if epoch == epochs:
                best.update({"auc": float("nan"), "epoch": epoch,
                             "state": {k: v.detach().cpu().clone()
                                       for k, v in model.state_dict().items()}})
        # CosineAnnealingLR advances once per EPOCH on its own schedule; it is
        # not driven by the validation metric the way ReduceLROnPlateau was.
        if scheduler is not None:
            scheduler.step()
        history.append(row)

        if cfg.verbose and (epoch == 1 or epoch % 5 == 0 or epoch == epochs):
            msg = f"      [{tag}] epoch {epoch:>3}/{epochs}  loss {train_loss:.4f}"
            if not math.isnan(row.get("val_auc", float("nan"))):
                msg += f"  val_auc {row['val_auc']:.4f}"
            print(msg, flush=True)

        if X_va_s is not None and epoch >= cfg.min_epochs and since_improve >= cfg.early_stop_patience:
            if cfg.verbose:
                print(f"      [{tag}] early stop at epoch {epoch} "
                      f"(no val improvement for {since_improve} epochs)")
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])

    result: Dict[str, Any] = {
        "model": model, "scaler": scaler, "history": history,
        "best_epoch": best["epoch"], "best_val_auc": best["auc"],
        "best_val_metrics": best["metrics"],
        "pos_weight": float(pos_weight.item()) if pos_weight is not None else None,
        "n_train_rows_before_aug": n_before,
        "n_train_rows_after_aug": n_after,
        "rotation_angles": list(cfg.rotation_angles) if cfg.augment_train else [],
    }
    return result


def evaluate_model(model, scaler: StandardScaler, X: np.ndarray, y: np.ndarray,
                   groups: np.ndarray, device, threshold: float) -> Dict[str, Any]:
    probs = predict_proba_batched(model, apply_scaler(scaler, X), device)
    subjects, subj_probs = aggregate_by_subject(probs, groups)
    label_of = {}
    for s in subjects:
        label_of[s] = int(round(float(np.mean(y[groups == s]))))
    y_true = np.array([label_of[s] for s in subjects])
    met = subject_metrics(y_true, subj_probs, threshold)
    met["per_subject"] = {str(s): float(p) for s, p in zip(subjects, subj_probs)}
    met["per_subject_truth"] = {str(s): int(t) for s, t in zip(subjects, y_true)}
    return met


# ===========================================================================
# 9. Device-leakage probe / Dò rò rỉ thiết bị
# ===========================================================================
def device_leakage_probe(bundle: Dict[str, Any], table: Dict[str, Dict[str, Any]],
                         subjects: Sequence[str], seed: int) -> Dict[str, Any]:
    """Can a linear model tell UCI from app from the features alone?

    If yes, then "PD vs healthy" accuracy is partly "Wacom vs touchscreen"
    accuracy, and the clinical claim is not supported by this dataset.
    """
    X, y, groups, sources = bundle["X"], bundle["y"], bundle["groups"], bundle["sources"]
    mask = np.isin(groups, list(subjects))
    Xs, gsub, ssub = X[mask], groups[mask], sources[mask]

    if len(np.unique(ssub)) < 2:
        return {"status": "skipped", "reason": "only one device/source present"}

    # Compact per-trial summary: mean/std per channel + log duration proxy.
    feats = []
    for i in range(len(Xs)):
        mat = Xs[i]
        feats.append(np.concatenate([mat.mean(axis=1), mat.std(axis=1)]))
    F = np.vstack(feats)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)

    subj_list = sorted(set(gsub))
    y_sub = np.array([int(round(float(np.mean(ssub[gsub == s])))) for s in subj_list])
    folds = make_folds({s: {"label": int(l), "source": 0, "n_trials": 1}
                        for s, l in zip(subj_list, y_sub)}, subj_list, 5, seed)
    if not folds:
        return {"status": "skipped", "reason": "not enough subjects for grouped folds"}

    aucs: List[float] = []
    for tr_subj, va_subj in folds:
        tr = np.isin(gsub, tr_subj)
        va = np.isin(gsub, va_subj)
        if len(np.unique(ssub[tr])) < 2 or len(np.unique(ssub[va])) < 2:
            continue
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(F[tr], ssub[tr])
        p = clf.predict_proba(F[va])[:, 1]
        try:
            aucs.append(float(roc_auc_score(ssub[va], p)))
        except Exception:
            continue

    out: Dict[str, Any] = {"status": "ok" if aucs else "skipped",
                           "fold_aucs": aucs}
    if aucs:
        out["mean_auc"] = float(np.mean(aucs))
        out["std_auc"] = float(np.std(aucs))
    return out


# ===========================================================================
# 10. Reporting / Báo cáo
# ===========================================================================
def print_dataset_summary(bundle: Dict[str, Any], table: Dict[str, Dict[str, Any]],
                          stats: Counter, cfg: TrainConfig) -> None:
    section("DATASET / TẬP DỮ LIỆU")
    X, y, groups, sources = bundle["X"], bundle["y"], bundle["groups"], bundle["sources"]
    kv("windows (rows)", f"{human(len(y))}   shape {tuple(X.shape)}  (N, C, L)")
    kv("channels", ", ".join(bundle["feature_names"]))
    kv("sequence length L", X.shape[2])
    print("\n   augmentation / tăng cường dữ liệu:")
    kv("window / stride", f"{cfg.window} / {cfg.stride} samples")
    kv("windows per subject",
       f"median {stats.get('windows_per_subject_median', 0):.0f}  "
       f"(min {stats.get('windows_per_subject_min', 0)}, "
       f"max {stats.get('windows_per_subject_max', 0)})")
    kv("rotation angles (train only)",
       ", ".join(f"{a:+g}°" for a in cfg.rotation_angles) if cfg.augment_train
       else "disabled (--no-augment)")
    n_angles = len(cfg.rotation_angles) if cfg.augment_train else 0
    kv("train expansion", f"x{1 + n_angles}  (1 original + {n_angles} rotated)")
    total = stats.get("windows_kept", 0)
    aug_total = total * (1 + n_angles)
    kv("augmented windows (x{})".format(1 + n_angles),
       f"{human(aug_total)} samples  ({human(total)} sliced windows x {1 + n_angles})")
    flag = "OK" if aug_total >= cfg.min_samples else "BELOW TARGET"
    kv("total samples vs target",
       f"{human(aug_total)} / {human(cfg.min_samples)}  [{flag}]")

    per_source_label: Counter = Counter()
    trials_per_source_label: Counter = Counter()
    for sid, info in table.items():
        per_source_label[(info["source"], info["label"])] += 1
        trials_per_source_label[(info["source"], info["label"])] += info["n_trials"]

    print("\n   source      class      subjects   windows")
    print("   " + "-" * 46)
    for (src, lab), n in sorted(per_source_label.items()):
        src_name = "uci  (Wacom)" if src == 0 else "app  (touch)"
        lab_name = "PARKINSON" if lab == 1 else "healthy  "
        print(f"   {src_name:<12}{lab_name:<11}{human(n):>9}{human(trials_per_source_label[(src, lab)]):>10}")

    kv("total unique subjects", human(len(table)))
    kv("windows per subject (median)",
       f"{np.median([v['n_trials'] for v in table.values()]):.0f}  "
       f"(min {min(v['n_trials'] for v in table.values())}, "
       f"max {max(v['n_trials'] for v in table.values())})")

    print("\n   loader notes / ghi chú bộ nạp:")
    for key in sorted(k for k in stats if not k.startswith(("uci_error", "json_error", "json_task"))):
        if stats[key]:
            print(f"     {key:<38} {human(stats[key])}")
    if stats.get("json_task_counts"):
        print(f"     {'app task distribution':<38} {stats['json_task_counts']}")
    if stats.get("task_filter_resolved"):
        print(f"     {'app task filter resolved to':<38} {stats['task_filter_resolved']}")

    print("\n   [CONFOUND CHECK] PD subjects by source:")
    pd_by_src = Counter(table[s]["source"] for s in table if table[s]["label"] == 1)
    hc_by_src = Counter(table[s]["source"] for s in table if table[s]["label"] == 0)
    print(f"     PARKINSON : {dict(pd_by_src)}   (0=uci, 1=app)")
    print(f"     healthy   : {dict(hc_by_src)}")
    if pd_by_src and 1 not in pd_by_src:
        print("     !! EVERY Parkinson subject comes from UCI and no app subject has PD.")
        print("        Source and label are confounded — see section [7].")

    duration_report(bundle)


def duration_report(bundle: Dict[str, Any]) -> None:
    """Report how much drawing duration and sampling rate separate classes/devices.

    Resampling every trial to a fixed L makes the *shape* of a drawing
    comparable, but it also bakes timing into two places the 6-channel spec
    does not talk about:

      * the ``velocity`` channel keeps absolute speed (it is computed on the
        original sampling grid, in real seconds), and
      * the ``dt`` channel ends up nearly constant *within* a trial, so a
        single per-trial mean dt is effectively a duration / sampling-rate
        code that the network can read directly.

    Both are real signal (bradykinesia is a cardinal Parkinsonian sign) and
    both are also a device fingerprint (a 60 Hz touch screen vs a Wacom
    tablet). Reporting the numbers lets the reader judge which one dominates
    in their own data instead of guessing.
    """
    dur = bundle.get("duration")
    dtm = bundle.get("dt_mean")
    if dur is None:
        return
    y, groups, sources = bundle["y"], bundle["groups"], bundle["sources"]

    subj, ysub, dsub, ssub = [], [], [], []
    for sid in sorted(set(groups)):
        m = groups == sid
        subj.append(sid)
        ysub.append(int(round(float(np.mean(y[m])))))
        dsub.append(float(np.mean(dur[m])))
        ssub.append(int(sources[m][0]))
    ysub = np.asarray(ysub)
    dsub = np.asarray(dsub)
    ssub = np.asarray(ssub)

    print("\n   [TIMING CHECK] duration & sampling rate / thời gian & tần số lấy mẫu")
    for src, name in ((0, "uci  (Wacom)"), (1, "app  (touch)")):
        m = ssub == src
        if not m.any():
            continue
        pd_m, hc_m = m & (ysub == 1), m & (ysub == 0)
        pd_txt = f"{np.median(dsub[pd_m]):6.2f}s" if pd_m.any() else "    n/a"
        hc_txt = f"{np.median(dsub[hc_m]):6.2f}s" if hc_m.any() else "    n/a"
        trial_m = np.isin(groups, [subj[i] for i in range(len(subj)) if ssub[i] == src])
        dt_txt = f"{np.median(dtm[trial_m]) * 1000:5.1f}ms" if dtm is not None else "n/a"
        print(f"     {name:<12} median duration  PD {pd_txt}   healthy {hc_txt}"
              f"   | median sample interval {dt_txt}")

    auc = float(safe_roc_auc(ysub, dsub))
    print(f"     duration-only AUC, all subjects           {fmt_auc(auc)}")

    # Within one device only: this is the number that isolates the clinical
    # signal (bradykinesia) from the device effect, because both classes are
    # then recorded on the same hardware.
    for src, name in ((0, "uci  (Wacom)"), (1, "app  (touch)")):
        m = ssub == src
        if m.sum() < 4 or len(np.unique(ysub[m])) < 2:
            continue
        a = float(safe_roc_auc(ysub[m], dsub[m]))
        print(f"     duration-only AUC, {name} only        {fmt_auc(a)}")
        if src == 0 and not math.isnan(a):
            if a >= 0.80 or a <= 0.20:
                print("       -> duration separates PD from healthy *on the same device*.")
                print("          That is the bradykinesia signal, and it is trustworthy.")
                print("          Enable --append-duration-channel to give the network an")
                print("          explicit view of it instead of an implicit one.")
            elif 0.60 <= a < 0.80 or 0.20 < a <= 0.40:
                print("       -> moderate within-device duration signal.")

    # device signature carried by the dt channel
    a, b = dtm[sources == 0], dtm[sources == 1]
    if a.size and b.size:
        ratio = float(np.median(b) / max(np.median(a), 1e-12))
        print(f"     dt-channel medians  uci {np.median(a)*1000:.1f}ms  app {np.median(b)*1000:.1f}ms"
              f"  (ratio {ratio:.2f}x)")
        if ratio > 1.5 or ratio < 0.67:
            print("       -> the two devices sample at clearly different rates. Because the dt")
            print("          channel is near-constant within a trial, it is partly a DEVICE code.")
            print("          If the leakage probe in section [6] is high, try --geometry none")
            print("          plus dropping dt, or resample by --resample-by arclength.")


def verify_no_leakage(folds: List[Tuple[Any, Any]], holdout: set,
                      groups: np.ndarray) -> bool:
    """Prove the grouping claim instead of asserting it.

    Three independent checks, each of which can fail silently in a hand-rolled
    splitter: (1) no subject is both trained and validated inside a fold,
    (2) no holdout subject is trained on anywhere, and (3) every trial really
    is assigned to at most one side. A single shared subject between train and
    validation is enough to inflate the AUC, so this runs before any training.
    """
    seen_ids = {str(g) for g in groups}
    problems: List[str] = []

    def names(s: set, k: int = 3) -> str:
        return ", ".join(sorted(str(v) for v in s)[:k])

    for i, (tr_subj, va_subj) in enumerate(folds, start=1):
        tr, va = set(tr_subj), set(va_subj)
        if tr & va:
            problems.append(f"fold {i}: {len(tr & va)} subject(s) on both sides "
                            f"({names(tr & va)})")
        if va & holdout:
            problems.append(f"fold {i}: {len(va & holdout)} holdout subject(s) "
                            f"used for validation")
    all_train: set = set()
    for tr_subj, _ in folds:
        all_train |= set(tr_subj)
    if all_train & holdout:
        problems.append(f"{len(all_train & holdout)} holdout subject(s) leaked "
                        f"into training ({names(all_train & holdout)})")

    unknown = (all_train | set().union(*[set(v) for _, v in folds])) - seen_ids
    if unknown:
        problems.append(f"{len(unknown)} fold subject(s) not present in the data")

    print("\n   [LEAKAGE CHECK] subject-level split integrity / kiểm tra rò rỉ")
    if problems:
        for pr in problems:
            print(f"     FAIL  {pr}")
        return False
    n_val = sum(len(v) for _, v in folds)
    print(f"     OK  no subject on both sides of any fold "
          f"({len(folds)} folds, {n_val} validation slots)")
    print(f"     OK  {len(holdout)} holdout subject(s) absent from every "
          f"training fold")
    print(f"     OK  all {len(seen_ids)} subjects accounted for")
    return True


def audit_subjects(table: Dict[str, Dict[str, Any]], cfg: TrainConfig,
                   stats: Counter) -> Dict[str, Any]:
    """Verify the cohort is the one we think it is, before training on it.

    The point of a hard check rather than a printout: an extra duplicated
    participant or a mis-parsed subject code silently changes every number
    downstream, and a GroupKFold run will happily produce a plausible-looking
    AUC either way.
    """
    section("[0] SUBJECT AUDIT / KIỂM TOÁN NGƯỜI THAM GIA")
    n = len(table)
    n_pd = sum(1 for v in table.values() if v["label"] == 1)
    n_hc = n - n_pd
    by_src = Counter(v["source"] for v in table.values())
    pd_by_src = Counter(v["source"] for v in table.values() if v["label"] == 1)

    print("\n   cohort / đoàn hệ:")
    kv("unique subjects (found)", human(n))
    kv("unique subjects (expected)", human(cfg.expected_subjects))
    kv("Parkinson (found / expected)", f"{human(n_pd)} / {human(cfg.expected_pd)}")
    kv("healthy   (found / expected)", f"{human(n_hc)} / {human(cfg.expected_hc)}")
    print("\n   by source / theo nguồn:")
    for src, name in ((0, "uci  (Wacom, data/uci_raw)"),
                      (1, "app  (touch, data/healthy)")):
        k = by_src.get(src, 0)
        lab = sum(1 for v in table.values() if v["source"] == src and v["label"] == 1)
        print(f"     {name:<28} {human(k):>6} subjects   ({human(lab)} PD, "
              f"{human(k - lab)} HC)")

    checks = {
        "total_subjects": (n, cfg.expected_subjects),
        "parkinson": (n_pd, cfg.expected_pd),
        "healthy": (n_hc, cfg.expected_hc),
    }
    print("\n   checks / kiểm tra:")
    all_ok = True
    for name, (got, want) in checks.items():
        ok = (got == want)
        all_ok = all_ok and ok
        print(f"     [{'PASS' if ok else 'FAIL'}]  {name:<16} "
              f"found {human(got):>6}   expected {human(want):>6}")

    # duplicate / impossible situations that the count alone would hide
    if n_pd == 0 or n_hc == 0:
        print("     [FAIL]  one class is empty — nothing can be learned")
        all_ok = False
    if pd_by_src and 1 in pd_by_src:
        print(f"     [note]  {pd_by_src[1]} Parkinson subject(s) come from the app "
              f"device — good, that breaks the device confound")
    elif pd_by_src:
        print("     [note]  every Parkinson subject is on the Wacom tablet; the")
        print("             PD/healthy boundary is also a device boundary")

    result = {
        "n_subjects": n, "n_pd": n_pd, "n_hc": n_hc,
        "expected": {"subjects": cfg.expected_subjects,
                     "pd": cfg.expected_pd, "hc": cfg.expected_hc},
        "by_source": {("uci" if k == 0 else "app"): v for k, v in by_src.items()},
        "checks_passed": bool(all_ok),
    }
    verdict = ("AUDIT OK - the cohort matches the expected 92 subjects / 27 PD / 65 HC"
               if all_ok else
               "MISMATCH - inspect the loader notes above before trusting any metric")
    print("\n   verdict: " + verdict)
    return result


def print_fold_table(rows: List[Dict[str, Any]]) -> None:
    print("\n   fold   n_subj   acc     f1      sens    spec    auc     best_ep")
    print("   " + "-" * 66)
    for r in rows:
        def f(k):
            v = r["metrics"].get(k, float("nan"))
            return "  n/a" if v is None or math.isnan(v) else f"{v:6.3f}"
        print(f"   {r['fold']:>4}   {r['n_val_subjects']:>7}  {f('accuracy')}  {f('f1')}  "
              f"{f('sensitivity')}  {f('specificity')}  {f('roc_auc')}   {r['best_epoch']:>7}")

    def agg(key: str) -> str:
        vals = [r["metrics"].get(key, float("nan")) for r in rows]
        vals = [v for v in vals if v is not None and not math.isnan(v)]
        if not vals:
            return "n/a (undefined: a fold had only one class)"
        return f"{np.mean(vals):.3f} ± {np.std(vals):.3f}"
    print("   " + "-" * 66)
    print(f"   mean ± std (subject level, {len(rows)} folds)")
    for key in ("accuracy", "f1", "sensitivity", "specificity", "roc_auc"):
        print(f"     {key:<14} {agg(key)}")


# ===========================================================================
# 11. Main / Hàm chính
# ===========================================================================
def parse_angles(raw: str) -> Tuple[float, ...]:
    """'-30,-15,15,30' -> (-30.0, -15.0, 15.0, 30.0). Empty input disables."""
    out: List[float] = []
    for part in str(raw).split(","):
        part = part.strip().rstrip("dD\u00b0").replace("deg", "").strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except ValueError:
            continue
    return tuple(out)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="NeuroTrace v4 — subject-grouped hybrid CNN+BiLSTM trainer with "
                    "physics-preserving augmentation (research prototype, not a "
                    "medical device).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    d = TrainConfig()
    p.add_argument("--project-root", default=d.project_root)
    p.add_argument("--uci-root", default=None)
    p.add_argument("--json-root", default=None)
    p.add_argument("--model-dir", default=None)
    p.add_argument("--cache-path", default=None, help="npz cache for the built tensors")

    p.add_argument("--seq-len", type=int, default=d.seq_len, choices=[128, 256, 512])
    p.add_argument("--resample-by", default=d.resample_by, choices=["time", "arclength"])
    p.add_argument("--geometry", default=d.geometry, choices=["center_scale", "none"])
    p.add_argument("--z-fill", default=d.z_fill, choices=["median", "zero"])
    p.add_argument("--pressure-mapping", default=d.pressure_mapping, choices=["quantile", "none"])
    p.add_argument("--append-duration-channel", action="store_true")
    p.add_argument("--app-tasks", default=d.app_tasks, choices=["auto", "spiral", "all"])
    p.add_argument("--max-trials-per-subject", type=int, default=d.max_trials_per_subject)

    p.add_argument("--window", type=int, default=d.window,
                   help="sliding window length in raw samples")
    p.add_argument("--stride", type=int, default=d.stride,
                   help="sliding window stride in raw samples")
    p.add_argument("--rotation-angles", default=",".join(str(a) for a in d.rotation_angles),
                   help="comma-separated in-plane rotations in degrees. Use the "
                        "EQUALS form because the first angle is negative: "
                        "--rotation-angles=-30,-15,15,30")
    p.add_argument("--no-augment", action="store_true",
                   help="disable window slicing + rotation augmentation")
    p.add_argument("--min-samples", type=int, default=d.min_samples,
                   help="target number of augmented sequences")

    p.add_argument("--conv1", type=int, default=d.conv1)
    p.add_argument("--conv2", type=int, default=d.conv2)
    p.add_argument("--lstm-hidden", type=int, default=d.lstm_hidden)
    p.add_argument("--lstm-layers", type=int, default=d.lstm_layers)
    p.add_argument("--dropout", type=float, default=d.dropout)

    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    p.add_argument("--no-scheduler", action="store_true")

    p.add_argument("--n-splits", type=int, default=d.n_splits)
    p.add_argument("--holdout-frac", type=float, default=d.holdout_frac)
    p.add_argument("--threshold", type=float, default=d.threshold)

    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--num-workers", type=int, default=d.num_workers)
    p.add_argument("--limit-subjects", type=int, default=d.limit_subjects)
    p.add_argument("--expected-subjects", type=int, default=d.expected_subjects,
                   help="audit: unique participants the cohort must contain")
    p.add_argument("--expected-pd", type=int, default=d.expected_pd,
                   help="audit: expected Parkinson participants")
    p.add_argument("--expected-hc", type=int, default=d.expected_hc,
                   help="audit: expected healthy control participants")
    p.add_argument("--audit-only", action="store_true", help="build data, print summary, exit")
    p.add_argument("--device", default=d.device, choices=["auto", "cpu", "cuda"])
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    cfg = TrainConfig(
        project_root=args.project_root,
        uci_root=args.uci_root or "",
        json_root=args.json_root or "",
        model_dir=args.model_dir or "",
        cache_path=args.cache_path or "",
        seq_len=args.seq_len, resample_by=args.resample_by, geometry=args.geometry,
        z_fill=args.z_fill, pressure_mapping=args.pressure_mapping,
        append_duration_channel=args.append_duration_channel,
        app_tasks=args.app_tasks, max_trials_per_subject=args.max_trials_per_subject,
        conv1=args.conv1, conv2=args.conv2, lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers, dropout=args.dropout,
        batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
        weight_decay=args.weight_decay, use_scheduler=not args.no_scheduler,
        n_splits=args.n_splits, holdout_frac=args.holdout_frac, threshold=args.threshold,
        seed=args.seed, num_workers=args.num_workers,
        limit_subjects=args.limit_subjects, audit_only=args.audit_only,
        window=args.window, stride=args.stride,
        rotation_angles=parse_angles(args.rotation_angles),
        augment_train=not args.no_augment, min_samples=args.min_samples,
        expected_subjects=args.expected_subjects,
        expected_pd=args.expected_pd, expected_hc=args.expected_hc,
        device=args.device, verbose=not args.quiet,
    )

    path_notes: List[str] = []
    uci_root, json_root, model_dir = cfg.resolved_paths(path_notes)
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    print(rule("="))
    print(" NEUROTRACE v4 — SUBJECT-GROUPED HYBRID CNN + BiLSTM + AUGMENTATION")
    print(" Huấn luyện v4 — chia theo NGƯỜI, mạng lai CNN + BiLSTM, tăng cường dữ liệu")
    print(rule("="))
    print(f" run at       : {datetime.now().isoformat(timespec='seconds')}")
    print(f" torch        : {'yes' if HAS_TORCH else 'NO (install torch to train)'}   "
          f"scipy: {'yes' if HAS_SCIPY else 'no (linear interp)'}   "
          f"device: {device}")
    print(f" UCI root     : {uci_root}")
    print(f" JSON root    : {json_root}")
    print(f" model dir    : {model_dir}")
    for note in path_notes:
        print(f" note         : {note}")
    print("\n RESEARCH PROTOTYPE — a statistical risk indicator, NOT a diagnosis.")

    if not HAS_TORCH and not cfg.audit_only:
        print("\n[FATAL] torch is not installed. Install it (`pip install torch`) or "
              "run with --audit-only to check the data pipeline only.")
        return 2

    # ---------------------------------------------------------------- data
    section("[1] LOADING & FEATURE BUILD / NẠP DỮ LIỆU & XÂY ĐẶC TRƯNG")
    t0 = time.time()
    try:
        bundle, stats = build_dataset(cfg)
    except RuntimeError as exc:
        print(f"\n[FATAL] {exc}")
        print("       Check --project-root / --uci-root / --json-root. "
              "Kiểm tra lại đường dẫn dữ liệu.")
        return 5
    table = subject_table(bundle["groups"], bundle["y"], bundle["sources"])
    audit = audit_subjects(table, cfg, stats)
    print_dataset_summary(bundle, table, stats, cfg)
    print(f"\n   feature build time: {time.time() - t0:.1f}s")

    if cfg.audit_only:
        print("\n[audit-only] stopping before any training / dừng trước khi huấn luyện.")
        return 0

    # -------------------------------------------------------------- splits
    section("[2] SUBJECT-GROUPED SPLITS / CHIA TẬP THEO NGƯỜI")
    holdout = make_holdout(table, cfg.holdout_frac, cfg.seed)
    trainval_subjects = [s for s in table if s not in set(holdout)]
    folds = make_folds(table, trainval_subjects, cfg.n_splits, cfg.seed)

    kv("holdout subjects (never trained on)", f"{len(holdout)} / {len(table)}")
    kv("holdout by (source,label)",
       dict(Counter((table[s]["source"], table[s]["label"]) for s in holdout)))
    kv("train/val subjects", len(trainval_subjects))
    kv("GroupKFold splits", len(folds))
    if not folds:
        print("\n[FATAL] Not enough subjects to build grouped folds. "
              "You need at least 2 subjects per class after the holdout.")
        return 3

    if not verify_no_leakage(folds, set(holdout), bundle["groups"]):
        print("\n[FATAL] subject leakage detected - refusing to train / "
              "phát hiện rò rỉ người, từ chối huấn luyện.")
        return 4

    # ------------------------------------------------------------ CV loop
    section("[3] CROSS-VALIDATION / ĐÁNH GIÁ CHÉO (subject-grouped)")
    groups, y, sources = bundle["groups"], bundle["y"], bundle["sources"]
    X_all = bundle["X"]

    fold_rows: List[Dict[str, Any]] = []
    best_epochs: List[int] = []

    for i, (tr_subj, va_subj) in enumerate(folds, start=1):
        tr = np.isin(groups, tr_subj)
        va = np.isin(groups, va_subj)
        res = train_model(cfg, X_all[tr], y[tr], X_all[va], y[va], groups[va],
                          device, tag=f"fold{i}", groups_tr=groups[tr])
        met = evaluate_model(res["model"], res["scaler"], X_all[va], y[va],
                             groups[va], device, cfg.threshold)
        # The subject ids are stored alongside the metrics so that the
        # "no subject appears on both sides" claim can be re-verified later
        # from the JSON report alone, instead of having to trust the log.
        fold_rows.append({"fold": i, "n_train_subjects": len(tr_subj),
                          "n_val_subjects": len(va_subj), "metrics": met,
                          "best_epoch": res["best_epoch"],
                          "n_train_rows_before_aug": res.get("n_train_rows_before_aug"),
                          "n_train_rows_after_aug": res.get("n_train_rows_after_aug"),
                          "train_subjects": sorted(tr_subj),
                          "val_subjects": sorted(va_subj)})
        best_epochs.append(res["best_epoch"])
        aug_note = ""
        if res.get("n_train_rows_after_aug"):
            aug_note = (f", train {res['n_train_rows_before_aug']}"
                        f"->{res['n_train_rows_after_aug']} rows (augmented)")
        print(f"   fold {i}: {fmt_metrics(met)}  "
              f"(val subjects {len(va_subj)}, best epoch {res['best_epoch']}"
              f"{aug_note})")

    print_fold_table(fold_rows)
    tot_before = sum(int(r.get("n_train_rows_before_aug") or 0) for r in fold_rows)
    tot_after = sum(int(r.get("n_train_rows_after_aug") or 0) for r in fold_rows)
    if tot_after:
        kv("train rows before -> after augmentation",
           "{} -> {} (x{:.1f})".format(human(tot_before), human(tot_after),
                                       tot_after / max(1, tot_before)))
    n_rows_all = int(bundle["X"].shape[0])
    expansion = 1 + (len(cfg.rotation_angles) if cfg.augment_train else 0)
    tot_all = n_rows_all * expansion
    kv("augmented dataset size vs target",
       "{} / {}  [{}]".format(
           human(tot_all), human(cfg.min_samples),
           "OK" if tot_all >= cfg.min_samples else "BELOW TARGET"))
    mean_cv_auc = np.nanmean([r["metrics"].get("roc_auc", np.nan) for r in fold_rows]) \
        if fold_rows else float("nan")

    # ---------------------------------------------------- final + holdout
    section("[4] FINAL MODEL + SAME-DEVICE HOLDOUT / MÔ HÌNH CUỐI + HOLDOUT CÙNG THIẾT BỊ")
    final_epochs = int(max(cfg.min_epochs, round(float(np.mean(best_epochs))))) \
        if best_epochs else cfg.epochs
    tv = np.isin(groups, trainval_subjects)
    kv("retraining on all train/val subjects", f"{len(trainval_subjects)} subjects, "
                                               f"{int(tv.sum())} windows")
    kv("epochs", f"{final_epochs} (mean best epoch across folds)")

    final = train_model(cfg, X_all[tv], y[tv], None, None, None, device,
                        tag="final", epochs_override=final_epochs,
                        groups_tr=groups[tv])

    holdout_metrics: Dict[str, Any] = {}
    holdout_by_source: Dict[str, Any] = {}
    if holdout:
        ho = np.isin(groups, holdout)
        holdout_metrics = evaluate_model(final["model"], final["scaler"], X_all[ho], y[ho],
                                         groups[ho], device, cfg.threshold)
        print(f"\n   SAME-DEVICE HOLDOUT ({len(holdout)} unseen subjects):")
        print(f"     {fmt_metrics(holdout_metrics)}")
        for src, name in ((0, "uci (Wacom)"), (1, "app (touch)")):
            subs = [s for s in holdout if table[s]["source"] == src]
            if not subs:
                continue
            m = np.isin(groups, subs)
            met = evaluate_model(final["model"], final["scaler"], X_all[m], y[m],
                                 groups[m], device, cfg.threshold)
            holdout_by_source[name] = met
            print(f"     {name:<14} n={len(subs):<4} {fmt_metrics(met)}")

    # ------------------------------------------------------- cross-device
    section("[5] CROSS-DEVICE HOLDOUT / HOLDOUT KHÁC THIẾT BỊ")
    cross: Dict[str, Any] = {"status": "skipped", "reason": ""}
    uci_train_subjects = [s for s in trainval_subjects if table[s]["source"] == 0]
    app_holdout = [s for s in holdout if table[s]["source"] == 1]
    uci_holdout = [s for s in holdout if table[s]["source"] == 0]

    if not uci_train_subjects or not app_holdout:
        cross["reason"] = "need both UCI training subjects and app holdout subjects"
        print(f"   skipped: {cross['reason']}")
    else:
        m_tr = np.isin(groups, uci_train_subjects)
        print(f"   training a UCI-ONLY model on {len(uci_train_subjects)} subjects "
              f"({int(m_tr.sum())} trials) — it has never seen touch-screen data.")
        uci_only = train_model(cfg, X_all[m_tr], y[m_tr], None, None, None, device,
                               tag="uci-only", epochs_override=final_epochs,
                               groups_tr=groups[m_tr])

        m_app = np.isin(groups, app_holdout)
        app_met = evaluate_model(uci_only["model"], uci_only["scaler"], X_all[m_app],
                                 y[m_app], groups[m_app], device, cfg.threshold)
        cross["app_holdout_cross_device"] = app_met
        print(f"\n   CROSS-DEVICE  -> app holdout ({len(app_holdout)} unseen healthy subjects):")
        print(f"     specificity {app_met.get('specificity', float('nan')):.3f}   "
              f"false-positive rate "
              f"{(1 - app_met['specificity']) if not math.isnan(app_met.get('specificity', float('nan'))) else float('nan'):.3f}")
        print("     sensitivity : UNDEFINED — the app holdout contains healthy people only,")
        print("                   so no true-positive rate can be estimated. Not invented here.")

        if uci_holdout:
            m_uci = np.isin(groups, uci_holdout)
            uci_met = evaluate_model(uci_only["model"], uci_only["scaler"], X_all[m_uci],
                                     y[m_uci], groups[m_uci], device, cfg.threshold)
            cross["uci_holdout_same_device"] = uci_met
            print(f"\n   SAME-DEVICE  -> UCI holdout ({len(uci_holdout)} unseen subjects):")
            print(f"     {fmt_metrics(uci_met)}")

    # ------------------------------------------------------ device probe
    section("[6] DEVICE-LEAKAGE PROBE / DÒ RÒ RỈ THIẾT BỊ")
    probe = device_leakage_probe(bundle, table, trainval_subjects, cfg.seed)
    if probe.get("status") != "ok":
        print(f"   skipped: {probe.get('reason', 'unknown')}")
    else:
        auc = probe["mean_auc"]
        print(f"   linear probe predicting SOURCE (UCI vs app) from features:")
        print(f"     grouped 5-fold AUC = {auc:.3f} ± {probe['std_auc']:.3f}")
        print("   interpretation / diễn giải:")
        if auc >= 0.90:
            print("     !! SEVERE. The two sources are almost perfectly separable from")
            print("        the feature vector alone. Because every PD subject is on UCI,")
            print("        the classifier can reach high accuracy by detecting the DEVICE.")
            print("        Treat the PD/healthy metric as device detection, not screening.")
        elif auc >= 0.75:
            print("     !  MODERATE. A device signature is present. PD/healthy metrics are")
            print("        optimistic until PD data exists on the app device.")
        else:
            print("     ~  LOW. The two sources look broadly comparable after the")
            print("        geometry/pressure harmonisation — a good sign, though it does")
            print("        not by itself prove the model learned clinical signal.")

    # ------------------------------------------------------------- saving
    section("[7] SAVING ARTEFACTS / LƯU KẾT QUẢ")
    model_dir.mkdir(parents=True, exist_ok=True)
    # Fixed names (v4 spec). Unlike v3 the report is NOT timestamped, so a
    # deployment step can always find "train_report_v4.json".
    model_path = model_dir / MODEL_FILENAME
    scaler_path = model_dir / SCALER_FILENAME
    report_path = model_dir / REPORT_FILENAME
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        torch.save({
            "model_state_dict": final["model"].state_dict(),
            "config": asdict(cfg),
            "feature_names": bundle["feature_names"],
            "n_channels": bundle["n_channels"],
            "seq_len": int(X_all.shape[2]),
            "threshold": cfg.threshold,
            "input_layout": "(N, C, L) with C = " + ", ".join(bundle["feature_names"]),
            "output": "logit — apply sigmoid for a probability",
            "scaler_mean": final["scaler"].mean_.tolist(),
            "scaler_scale": final["scaler"].scale_.tolist(),
            "cv_metrics": [{k: (v if k != "metrics" else
                                {kk: vv for kk, vv in v.items()
                                 if kk not in ("per_subject", "per_subject_truth")})
                            for k, v in row.items()} for row in fold_rows],
            "holdout_metrics": {k: v for k, v in holdout_metrics.items()
                                if k not in ("per_subject", "per_subject_truth")},
            "cross_device": cross,
            "device_leakage_probe": probe,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "code_version": "v4-augmented",
            "disclaimer": "RESEARCH PROTOTYPE — not a medical device, not a diagnosis.",
        }, model_path)
        print(f"   model  : {model_path}")

        with open(scaler_path, "wb") as fh:
            pickle.dump(final["scaler"], fh)
        print(f"   scaler : {scaler_path}")

        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump({
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "config": asdict(cfg),
                "loader_stats": {k: (int(v) if isinstance(v, (int, np.integer)) else v)
                                 for k, v in stats.items()},
                "subjects": {s: table[s] for s in sorted(table)},
                "audit": audit,
                "augmentation": {
                    "window": cfg.window, "stride": cfg.stride,
                    "rotation_angles": list(cfg.rotation_angles),
                    "augment_train": bool(cfg.augment_train),
                    "windows_total": int(stats.get("windows_kept", 0)),
                    "windows_per_subject_median": stats.get("windows_per_subject_median", 0),
                    "train_expansion": 1 + (len(cfg.rotation_angles) if cfg.augment_train else 0),
                },
                "holdout_subjects": holdout,
                "cv_folds": [{k: (v if k != "metrics" else
                                  {kk: vv for kk, vv in v.items()
                                   if kk not in ("per_subject", "per_subject_truth")})
                              for k, v in row.items()} for row in fold_rows],
                "mean_cv_auc": None if math.isnan(mean_cv_auc) else float(mean_cv_auc),
                "holdout_same_device": {k: v for k, v in holdout_metrics.items()
                                        if k not in ("per_subject", "per_subject_truth")},
                "holdout_by_source": {k: {kk: vv for kk, vv in v.items()
                                          if kk not in ("per_subject", "per_subject_truth")}
                                      for k, v in holdout_by_source.items()},
                "cross_device": cross,
                "device_leakage_probe": probe,
            }, fh, indent=2, ensure_ascii=False, default=str)
        print(f"   report : {report_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"   [ERROR] could not save artefacts: {exc}")
        traceback.print_exc()

    # ------------------------------------------------------------ summary
    section("SUMMARY / TỔNG KẾT")
    print(f"   subjects total            : {len(table)}  "
          f"(PD {sum(1 for v in table.values() if v['label'] == 1)}, "
          f"healthy {sum(1 for v in table.values() if v['label'] == 0)})")
    print(f"   GroupKFold                : {len(folds)} folds, grouped by subject_id")
    print(f"   holdout (never trained)   : {len(holdout)} subjects "
          f"({cfg.holdout_frac:.0%})")
    cv_auc = [r["metrics"].get("roc_auc", float("nan")) for r in fold_rows]
    cv_auc = [v for v in cv_auc if not math.isnan(v)]
    print(f"   CV subject-level AUC      : "
          f"{np.mean(cv_auc):.3f} ± {np.std(cv_auc):.3f}" if cv_auc
          else "   CV subject-level AUC      : n/a (a fold had a single class)")
    if holdout_metrics:
        print(f"   holdout AUC (same-device) : "
              f"{holdout_metrics.get('roc_auc', float('nan')):.3f}")
    if probe.get("status") == "ok":
        print(f"   device-leakage probe AUC  : {probe['mean_auc']:.3f}")
    print("\n   REMINDERS / NHẮC NHỞ:")
    print("     * Research prototype — not a diagnosis, not a medical device.")
    print("     * No Parkinson data exists on the app device yet; the PD/healthy")
    print("       boundary is currently also a Wacom/touchscreen boundary (section [6]).")
    print("     * Metrics are subject-level; trial-level numbers would be optimistic.")
    print("     * Re-run this script after every data drop and compare the probe AUC.")

    print("\n" + rule("="))
    print(" TRAINING COMPLETE / HUẤN LUYỆN HOÀN TẤT")
    print(rule("=") + "\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:  # pragma: no cover
        print("\n[INTERRUPTED] training cancelled / đã huỷ.")
        sys.exit(130)
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover
        # Flush first: stdout is block-buffered when piped, so without this the
        # traceback (stderr, unbuffered) prints ABOVE the message and the one
        # line the user actually needs scrolls away.
        sys.stdout.flush()
        print(f"\n[FATAL] unexpected error / lỗi không mong đợi: "
              f"{type(exc).__name__}: {exc}")
        sys.stdout.flush()
        traceback.print_exc()
        sys.exit(1)
