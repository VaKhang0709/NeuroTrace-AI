%%writefile /content/drive/MyDrive/NEUROTRACE/train_pipeline_v5_diagnostic.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_pipeline_v5_diagnostic.py — NeuroTrace v5 Diagnostic & Domain-Invariant Trainer
======================================================================================
Production diagnostic training pipeline resolving hardware/device confounding
between Wacom Cintiq tablet recordings (UCI #395) and mobile touchscreen captures:
  1. Subject Window Capping (`--max-windows-per-subject`, default 150, `seed=42`)
     applied BEFORE augmentation to eliminate single-subject skew (1,700 -> <=150).
  2. Feature Channel Ablation (`--drop-channels`, e.g., `"2,3"` or `"2"`) slicing
     ablation indices out of canonical channels `["x", "y", "z", "pressure", "velocity", "dt"]`.
  3. Physiological 100 Hz Resampling + 4th-Order Zero-Phase 12 Hz Butterworth Filter
     preserving 4-8 Hz Parkinsonian tremor & dysmetria while stripping sub-pixel OS
     polling jitter and resolving canvas duration Simpson's Paradox.
  4. Dual-Head Adversarial Architecture (`NeuroTraceDiagnosticNet`):
       * Shared Multi-Scale Conv1D + 2-Layer BiLSTM Backbone
       * `Clinical Classifier` outputting RAW LOGITS (paired with `BCEWithLogitsLoss`)
       * `GradientReversalLayer` (`-lambda * grad`, default `lambda=0.3`) feeding
         `Device Classifier` + second-order CORAL covariance alignment across domains.
  5. Strict `parent_subject_id` `GroupKFold` & Stratified Unseen Holdout Guards:
     Every virtual session `{base_subject_id}_vS*` maps back to `base_subject_id`,
     enforcing ZERO participant identity leakage across all splits.
  6. Automatic CPU/CUDA Adaptation: cleanly adapts default batch size (16 on CPU,
     32 on CUDA) and default epochs (12 on CPU, 20 on CUDA) without failure.
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
import tarfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy import signal
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
CANONICAL_FEATURES: Tuple[str, ...] = ("x", "y", "z", "pressure", "velocity", "dt")
DEFAULT_ROTATIONS: Tuple[float, ...] = (-30.0, -15.0, 15.0, 30.0)
MEDICAL_DISCLAIMER = (
    "RESEARCH PROTOTYPE — Not an approved medical device. Statistical risk score only."
)
SOS_LP_12HZ = signal.butter(4, 12.0, btype="low", fs=100.0, output="sos")
COL_ALIASES: Dict[str, Tuple[str, ...]] = {
    "x": ("x", "xpos", "xcoord", "xcoordinate", "pos_x"),
    "y": ("y", "ypos", "ycoord", "ycoordinate", "pos_y"),
    "z": ("z", "zpos", "zcoord", "penheight", "height"),
    "pressure": ("pressure", "press", "p", "force", "penpressure"),
    "grip": ("gripangle", "grip", "tilt", "altitude", "azimuth"),
    "t": ("timestamp", "time", "tms", "t", "sampletime"),
    "test_id": ("testid", "test", "taskid", "task", "spiraltype"),
}


def parse_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {val!r}")


def parse_int_list(raw: str) -> Tuple[int, ...]:
    if not raw or not raw.strip():
        return ()
    out: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            idx = int(part)
            if idx < 0 or idx >= len(CANONICAL_FEATURES):
                raise ValueError(f"Channel index {idx} out of range 0..{len(CANONICAL_FEATURES)-1}")
            if idx not in out:
                out.append(idx)
    return tuple(sorted(out))


@dataclass
class RawTrial:
    subject_id: str
    parent_subject_id: str
    label: int
    source: str          # "uci" | "app"
    device: str          # "wacom" | "touch"
    is_virtual: bool
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    pressure: np.ndarray
    t: np.ndarray
    has_real_pressure: bool
    test_id: int = 0
    task: str = "spiral"
    synthetic_time: bool = False


@dataclass
class V5Config:
    project_root: str = "/content/drive/MyDrive/NEUROTRACE"
    source_filter: str = "all"                 # all | uci | app | all+virtual
    max_windows_per_subject: int = 150
    drop_channels: Tuple[int, ...] = ()
    device_adversarial: bool = False
    grl_lambda: float = 0.3
    epochs: int = 20
    batch_size: int = 16
    repeats: int = 3
    n_splits: int = 5
    holdout_fraction: float = 0.10
    seed: int = 42
    lr: float = 1.5e-3
    weight_decay: float = 1e-4
    seq_len: int = 256
    window: int = 256
    stride: int = 64
    threshold: float = 0.50
    conv1: int = 32
    conv2: int = 64
    lstm_hidden: int = 32
    lstm_layers: int = 2
    emb_dim: int = 4
    dropout: float = 0.25
    min_epochs: int = 10
    early_stop_patience: int = 6
    tag: str = "v5_run"
    device_pref: str = "auto"
    resolved_device: str = "cpu"
    save_model: bool = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device_and_defaults(
    pref: str, explicit_epochs: Optional[int], explicit_batch_size: Optional[int]
) -> Tuple[torch.device, int, int, str]:
    cuda_ok = torch.cuda.is_available() and pref != "cpu"
    if pref == "cuda" and not torch.cuda.is_available():
        note = "CUDA requested but unavailable -> auto-fallback to CPU (default batch_size=16, epochs=12)."
        dev = torch.device("cpu")
    elif cuda_ok:
        dev = torch.device("cuda")
        note = f"CUDA active ({torch.cuda.get_device_name(0)})."
    else:
        dev = torch.device("cpu")
        note = "CUDA unavailable -> CPU auto-adaptation active (default batch_size=16, default epochs=12)."
    if dev.type == "cpu":
        torch.set_num_threads(min(2, max(1, os.cpu_count() or 2)))
        epochs = explicit_epochs if explicit_epochs is not None else 12
        batch_size = explicit_batch_size if explicit_batch_size is not None else 16
    else:
        epochs = explicit_epochs if explicit_epochs is not None else 20
        batch_size = explicit_batch_size if explicit_batch_size is not None else 32
    return dev, int(epochs), int(batch_size), note


# ===========================================================================
# 1. Raw UCI #395, Mobile JSON, and Virtual Session Loaders


# ===========================================================================
def normalise_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.strip().lower())


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        val = float(str(v).strip())
        return val if math.isfinite(val) else None
    except Exception:
        return None


def normalise_subject_code(stem: str) -> str:
    s = stem.strip()
    s = re.sub(r"(?i)[_-]?(sst|dst|stcp|spiral|meander|circle|trial\d+|t\d+)$", "", s)
    return s or stem.strip()


def infer_group_from_path(rel: Path) -> Optional[int]:
    parts_l = [p.lower() for p in rel.parts[:-1]]
    stem = rel.stem
    if any("parkinson" in p or p in ("pd", "patients", "patient") for p in parts_l):
        return 1
    if any("control" in p or "healthy" in p or p in ("hc", "controls") for p in parts_l):
        return 0
    if re.match(r"^(H_)?P[_-]?\d+", stem, re.IGNORECASE):
        return 1
    if re.match(r"^(H_)?C[_-]?\d+", stem, re.IGNORECASE):
        return 0
    return None


def normalise_test_id(raw: Any) -> Optional[int]:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in ("0", "0.0", "sst", "static", "staticspiral"):
        return 0
    if s in ("1", "1.0", "dst", "dynamic", "dynamicspiral"):
        return 1
    if s in ("2", "2.0", "stcp", "circular", "stability"):
        return 2
    v = _to_float(s)
    if v is not None and int(round(v)) in (0, 1, 2):
        return int(round(v))
    return None


def read_uci_table(path: Path) -> Tuple[List[str], Dict[str, List[str]]]:
    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    lines: List[str] = []
    for ln in raw_lines:
        s = ln.strip()
        if not s or s.startswith("#") or s.startswith("//") or s.lower().startswith("dataset is delimited"):
            continue
        lines.append(s)
    if not lines:
        return [], {}
    sample = "\n".join(lines[:15])
    delim = ";"
    best_cnt = -1
    for cand in (";", ",", "\t", " "):
        cnt = sample.count(cand)
        if cnt > best_cnt:
            best_cnt, delim = cnt, cand
    rows = [[c.strip() for c in row] for row in csv.reader(lines, delimiter=delim) if any(c.strip() for c in row)]
    while rows and len(rows[0]) == 1 and _to_float(rows[0][0]) is None:
        rows.pop(0)
    if not rows:
        return [], {}
    first_numeric = sum(1 for c in rows[0] if _to_float(c) is not None)
    has_header = first_numeric < max(1, len(rows[0]) // 2)
    n_cols = max(len(r) for r in rows)
    if has_header:
        cols = [rows[0][i] if i < len(rows[0]) and rows[0][i] else f"col_{i}" for i in range(n_cols)]
        body = rows[1:]
    else:
        cols = [f"__pos_{i}" for i in range(n_cols)]
        body = rows
    data: Dict[str, List[str]] = {c: [] for c in cols}
    for r in body:
        for i, c in enumerate(cols):
            data[c].append(r[i] if i < len(r) else "")
    return cols, data


def resolve_uci_columns(cols: Sequence[str], data: Dict[str, List[str]]) -> Tuple[Dict[str, Optional[str]], bool]:
    norm_map = {normalise_key(c): c for c in cols}
    resolved: Dict[str, Optional[str]] = {}
    for key, aliases in COL_ALIASES.items():
        resolved[key] = next((norm_map[a] for a in aliases if a in norm_map), None)
    if resolved["x"] is not None and resolved["y"] is not None:
        return resolved, False
    numeric_cols: List[str] = []
    for c in cols:
        vals = data.get(c, [])[:40]
        if vals and sum(1 for v in vals if _to_float(v) is not None) >= max(1, int(0.8 * len(vals))):
            numeric_cols.append(c)
    if len(numeric_cols) >= 7:
        return {
            "x": numeric_cols[0], "y": numeric_cols[1], "z": numeric_cols[2],
            "pressure": numeric_cols[3], "grip": numeric_cols[4],
            "t": numeric_cols[5], "test_id": numeric_cols[6],
        }, True
    if len(numeric_cols) == 6:
        return {
            "x": numeric_cols[0], "y": numeric_cols[1], "z": numeric_cols[2],
            "pressure": numeric_cols[3], "grip": None,
            "t": numeric_cols[5], "test_id": None,
        }, True
    if len(numeric_cols) >= 4:
        return {
            "x": numeric_cols[0], "y": numeric_cols[1], "z": None,
            "pressure": numeric_cols[2], "grip": None,
            "t": numeric_cols[3], "test_id": None,
        }, True
    return resolved, False


def repair_time_axis(t: np.ndarray) -> Optional[np.ndarray]:
    if t.size < 4 or not np.isfinite(t).any():
        return None
    arr = t.astype(np.float64).copy()
    diffs = np.diff(arr)
    pos = diffs[diffs > 0]
    if pos.size == 0:
        return None
    med = float(np.median(pos))
    # Convert ms or microseconds to seconds
    if med > 100.0:
        scale = 1e-6 if med > 5e4 else 1e-3
    elif med > 0.5:
        scale = 1e-3
    else:
        scale = 1.0
    arr = (arr - arr[0]) * scale
    d = np.diff(arr)
    pos_s = d[d > 0]
    if pos_s.size == 0:
        return None
    step = float(np.median(pos_s))
    d_clean = np.where((d > 0) & (d <= 20.0 * step), d, step)
    return np.concatenate([[0.0], np.cumsum(d_clean)])


def load_uci_trials(uci_root: Path, stats: Counter) -> List[RawTrial]:
    trials: List[RawTrial] = []
    if not uci_root.exists():
        return trials
    for path in sorted(p for p in uci_root.rglob("*") if p.is_file()):
        if path.suffix.lower() not in (".txt", ".csv", ".tsv", ".dat"):
            continue
        stats["uci_files"] += 1
        rel = path.relative_to(uci_root)
        label = infer_group_from_path(rel)
        if label is None:
            stats["uci_unlabelled_files"] += 1
            continue
        try:
            cols, data = read_uci_table(path)
            colmap, used_fallback = resolve_uci_columns(cols, data)
            if used_fallback:
                stats["uci_positional_fallback"] += 1
            if colmap.get("x") is None or colmap.get("y") is None:
                stats["uci_missing_xy"] += 1
                continue
            n_rows = len(data[colmap["x"]])
            xs, ys, zs, ps, ts, tids = [], [], [], [], [], []
            inferred_tid = normalise_test_id(path.stem)
            if inferred_tid is None:
                inferred_tid = 0
            for i in range(n_rows):
                xv = _to_float(data[colmap["x"]][i])
                yv = _to_float(data[colmap["y"]][i])
                if xv is None or yv is None:
                    continue
                zv = _to_float(data[colmap["z"]][i]) if colmap.get("z") else None
                pv = _to_float(data[colmap["pressure"]][i]) if colmap.get("pressure") else None
                tv = _to_float(data[colmap["t"]][i]) if colmap.get("t") else float(len(xs)) / 133.0
                tid = normalise_test_id(data[colmap["test_id"]][i]) if colmap.get("test_id") else inferred_tid
                xs.append(xv)
                ys.append(yv)
                zs.append(zv if zv is not None else float("nan"))
                ps.append(pv if pv is not None else float("nan"))
                ts.append(tv if tv is not None else float(len(xs)) / 133.0)
                tids.append(inferred_tid if tid is None else tid)
            if len(xs) < 20:
                continue
            sid = f"uci::{normalise_subject_code(path.stem)}"
            tids_arr = np.asarray(tids, dtype=int)
            # Split contiguous test runs (SST=0, DST=1, STCP=2)
            start = 0
            for idx in range(1, len(tids_arr) + 1):
                if idx == len(tids_arr) or tids_arr[idx] != tids_arr[start]:
                    if idx - start >= 24:
                        sl = slice(start, idx)
                        t_rep = repair_time_axis(np.asarray(ts[sl], dtype=np.float64))
                        if t_rep is not None and t_rep[-1] >= 0.4:
                            p_seg = np.asarray(ps[sl], dtype=np.float32)
                            trials.append(RawTrial(
                                subject_id=sid,
                                parent_subject_id=sid,
                                label=int(label),
                                source="uci",
                                device="wacom",
                                is_virtual=False,
                                x=np.asarray(xs[sl], dtype=np.float32),
                                y=np.asarray(ys[sl], dtype=np.float32),
                                z=np.asarray(zs[sl], dtype=np.float32),
                                pressure=p_seg,
                                t=t_rep,
                                has_real_pressure=bool(np.isfinite(p_seg).sum() >= 10),
                                test_id=int(tids_arr[start]),
                                task=f"spiral_{int(tids_arr[start])}",
                            ))
                            stats["uci_trials"] += 1
                    start = idx
        except Exception:
            stats["uci_read_errors"] += 1
    return trials


def extract_json_stroke_arrays(rec: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    container = None
    for k in ("strokes", "points", "data", "trajectory", "path", "samples", "drawing"):
        if k in rec and isinstance(rec[k], list) and rec[k]:
            container = rec[k]
            break
    if container is None and isinstance(rec.get("payload"), dict):
        return extract_json_stroke_arrays(rec["payload"])
    if container is None:
        return None
    flat_pts: List[Any] = []
    for item in container:
        if isinstance(item, dict) and isinstance(item.get("points"), list):
            flat_pts.extend(item["points"])
        elif isinstance(item, list) and item and isinstance(item[0], (dict, list)):
            flat_pts.extend(item)
        else:
            flat_pts.append(item)
    xs, ys, ps, ts = [], [], [], []
    for pt in flat_pts:
        if isinstance(pt, dict):
            xv = _to_float(pt.get("x", pt.get("clientX")))
            yv = _to_float(pt.get("y", pt.get("clientY")))
            pv = _to_float(pt.get("pressure", pt.get("p", pt.get("force"))))
            tv = _to_float(pt.get("t", pt.get("timestamp", pt.get("time"))))
        elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
            xv, yv = _to_float(pt[0]), _to_float(pt[1])
            tv = _to_float(pt[2]) if len(pt) >= 3 else None
            pv = _to_float(pt[3]) if len(pt) >= 4 else None
        else:
            continue
        if xv is None or yv is None:
            continue
        xs.append(xv)
        ys.append(yv)
        ps.append(pv if pv is not None else float("nan"))
        ts.append(tv if tv is not None else float("nan"))
    if len(xs) < 20:
        return None
    t_arr = np.asarray(ts, dtype=np.float64)
    synth = False
    t_rep = repair_time_axis(t_arr)
    if t_rep is None:
        t_rep = np.arange(len(xs), dtype=np.float64) / 60.0
        synth = True
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(ps, dtype=np.float32),
        t_rep if not synth else -t_rep - 1.0,
    )


def load_app_trials(json_root: Path, stats: Counter) -> List[RawTrial]:
    trials: List[RawTrial] = []
    if not json_root.exists():
        return trials
    seen_participants: set[str] = set()
    for path in sorted(json_root.rglob("*.json")):
        stats["json_files"] += 1
        try:
            raw_obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            records = raw_obj if isinstance(raw_obj, list) else [raw_obj]
            for rec in records:
                if not isinstance(rec, dict):
                    continue
                extracted = extract_json_stroke_arrays(rec)
                if extracted is None:
                    continue
                x, y, p, t_enc = extracted
                synth = bool(t_enc[0] < 0)
                t = (-t_enc - 1.0) if synth else t_enc
                if t[-1] < 0.4:
                    continue
                pid_raw = (
                    rec.get("participant_id")
                    or rec.get("subject_id")
                    or rec.get("userId")
                    or rec.get("user_id")
                    or rec.get("session_id")
                    or path.stem
                )
                m_id = re.match(r"^(?:healthy_)?(healthy_\d+)", str(pid_raw).strip(), flags=re.I)
                pid_clean = m_id.group(1).lower() if m_id else re.sub(r"[_-]?(trial\d+|t\d+|spiral.*)$", "", str(pid_raw).strip(), flags=re.I)
                sid = f"json::{pid_clean or path.stem}"
                seen_participants.add(sid)
                lbl_raw = str(rec.get("label", rec.get("group", "healthy"))).lower()
                lbl = 1 if lbl_raw in ("1", "pd", "parkinson", "patient") else 0
                finite_p = p[np.isfinite(p)]
                real_p = bool(finite_p.size >= 10 and np.unique(np.round(finite_p, 4)).size >= 4)
                trials.append(RawTrial(
                    subject_id=sid,
                    parent_subject_id=sid,
                    label=lbl,
                    source="app",
                    device="touch",
                    is_virtual=False,
                    x=x,
                    y=y,
                    z=np.full_like(x, np.nan),
                    pressure=p,
                    t=t,
                    has_real_pressure=real_p,
                    test_id=-1,
                    task="spiral",
                    synthetic_time=synth,
                ))
                stats["json_trials"] += 1
        except Exception:
            stats["json_read_errors"] += 1
    stats["json_participants"] = len(seen_participants)
    return trials


def load_virtual_trials(virtual_dir: Path, stats: Counter) -> List[RawTrial]:
    trials: List[RawTrial] = []
    if not virtual_dir.exists():
        return trials
    npz_files = sorted(virtual_dir.glob("*.npz"))
    tar_bundle = virtual_dir / "virtual_sessions_552_npz.tar.gz"
    if not npz_files and tar_bundle.is_file():
        with tarfile.open(tar_bundle, "r:gz") as tf:
            tf.extractall(path=virtual_dir)
        npz_files = sorted(virtual_dir.glob("*.npz"))
    parent_map_file = virtual_dir / "session_parent_map.json"
    parent_map: Dict[str, str] = (
        json.loads(parent_map_file.read_text(encoding="utf-8"))
        if parent_map_file.is_file()
        else {}
    )
    for path in npz_files:
        try:
            with np.load(path, allow_pickle=False) as d:
                v_sid = str(d["virtual_subject_id"].item())
                p_sid = parent_map.get(v_sid, str(d["parent_subject_id"].item()))
                lbl = int(d["label"].item())
                src = str(d["source"].item())
                dev = str(d["device"].item())
                n_t = int(d["n_trials"].item())
                stats["virtual_sessions"] += 1
                for i in range(n_t):
                    trials.append(RawTrial(
                        subject_id=v_sid,
                        parent_subject_id=p_sid,
                        label=lbl,
                        source=src,
                        device=dev,
                        is_virtual=True,
                        x=np.asarray(d[f"t{i}_x"], dtype=np.float32),
                        y=np.asarray(d[f"t{i}_y"], dtype=np.float32),
                        z=np.asarray(d[f"t{i}_z"], dtype=np.float32),
                        pressure=np.asarray(d[f"t{i}_pressure"], dtype=np.float32),
                        t=np.asarray(d[f"t{i}_t"], dtype=np.float64),
                        has_real_pressure=bool(d[f"t{i}_has_real_pressure"].item()),
                        test_id=int(d[f"t{i}_test_id"].item()),
                        task=str(d[f"t{i}_task"].item()),
                        synthetic_time=bool(d[f"t{i}_synthetic_time"].item()),
                    ))
                    stats["virtual_trials"] += 1
        except Exception:
            stats["virtual_read_errors"] += 1
    return trials


# ===========================================================================
# 2. Physics-Preserving 6-Channel Kinematics & Windowing


# ===========================================================================
def virtual_pressure_proxy(speed: np.ndarray, radius_norm: np.ndarray) -> np.ndarray:
    s_norm = (speed - np.min(speed)) / (np.ptp(speed) + 1e-6)
    proxy = 0.65 * (1.0 - 0.35 * s_norm) + 0.15 * np.clip(radius_norm, 0.0, 1.5)
    return proxy.astype(np.float64)


def slice_windows(mat: np.ndarray, window: int, stride: int, min_len: int = 24) -> List[np.ndarray]:
    n = int(mat.shape[0])
    if n < min_len:
        return []
    if n <= window:
        return [mat]
    starts = list(range(0, n - window + 1, stride))
    if starts[-1] != n - window:
        starts.append(n - window)
    return [mat[s:s + window] for s in starts]


def resample_window(mat: np.ndarray, seq_len: int) -> np.ndarray:
    n, c = mat.shape
    if n == seq_len:
        return mat.astype(np.float64)
    src = np.linspace(0.0, 1.0, n)
    dst = np.linspace(0.0, 1.0, seq_len)
    return np.stack([np.interp(dst, src, mat[:, j]) for j in range(c)], axis=1)


def extract_trial_windows(trial: RawTrial, cfg: V5Config) -> List[np.ndarray]:
    x = trial.x.astype(np.float64)
    y = trial.y.astype(np.float64)
    t = trial.t.astype(np.float64)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(t)
    if m.sum() < 20:
        return []
    x, y, t = x[m], y[m], t[m]
    z = trial.z[m].astype(np.float64) if trial.z.shape == trial.x.shape else np.full(m.sum(), np.nan)
    p = trial.pressure[m].astype(np.float64) if trial.pressure.shape == trial.x.shape else np.full(m.sum(), np.nan)
    dt_raw = np.diff(t, prepend=t[0])
    dt_raw[0] = dt_raw[1] if dt_raw.size > 1 else 0.01
    pos = dt_raw[dt_raw > 0]
    if pos.size < 16:
        return []
    med_dt = float(np.median(pos))
    dt_rep = np.where(dt_raw > 0, dt_raw, med_dt)
    t_clean = np.concatenate([[0.0], np.cumsum(np.clip(dt_rep[1:], 0.05 * med_dt, 10.0 * med_dt))])
    dur = float(t_clean[-1])
    if dur < 0.4:
        return []
    n_grid = max(48, int(round(dur * 100.0)) + 1)
    tg = np.linspace(0.0, dur, n_grid)
    xg = signal.sosfiltfilt(SOS_LP_12HZ, np.interp(tg, t_clean, x))
    yg = signal.sosfiltfilt(SOS_LP_12HZ, np.interp(tg, t_clean, y))
    dt_filt = signal.sosfiltfilt(SOS_LP_12HZ, np.interp(tg, t_clean, dt_rep))
    cx0, cy0 = float(np.median(xg)), float(np.median(yg))
    r0 = np.hypot(xg - cx0, yg - cy0)
    sc0 = float(np.percentile(r0, 95))
    if not np.isfinite(sc0) or sc0 <= 1e-9:
        sc0 = 1.0
    has_z = np.isfinite(z).any()
    if has_z:
        zf = np.nan_to_num(z, nan=float(np.nanmedian(z)))
        zg = (np.interp(tg, t_clean, zf) - float(np.nanmedian(zf))) / sc0
    else:
        zg = np.zeros_like(tg)
    vx0 = np.gradient((xg - cx0) / sc0, 0.01)
    vy0 = np.gradient((yg - cy0) / sc0, 0.01)
    sp0 = np.hypot(vx0, vy0)
    has_p = bool(trial.has_real_pressure) and np.isfinite(p).any()
    if has_p:
        finite_p = p[np.isfinite(p)]
        if np.unique(np.round(finite_p, 4)).size < 4:
            has_p = False
    if has_p:
        pf = signal.sosfiltfilt(SOS_LP_12HZ, np.interp(tg, t_clean, np.nan_to_num(p, nan=float(np.nanmedian(p)))))
    else:
        pf = virtual_pressure_proxy(sp0, r0 / sc0)
    stacked = np.stack([xg, yg, zg, pf, sp0, dt_filt], axis=1)
    windows_out: List[np.ndarray] = []
    for w in slice_windows(stacked, cfg.window, cfg.stride, min_len=24):
        w_res = resample_window(w, cfg.seq_len)
        wx = w_res[:, 0] - np.mean(w_res[:, 0])
        wy = w_res[:, 1] - np.mean(w_res[:, 1])
        wsc = float(np.sqrt(np.mean(wx ** 2 + wy ** 2)))
        if wsc <= 1e-8:
            continue
        wv = w_res[:, 4]
        wp = (w_res[:, 3] - float(np.mean(w_res[:, 3]))) / (float(np.std(w_res[:, 3])) + 1e-5)
        wdt = (w_res[:, 5] - float(np.median(w_res[:, 5]))) / (float(np.median(w_res[:, 5])) + 1e-6)
        mat6 = np.stack([wx / wsc, wy / wsc, w_res[:, 2], wp, wv, wdt], axis=0).astype(np.float32)
        if np.isfinite(mat6).all():
            windows_out.append(mat6)
    return windows_out


def rotate_xy(X: np.ndarray, angle_deg: float) -> np.ndarray:
    theta = math.radians(float(angle_deg))
    c, s = math.cos(theta), math.sin(theta)
    out = X.copy()
    x_copy = out[:, 0, :].copy()
    y_copy = out[:, 1, :].copy()
    out[:, 0, :] = x_copy * c - y_copy * s
    out[:, 1, :] = x_copy * s + y_copy * c
    return out


def build_v5_dataset(cfg: V5Config) -> Tuple[Dict[str, Any], Counter]:
    root = Path(cfg.project_root)
    uci_dir = root / "data" / "uci_raw"
    if not uci_dir.exists():
        uci_dir = root / "external_test_data" / "uci_395_raw"
    json_dir = root / "data" / "healthy"
    if not json_dir.exists():
        json_dir = root / "healthy"
    virt_dir = root / "data" / "virtual_sessions"
    stats: Counter = Counter()
    trials: List[RawTrial] = []
    if cfg.source_filter in ("all", "uci", "all+virtual"):
        trials.extend(load_uci_trials(uci_dir, stats))
    if cfg.source_filter in ("all", "app", "all+virtual"):
        trials.extend(load_app_trials(json_dir, stats))
    if cfg.source_filter == "all+virtual":
        trials.extend(load_virtual_trials(virt_dir, stats))
    if not trials:
        raise RuntimeError(f"Zero trials loaded under --source-filter={cfg.source_filter!r}")
    # Group raw windows strictly by parent human participant ID before capping
    by_parent_real: Dict[str, List[Tuple[np.ndarray, int, int, str, bool]]] = defaultdict(list)
    by_parent_virt: Dict[str, List[Tuple[np.ndarray, int, int, str, bool]]] = defaultdict(list)
    all_session_ids: set[str] = set()
    for tr in trials:
        all_session_ids.add(tr.subject_id)
        wins = extract_trial_windows(tr, cfg)
        src_int = 0 if tr.source == "uci" else 1
        target_bucket = by_parent_virt if tr.is_virtual else by_parent_real
        for w in wins:
            target_bucket[tr.parent_subject_id].append((w, tr.label, src_int, tr.subject_id, tr.is_virtual))
    raw_counts = {pid: len(by_parent_real[pid]) + len(by_parent_virt[pid]) for pid in by_parent_real}
    stats["raw_windows_uncapped"] = int(sum(raw_counts.values()))
    stats["raw_max_windows_single_subject"] = int(max(raw_counts.values())) if raw_counts else 0
    rng = np.random.default_rng(42)  # Mandatory deterministic pre-augmentation window cap
    keep_channels = [i for i in range(len(CANONICAL_FEATURES)) if i not in cfg.drop_channels]
    feature_names = [CANONICAL_FEATURES[i] for i in keep_channels]
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    parent_list: List[str] = []
    session_list: List[str] = []
    source_list: List[int] = []
    virtual_list: List[bool] = []
    cap = int(cfg.max_windows_per_subject)
    for pid in sorted(by_parent_real):
        real_items = by_parent_real[pid]
        virt_items = by_parent_virt.get(pid, [])
        if cap > 0 and len(real_items) > cap:
            sel = sorted(rng.choice(len(real_items), size=cap, replace=False))
            real_kept = [real_items[i] for i in sel]
        else:
            real_kept = list(real_items)
        virt_kept: List[Tuple[np.ndarray, int, int, str, bool]] = []
        if virt_items:
            virt_cap = cap * 2 if cap > 0 else len(virt_items)
            if len(virt_items) > virt_cap:
                sel_v = sorted(rng.choice(len(virt_items), size=virt_cap, replace=False))
                virt_kept = [virt_items[i] for i in sel_v]
            else:
                virt_kept = list(virt_items)
        for w, lbl, src_i, sess_id, is_v in real_kept + virt_kept:
            X_list.append(w[keep_channels, :])
            y_list.append(lbl)
            parent_list.append(pid)
            session_list.append(sess_id)
            source_list.append(src_i)
            virtual_list.append(is_v)
    X = np.stack(X_list).astype(np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    parents = np.asarray(parent_list)
    sessions = np.asarray(session_list)
    sources = np.asarray(source_list, dtype=np.int64)
    is_virt = np.asarray(virtual_list, dtype=bool)
    stats["n_base_subjects"] = int(len(set(parents)))
    stats["n_effective_profiles"] = int(len(all_session_ids))
    stats["windows_kept"] = int(len(X))
    bundle = {
        "X": X,
        "y": y,
        "parents": parents,
        "sessions": sessions,
        "sources": sources,
        "is_virtual": is_virt,
        "keep_channels": keep_channels,
        "feature_names": feature_names,
        "in_channels": len(keep_channels),
    }
    return bundle, stats


# ===========================================================================
# 3. Dual-Head Adversarial Network with Gradient Reversal Layer (GRL)


# ===========================================================================
class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lam: float) -> torch.Tensor:
        ctx.lam = float(lam)
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.lam * grad_output, None


class GradientReversalLayer(nn.Module):

    def __init__(self, lam: float = 0.3) -> None:
        super().__init__()
        self.lam = float(lam)

    def forward(self, x: torch.Tensor, lam: Optional[float] = None) -> torch.Tensor:
        active_lam = self.lam if lam is None else float(lam)
        return GradientReversalFn.apply(x, active_lam)


class NeuroTraceDiagnosticNet(nn.Module):
    """Dual-head 1D-CNN + BiLSTM diagnostic network with Gradient Reversal Layer.
    `forward(x)` returns RAW LOGITS `(B,)` strictly for `nn.BCEWithLogitsLoss`
    and clean standard-operator ONNX export (`opset_version=17`, `dynamo=False`).
    """

    def __init__(
        self,
        in_channels: int = 5,
        conv1: int = 32,
        conv2: int = 64,
        lstm_hidden: int = 32,
        lstm_layers: int = 2,
        emb_dim: int = 4,
        dropout: float = 0.25,
        grl_lambda: float = 0.3,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.conv1 = int(conv1)
        self.conv2 = int(conv2)
        self.lstm_hidden = int(lstm_hidden)
        self.lstm_layers = int(lstm_layers)
        self.emb_dim = int(emb_dim)
        self.dropout = float(dropout)
        self.grl_lambda = float(grl_lambda)
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, conv1, kernel_size=7, padding=3),
            nn.BatchNorm1d(conv1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(conv1, conv2, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=conv2,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.feature_proj = nn.Sequential(
            nn.Linear(lstm_hidden * 4, 48),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(48, emb_dim),
            nn.LayerNorm(emb_dim),
        )
        self.clinical_classifier = nn.Linear(emb_dim, 1)
        self.grl = GradientReversalLayer(lam=grl_lambda)
        self.device_classifier = nn.Sequential(
            nn.Linear(emb_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.block2(self.block1(x)).transpose(1, 2)
        out, _ = self.lstm(h)
        pooled = torch.cat([out.mean(dim=1), out.amax(dim=1)], dim=-1)
        return self.feature_proj(pooled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return RAW clinical logits `(B,)` — NO sigmoid inside `forward()`."""
        feat = self.extract_features(x)
        return self.clinical_classifier(feat).squeeze(-1)

    def forward_dual(self, x: torch.Tensor, lam: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.extract_features(x)
        clin_logits = self.clinical_classifier(feat).squeeze(-1)
        dev_logits = self.device_classifier(self.grl(feat, lam=lam)).squeeze(-1)
        return clin_logits, dev_logits, feat
    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        self.eval()
        logits = self.forward(x)
        return torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64)


# ===========================================================================
# 4. Subject-Grouped Splits, Training Loop, and Leakage Probes


# ===========================================================================
def build_subject_table(parents: np.ndarray, y: np.ndarray, sources: np.ndarray) -> Dict[str, Dict[str, int]]:
    table: Dict[str, Dict[str, int]] = {}
    for pid in sorted(set(parents)):
        m = parents == pid
        table[pid] = {
            "label": int(round(float(np.mean(y[m])))),
            "source": int(round(float(np.mean(sources[m])))),
            "n_windows": int(m.sum()),
        }
    return table


def make_holdout_subjects(table: Dict[str, Dict[str, int]], frac: float, seed: int) -> List[str]:
    rng = random.Random(seed)
    strata: Dict[Tuple[int, int], List[str]] = defaultdict(list)
    for pid, info in table.items():
        strata[(info["source"], info["label"])].append(pid)
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
            k = 0
        holdout.extend(members[:k])
    return sorted(holdout)


def make_repeated_group_folds(
    table: Dict[str, Dict[str, int]], subjects: Sequence[str], n_splits: int, repeat_seed: int
) -> List[Tuple[List[str], List[str]]]:
    rng = random.Random(repeat_seed)
    ordered = list(sorted(subjects))
    rng.shuffle(ordered)
    arr = np.asarray(ordered)
    y_sub = np.asarray([table[s]["label"] for s in ordered])
    gkf = GroupKFold(n_splits=min(n_splits, len(ordered)))
    folds: List[Tuple[List[str], List[str]]] = []
    for tr_i, va_i in gkf.split(arr, y_sub, groups=arr):
        tr_s, va_s = list(arr[tr_i]), list(arr[va_i])
        assert not (set(tr_s) & set(va_s)), "CRITICAL: GroupKFold identity leakage detected!"
        folds.append((tr_s, va_s))
    return folds


def fit_channel_scaler(X: np.ndarray) -> StandardScaler:
    _, c, _ = X.shape
    flat = np.transpose(X, (0, 2, 1)).reshape(-1, c)
    sc = StandardScaler()
    sc.fit(flat)
    return sc


def transform_with_scaler(scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    n, c, l = X.shape
    flat = np.transpose(X, (0, 2, 1)).reshape(-1, c)
    return np.transpose(scaler.transform(flat).reshape(n, l, c), (0, 2, 1)).astype(np.float32)


def coral_alignment_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    mean_loss = (a.mean(dim=0) - b.mean(dim=0)).pow(2).sum()
    ca = a - a.mean(dim=0, keepdim=True)
    cb = b - b.mean(dim=0, keepdim=True)
    cov_a = (ca.T @ ca) / max(len(a) - 1, 1)
    cov_b = (cb.T @ cb) / max(len(b) - 1, 1)
    return mean_loss + (cov_a - cov_b).pow(2).sum()


def compute_subject_metrics(
    probs: np.ndarray, y: np.ndarray, parents: np.ndarray, threshold: float = 0.50
) -> Dict[str, Any]:
    subs = sorted(set(parents))
    subj_probs = np.asarray([float(np.mean(probs[parents == s])) for s in subs], dtype=np.float64)
    y_true = np.asarray([int(round(float(np.mean(y[parents == s])))) for s in subs], dtype=int)
    preds = (subj_probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    has_both = (tp + fn) > 0 and (tn + fp) > 0
    return {
        "n_subjects": int(len(subs)),
        "n_pos": int(tp + fn),
        "n_neg": int(tn + fp),
        "accuracy": float(accuracy_score(y_true, preds)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, subj_probs)) if has_both else float("nan"),
        "average_precision": float(average_precision_score(y_true, subj_probs)) if has_both else float("nan"),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "per_subject_prob": {s: round(float(p), 5) for s, p in zip(subs, subj_probs)},
    }


def predict_batched(model: NeuroTraceDiagnosticNet, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    chunks: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            xb = torch.from_numpy(X[i:i + 256]).to(device)
            chunks.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(chunks, axis=0) if chunks else np.empty(0, dtype=np.float64)


def extract_embeddings_batched(model: NeuroTraceDiagnosticNet, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    chunks: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            xb = torch.from_numpy(X[i:i + 256]).to(device)
            chunks.append(model.extract_features(xb).cpu().numpy())
    return np.vstack(chunks) if chunks else np.empty((0, model.emb_dim), dtype=np.float32)


def train_single_model(
    cfg: V5Config,
    X_tr_raw: np.ndarray,
    y_tr: np.ndarray,
    p_tr: np.ndarray,
    s_tr: np.ndarray,
    X_va_raw: Optional[np.ndarray],
    y_va: Optional[np.ndarray],
    p_va: Optional[np.ndarray],
    device: torch.device,
    seed: int,
) -> Tuple[NeuroTraceDiagnosticNet, StandardScaler, Dict[str, Any]]:
    set_seed(seed)
    scaler = fit_channel_scaler(X_tr_raw)
    X_tr = transform_with_scaler(scaler, X_tr_raw)
    X_va = transform_with_scaler(scaler, X_va_raw) if X_va_raw is not None else None
    model = NeuroTraceDiagnosticNet(
        in_channels=X_tr.shape[1],
        conv1=cfg.conv1,
        conv2=cfg.conv2,
        lstm_hidden=cfg.lstm_hidden,
        lstm_layers=cfg.lstm_layers,
        emb_dim=cfg.emb_dim,
        dropout=cfg.dropout,
        grl_lambda=cfg.grl_lambda,
    ).to(device)
    backbone_body = [p for n, p in model.named_parameters() if not n.startswith(("device_classifier", "clinical_classifier"))]
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("device_classifier")]
    opt_main = torch.optim.AdamW([
        {"params": backbone_body, "lr": cfg.lr, "weight_decay": cfg.weight_decay},
        {"params": list(model.clinical_classifier.parameters()), "lr": cfg.lr * 4.0, "weight_decay": 0.0},
    ])
    opt_disc = torch.optim.AdamW(model.device_classifier.parameters(), lr=5e-3, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt_main, T_max=max(1, cfg.epochs))
    sub_counts = Counter(p_tr)
    inv_sub = np.asarray([1.0 / sub_counts[pid] for pid in p_tr], dtype=np.float64)
    # Exact class-balanced subject sampling probabilities
    pos_mask = y_tr == 1
    neg_mask = y_tr == 0
    sample_prob = np.zeros(len(y_tr), dtype=np.float64)
    if pos_mask.any():
        sample_prob[pos_mask] = 0.50 * inv_sub[pos_mask] / inv_sub[pos_mask].sum()
    if neg_mask.any():
        sample_prob[neg_mask] = 0.50 * inv_sub[neg_mask] / inv_sub[neg_mask].sum()
    sample_prob /= sample_prob.sum()
    has_two_devices = cfg.device_adversarial and (len(np.unique(s_tr)) >= 2)
    wac_hc_idx = np.where((y_tr == 0) & (s_tr == 0))[0]
    app_hc_idx = np.where((y_tr == 0) & (s_tr == 1))[0]
    wac_all_idx = np.where(s_tr == 0)[0]
    app_all_idx = np.where(s_tr == 1)[0]
    epoch_budget = min(len(X_tr), 512)
    dom_interval = 2
    best_score = -1.0
    best_epoch = cfg.epochs
    best_state: Optional[Dict[str, torch.Tensor]] = None
    no_improve = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        ep_indices = np.random.choice(len(X_tr), size=epoch_budget, replace=False, p=sample_prob)
        for step, start in enumerate(range(0, len(ep_indices), cfg.batch_size)):
            bi = ep_indices[start:start + cfg.batch_size]
            if len(bi) < 4:
                continue
            xb_np = X_tr[bi].copy()
            ang = random.choice((0.0,) + DEFAULT_ROTATIONS)
            if ang != 0.0:
                xb_np = rotate_xy(xb_np, ang)
            xb = torch.from_numpy(xb_np).to(device)
            yb = torch.from_numpy(y_tr[bi]).to(device)
            opt_main.zero_grad(set_to_none=True)
            clin_logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(clin_logits, yb)
            if has_two_devices and (step % dom_interval == 0) and len(wac_hc_idx) >= 4 and len(app_hc_idx) >= 4:
                n_hc = min(24, len(wac_hc_idx), len(app_hc_idx))
                dhw = np.random.choice(wac_hc_idx, size=n_hc, replace=False)
                dha = np.random.choice(app_hc_idx, size=n_hc, replace=False)
                dom_idx = np.concatenate([dhw, dha])
                xd = torch.from_numpy(X_tr[dom_idx]).to(device)
                sd = torch.cat([
                    torch.zeros(n_hc, device=device),
                    torch.ones(n_hc, device=device),
                ])
                with torch.no_grad():
                    fd = model.extract_features(xd)
                for _ in range(5):
                    opt_disc.zero_grad(set_to_none=True)
                    disc_loss = F.binary_cross_entropy_with_logits(
                        model.device_classifier(fd).squeeze(-1), sd
                    )
                    disc_loss.backward()
                    opt_disc.step()
                _, dev_logits, feat_dom = model.forward_dual(xd, lam=cfg.grl_lambda)
                adv_loss = F.binary_cross_entropy_with_logits(dev_logits, sd)
                align_loss = coral_alignment_loss(feat_dom[:n_hc], feat_dom[n_hc:2 * n_hc])
                loss = loss + 2.94 * adv_loss + 2.02 * align_loss
            loss.backward()
            nn.utils.clip_grad_norm_(backbone_params, 2.0)
            opt_main.step()
        sched.step()
        if X_va is not None and y_va is not None and p_va is not None:
            va_probs = predict_batched(model, X_va, device)
            va_met = compute_subject_metrics(va_probs, y_va, p_va, cfg.threshold)
            score = va_met["roc_auc"] if not math.isnan(va_met["roc_auc"]) else va_met["accuracy"]
            if score > best_score + 1e-5:
                best_score = score
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if not cfg.device_adversarial and epoch >= cfg.min_epochs and no_improve >= cfg.early_stop_patience:
                break
    if best_state is not None and not cfg.device_adversarial:
        model.load_state_dict(best_state)
    else:
        best_epoch = cfg.epochs
    # Training-set-only Youden-J threshold alignment onto decision boundary p = 0.50 (ZERO test peeking)
    tr_probs = predict_batched(model, X_tr, device)
    tr_subs = sorted(set(p_tr))
    sp_tr = np.asarray([float(np.mean(tr_probs[p_tr == s])) for s in tr_subs])
    y_sub_tr = np.asarray([int(round(float(np.mean(y_tr[p_tr == s])))) for s in tr_subs])
    if (y_sub_tr == 1).any() and (y_sub_tr == 0).any():
        best_j, best_th = -2.0, 0.50
        for th in np.linspace(0.15, 0.85, 141):
            j_val = float((sp_tr[y_sub_tr == 1] >= th).mean() + 1.12 * (sp_tr[y_sub_tr == 0] < th).mean() - 1.0)
            if j_val > best_j:
                best_j, best_th = j_val, float(th)
        q_th = 0.5 * (float(np.percentile(sp_tr[y_sub_tr == 1], 6)) + float(np.percentile(sp_tr[y_sub_tr == 0], 95)))
        cal_th = float(np.clip(0.5 * (best_th + q_th), 0.15, 0.85))
        with torch.no_grad():
            model.clinical_classifier.bias.sub_(math.log(cal_th / (1.0 - cal_th)))
    return model, scaler, {"best_epoch": int(best_epoch)}


def run_device_leakage_probes(
    model: NeuroTraceDiagnosticNet,
    scaler: StandardScaler,
    bundle: Dict[str, Any],
    device: torch.device,
    seed: int = 42,
) -> Dict[str, Any]:
    real_mask = ~bundle["is_virtual"]
    X_real = bundle["X"][real_mask]
    y_real = bundle["y"][real_mask]
    p_real = bundle["parents"][real_mask]
    s_real = bundle["sources"][real_mask]
    if len(np.unique(s_real)) < 2:
        return {
            "status": "skipped",
            "reason": "single hardware source",
            "mean_auc": None,
            "std_auc": None,
        }
    X_scaled = transform_with_scaler(scaler, X_real)
    emb = extract_embeddings_batched(model, X_scaled, device)
    raw_feat = np.hstack([X_real.mean(axis=2), X_real.std(axis=2)])
    # Primary Hardware Leakage Metric: 5-fold subject-grouped GroupKFold across matched
    # clinical controls (Wacom HC [n=15] vs Touchscreen HC [n=50]), isolating pure device
    # confounding free of P(PD | Device) label-shift bias (Zhao et al., ICML 2019).
    hc_mask = y_real == 0
    gkf = GroupKFold(n_splits=5)
    emb_hc_aucs: List[float] = []
    raw_hc_aucs: List[float] = []
    for tr_i, va_i in gkf.split(emb[hc_mask], s_real[hc_mask], p_real[hc_mask]):
        clf_e = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf_e.fit(emb[hc_mask][tr_i], s_real[hc_mask][tr_i])
        emb_hc_aucs.append(float(roc_auc_score(s_real[hc_mask][va_i], clf_e.predict_proba(emb[hc_mask][va_i])[:, 1])))
        clf_r = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf_r.fit(raw_feat[hc_mask][tr_i], s_real[hc_mask][tr_i])
        raw_hc_aucs.append(float(roc_auc_score(s_real[hc_mask][va_i], clf_r.predict_proba(raw_feat[hc_mask][va_i])[:, 1])))
    emb_all_aucs: List[float] = []
    for tr_i, va_i in gkf.split(emb, s_real, p_real):
        clf_a = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf_a.fit(emb[tr_i], s_real[tr_i])
        emb_all_aucs.append(float(roc_auc_score(s_real[va_i], clf_a.predict_proba(emb[va_i])[:, 1])))
    return {
        "status": "ok",
        "mean_auc": float(np.mean(emb_hc_aucs)),
        "std_auc": float(np.std(emb_hc_aucs)),
        "fold_aucs": [round(a, 5) for a in emb_hc_aucs],
        "raw_input_hc_probe_auc": float(np.mean(raw_hc_aucs)),
        "unconditioned_all_subjects_auc": float(np.mean(emb_all_aucs)),
        "theoretical_prevalence_floor_auc": round(1.0 - 15.0 / 84.0, 5),
    }


# ===========================================================================
# 5. CLI & Full Diagnostic Pipeline Execution


# ===========================================================================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NeuroTrace v5 Diagnostic & Adversarial Trainer")
    p.add_argument("--project-root", type=str, default="/content/drive/MyDrive/NEUROTRACE")
    p.add_argument("--source-filter", type=str, default="all", choices=("all", "uci", "app", "all+virtual"))
    p.add_argument("--max-windows-per-subject", type=int, default=150)
    p.add_argument("--drop-channels", type=str, default="")
    p.add_argument("--device-adversarial", nargs="?", const=True, default=False, type=parse_bool)
    p.add_argument("--grl-lambda", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--save-model", type=parse_bool, default=True)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    t_start = time.time()
    args = build_argparser().parse_args(argv)
    dev, epochs, batch_size, dev_note = resolve_device_and_defaults(
        args.device, args.epochs, args.batch_size
    )
    cfg = V5Config(
        project_root=args.project_root,
        source_filter=args.source_filter,
        max_windows_per_subject=args.max_windows_per_subject,
        drop_channels=parse_int_list(args.drop_channels),
        device_adversarial=bool(args.device_adversarial),
        grl_lambda=float(args.grl_lambda),
        epochs=epochs,
        batch_size=batch_size,
        repeats=args.repeats,
        n_splits=args.n_splits,
        seed=args.seed,
        tag=args.tag,
        device_pref=args.device,
        resolved_device=str(dev),
        save_model=bool(args.save_model),
    )
    print("=" * 88)
    print(f" NEUROTRACE v5 DIAGNOSTIC PIPELINE — TAG: {cfg.tag}")
    print("=" * 88)
    print(f"  Device Adaptation    : {dev_note}")
    print(f"  Source Filter        : {cfg.source_filter} | Max Windows/Subject: {cfg.max_windows_per_subject}")
    print(f"  Dropped Channels     : {list(cfg.drop_channels)} | GRL Adversarial: {cfg.device_adversarial} (lambda={cfg.grl_lambda})")
    print(f"  Schedule             : epochs={cfg.epochs}, batch_size={cfg.batch_size}, repeats={cfg.repeats}, n_splits={cfg.n_splits}")
    bundle, stats = build_v5_dataset(cfg)
    X, y, parents, sources, is_virt = (
        bundle["X"], bundle["y"], bundle["parents"], bundle["sources"], bundle["is_virtual"]
    )
    print(f"  Effective Profiles   : {stats['n_effective_profiles']} ({stats['n_base_subjects']} unique human subjects)")
    print(f"  Active Channels ({bundle['in_channels']})  : {bundle['feature_names']}")
    print(f"  Window Skew Guard    : {stats['raw_windows_uncapped']} raw (max {stats['raw_max_windows_single_subject']}/subj) -> {len(X)} capped")
    table = build_subject_table(parents, y, sources)
    holdout_subjects = make_holdout_subjects(table, cfg.holdout_fraction, cfg.seed)
    holdout_set = set(holdout_subjects)
    cv_subjects = sorted(s for s in table if s not in holdout_set)
    assert not (set(cv_subjects) & holdout_set), "Holdout overlap detected!"
    cv_rows: List[Dict[str, Any]] = []
    for r in range(cfg.repeats):
        rep_seed = cfg.seed + r * 100
        folds = make_repeated_group_folds(table, cv_subjects, cfg.n_splits, rep_seed)
        for fold_idx, (tr_subs, va_subs) in enumerate(folds, start=1):
            tr_mask = np.isin(parents, tr_subs)
            # Validation evaluates strictly on real human windows of held-out validation subjects
            va_mask = np.isin(parents, va_subs) & (~is_virt)
            assert not (set(parents[tr_mask]) & set(parents[va_mask])), "Fold parent leakage!"
            model_f, scaler_f, info_f = train_single_model(
                cfg,
                X[tr_mask], y[tr_mask], parents[tr_mask], sources[tr_mask],
                X[va_mask], y[va_mask], parents[va_mask],
                dev,
                seed=rep_seed + fold_idx,
            )
            va_probs = predict_batched(model_f, transform_with_scaler(scaler_f, X[va_mask]), dev)
            met = compute_subject_metrics(va_probs, y[va_mask], parents[va_mask], cfg.threshold)
            row = {
                "repeat": r + 1,
                "fold": fold_idx,
                "best_epoch": info_f["best_epoch"],
                **{k: met[k] for k in ("n_subjects", "accuracy", "f1", "sensitivity", "specificity", "roc_auc")},
            }
            cv_rows.append(row)
            print(
                f"  [Repeat {r+1}/{cfg.repeats} | Fold {fold_idx}/{len(folds)}] "
                f"acc={met['accuracy']:.3f} | f1={met['f1']:.3f} | "
                f"sens={met['sensitivity']:.3f} | spec={met['specificity']:.3f} | auc={met['roc_auc']:.3f}",
                flush=True,
            )

    def _agg_cv(key: str) -> Tuple[float, float]:
        vals = [float(r[key]) for r in cv_rows if r[key] is not None and not math.isnan(float(r[key]))]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), float("nan"))
    cv_summary = {}
    for k in ("accuracy", "f1", "sensitivity", "specificity", "roc_auc"):
        m_val, s_val = _agg_cv(k)
        cv_summary[f"{k}_mean"] = m_val
        cv_summary[f"{k}_std"] = s_val
    # Final Production / Candidate Model trained on all CV subjects and evaluated on unseen holdout
    full_tr_mask = np.isin(parents, cv_subjects)
    ho_mask = np.isin(parents, holdout_subjects) & (~is_virt)
    assert not (set(parents[full_tr_mask]) & set(parents[ho_mask])), "Holdout parent leakage!"
    final_model, final_scaler, _ = train_single_model(
        cfg,
        X[full_tr_mask], y[full_tr_mask], parents[full_tr_mask], sources[full_tr_mask],
        None, None, None,
        dev,
        seed=cfg.seed,
    )

    ho_probs = predict_batched(final_model, transform_with_scaler(final_scaler, X[ho_mask]), dev)
    holdout_metrics = compute_subject_metrics(ho_probs, y[ho_mask], parents[ho_mask], cfg.threshold)
    holdout_by_source: Dict[str, Any] = {}
    for src_code, src_name in ((0, "uci_wacom"), (1, "app_touch")):
        sm = ho_mask & (sources == src_code)
        if sm.any():
            sm_probs = predict_batched(final_model, transform_with_scaler(final_scaler, X[sm]), dev)
            holdout_by_source[src_name] = compute_subject_metrics(
                sm_probs, y[sm], parents[sm], cfg.threshold
            )
    leakage_probe = run_device_leakage_probes(final_model, final_scaler, bundle, dev, seed=cfg.seed)
    elapsed_s = round(time.time() - t_start, 2)
    models_dir = Path(cfg.project_root) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = models_dir / f"neurotrace_{cfg.tag}.pth"
    scaler_path = models_dir / f"scaler_{cfg.tag}.pkl"
    report_path = models_dir / f"train_report_{cfg.tag}.json"
    if cfg.save_model:
        torch.save(
            {
                "model_version": "v5-production",
                "tag": cfg.tag,
                "model_state_dict": {k: v.cpu() for k, v in final_model.state_dict().items()},
                "model_kwargs": {
                    "in_channels": final_model.in_channels,
                    "conv1": final_model.conv1,
                    "conv2": final_model.conv2,
                    "lstm_hidden": final_model.lstm_hidden,
                    "lstm_layers": final_model.lstm_layers,
                    "emb_dim": final_model.emb_dim,
                    "dropout": final_model.dropout,
                    "grl_lambda": final_model.grl_lambda,
                },
                "feature_names": bundle["feature_names"],
                "in_channels": bundle["in_channels"],
                "seq_len": cfg.seq_len,
                "scaler_mean": [float(v) for v in final_scaler.mean_],
                "scaler_scale": [float(v) for v in final_scaler.scale_],
                "classification_threshold": cfg.threshold,
                "cv_summary": cv_summary,
                "holdout_same_device": holdout_metrics,
                "holdout_by_source": holdout_by_source,
                "device_leakage_probe": leakage_probe,
                "disclaimer": MEDICAL_DISCLAIMER,
            },
            ckpt_path,
        )
        with open(scaler_path, "wb") as fh:
            pickle.dump(final_scaler, fh)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_s,
        "tag": cfg.tag,
        "config": asdict(cfg),
        "device_note": dev_note,
        "dataset": {
            "n_base_subjects": int(stats["n_base_subjects"]),
            "n_effective_profiles": int(stats["n_effective_profiles"]),
            "raw_windows_uncapped": int(stats["raw_windows_uncapped"]),
            "raw_max_windows_single_subject": int(stats["raw_max_windows_single_subject"]),
            "n_windows": int(len(X)),
            "in_channels": bundle["in_channels"],
            "feature_names": bundle["feature_names"],
            "zero_subject_identity_leakage": True,
        },
        "cv_summary": cv_summary,
        "cv_folds": cv_rows,
        "holdout_subjects": holdout_subjects,
        "holdout_same_device": holdout_metrics,
        "holdout_by_source": holdout_by_source,
        "device_leakage_probe": leakage_probe,
        "artifacts": {
            "checkpoint": str(ckpt_path) if cfg.save_model else None,
            "scaler_pkl": str(scaler_path) if cfg.save_model else None,
            "report_json": str(report_path),
        },
        "disclaimer": MEDICAL_DISCLAIMER,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    leak_str = (
        "N/A"
        if leakage_probe.get("mean_auc") is None
        else f"{leakage_probe['mean_auc']:.3f} +/- {leakage_probe['std_auc']:.3f}"
    )
    print("-" * 88)
    print(f"  CV Summary ({len(cv_rows)} runs) : acc={cv_summary['accuracy_mean']:.3f} | auc={cv_summary['roc_auc_mean']:.3f} | sens={cv_summary['sensitivity_mean']:.3f} | spec={cv_summary['specificity_mean']:.3f}")
    print(f"  Holdout (n={holdout_metrics['n_subjects']})        : acc={holdout_metrics['accuracy']:.3f} | auc={holdout_metrics['roc_auc']:.3f} | sens={holdout_metrics['sensitivity']:.3f} | spec={holdout_metrics['specificity']:.3f}")
    print(f"  Device Leakage Probe : {leak_str} | Runtime: {elapsed_s:.1f}s")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
