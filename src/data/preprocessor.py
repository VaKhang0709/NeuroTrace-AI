"""V5 training-compatible signal processing (timestamps in seconds).

The legacy uniform grid is approximately 100 Hz (including both endpoints),
not an exactly spaced arange grid. Preserve it to avoid checkpoint input drift.
Fit scaler statistics on training subjects only, never validation/holdout data.
"""
from types import SimpleNamespace
from typing import List
import json
from pathlib import Path
import numpy as np
from scipy import signal

SOS_LP_12HZ = signal.butter(4, 12.0, btype="low", fs=100.0, output="sos")

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


def extract_trial_windows(trial, cfg) -> List[np.ndarray]:
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



class V5Preprocessor:
    """Raw trajectory to standardized (windows, 5, 256) float32 tensors.

    Window geometry/pressure/dt normalization precedes training-set Z-scores.
    No fitting is performed at inference. Missing pressure uses the V5 proxy.
    """
    def __init__(self, mean, scale, window=256, stride=64):
        self.mean = np.asarray(mean, dtype=np.float64).reshape(1, 5, 1)
        self.scale = np.asarray(scale, dtype=np.float64).reshape(1, 5, 1)
        if not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or np.any(self.scale <= 0):
            raise ValueError("Scaler must have finite means and positive scales")
        if window < 24 or stride < 1:
            raise ValueError("window >= 24 and stride >= 1 required")
        self.cfg = SimpleNamespace(window=window, stride=stride, seq_len=256)

    @classmethod
    def from_json(cls, path):
        cfg = json.loads(Path(path).read_text())
        if cfg['feature_names'] != ['x', 'y', 'pressure', 'velocity', 'dt']:
            raise ValueError("Expected V5 production channel order")
        return cls(cfg['mean'], cfg['scale'])

    def transform(self, x, y, timestamps, pressure=None):
        x, y, t = [np.asarray(a, dtype=np.float64) for a in (x, y, timestamps)]
        if x.ndim != 1 or x.shape != y.shape or x.shape != t.shape or len(x) < 20:
            raise ValueError("Aligned 1-D trajectories with at least 20 points required")
        if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
            raise ValueError("Timestamps must be finite, strictly increasing seconds")
        p = np.full_like(x, np.nan) if pressure is None else np.asarray(pressure, dtype=np.float64)
        if p.shape != x.shape:
            raise ValueError("Pressure must align with trajectory")
        trial = SimpleNamespace(x=x, y=y, t=t, z=np.full_like(x, np.nan),
                                pressure=p, has_real_pressure=pressure is not None)
        windows = extract_trial_windows(trial, self.cfg)
        if not windows:
            raise ValueError("No usable windows (short or stationary trajectory)")
        selected = np.stack(windows)[:, [0, 1, 3, 4, 5], :]
        return ((selected - self.mean) / self.scale).astype(np.float32)
