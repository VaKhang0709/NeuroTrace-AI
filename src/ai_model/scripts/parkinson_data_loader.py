#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parkinson_data_loader.py — HandPD-style handwriting dataset loader & simulator
parkinson_data_loader.py — Bộ nạp & mô phỏng dữ liệu nét vẽ kiểu HandPD

===========================================================================
 WHAT THIS FILE DOES / FILE NÀY LÀM GÌ
===========================================================================
1. Loads a REAL handwriting dataset if ``--data-csv`` is given
   (schema: ``subject_id, group, task, t_ms, x, y``).
   Nếu truyền ``--data-csv`` thì đọc dữ liệu THẬT (đúng schema trên).

2. Otherwise SIMULATES the HandPD acquisition protocol exactly:
   - 92 subjects: 18 healthy controls + 74 Parkinson's disease (PD)  -> imbalanced
   - 4 spiral + 4 meander trials per subject  -> 736 trials
   - 100 Hz sampling, timestamps in milliseconds
   Tự động GIẢ LẬP đúng giao thức HandPD: 92 người (18 khỏe / 74 PD — mất cân bằng),
   mỗi người 4 spiral + 4 meander, lấy mẫu 100 Hz, timestamp tính bằng ms.

3. Extracts per-trial kinematic features with the medically-motivated pipeline:
   velocity / acceleration / jerk / tremor (FFT) / curvature stability.
   Trích xuất đặc trưng động học theo từng trial: vận tốc, gia tốc, jerk,
   run (tremor) qua FFT và độ ổn định độ cong.

4. Prints descriptive statistics (mean/std/min/max of jerk & tremor frequency per
   group + class balance), draws 2 figures and exports a clean CSV.
   In bảng thống kê + vẽ 2 hình + xuất CSV sạch.

===========================================================================
 MEDICAL BACKGROUND / BỐI CẢNH Y KHOA
===========================================================================
HandPD (Pereira et al., 2016 — "Handwritten dynamics assessment through
convolutional neural networks") is one of the reference datasets for
handwriting-based Parkinson screening. Patients draw an Archimedean spiral and a
meander ("Greek key") figure on a digitising tablet; the tablet records pen
position over time.

Bệnh nhân vẽ đường xoắn ốc (spiral) và đường zic-zắc (meander) trên bảng số hoá;
máy ghi lại quỹ đạo bút theo thời gian. Bốn dấu hiệu lâm sàng được mô phỏng:

  * BRADYKINESIA (chậm vận động) — PD subjects take 9–13 s to finish a stroke vs
    6–9 s for controls. Đây là nguyên nhân chính của vận tốc trung bình thấp.
  * MICROGRAPHIA (chữ nhỏ dần) — the writing shrinks: smaller radius and a
    progressive inward drift (co-contraction). Bán kính nhỏ hơn và co dần.
  * RESTING TREMOR (run khi nghỉ / run tư thế) — 4–6 Hz oscillation with
    1–1.8 mm amplitude in PD vs a physiological 8–12 Hz micro-tremor (<0.4 mm)
    in controls. Run PD được CẤY THEO PHƯƠNG VUÔNG GÓC với hướng di chuyển.
  * LARGER MOTOR NOISE / fragmentation — PD trajectories are noisier and have
    irregular speed (dyskinesia, arhythmicity). Nhiễu vận động lớn hơn.

---------------------------------------------------------------------------
 IMPORTANT TRAPS HANDLED HERE / CÁC "BẪY" ĐÃ XỬ LÝ
---------------------------------------------------------------------------
TRAP 1 — DUPLICATED TRIAL INDICES (BẪY SỐ TRIAL TRÙNG NHAU)
    Each subject has trial 1..4 for spiral AND trial 1..4 for meander. If you
    ``groupby("trial")`` you silently merge a spiral with a meander; within the
    merged group ``t_ms`` restarts at 0, so ``dt`` becomes negative / huge and
    the derivative explodes to ~1e15. We ALWAYS group by ["task", "trial"].
    Mỗi người có trial 1..4 cho spiral VÀ 1..4 cho meander. Nếu groupby("trial")
    thì spiral bị trộn với meander, t quay về 0 → đạo hàm nổ 1e15.
    => luôn groupby(["task", "trial"]).

TRAP 2 — WHITE NOISE DESTROYS DERIVATIVES (BẪY NHIỄU TRẮNG)
    Differentiating raw 100 Hz data amplifies white noise by 1/dt at each order
    (jerk ~ noise/dt**3). We therefore compute velocity/acceleration from a
    HEAVILY smoothed trajectory (0.5 s centred moving average) and jerk from a
    LIGHTLY smoothed one (5 samples), as clinically standard.
    Đạo hàm trực tiếp trên dữ liệu thô sẽ khiến velocity/jerk "nổ" trị số, nên
    velocity/gia tốc tính từ quỹ đạo SMOOTH NẶNG (trung bình trượt 0.5 s) và
    jerk từ smooth nhẹ 5 mẫu.

TRAP 3 — TREMOR IS PERPENDICULAR TO MOTION (BẪY RUN VUÔNG GÓC)
    A naive radial-only analysis (project the trajectory on r(t)) MISSES the
    tremor component that is perpendicular to the direction of travel — which is
    exactly how intention/resting tremor appears on a spiral. We detrend EACH
    axis (x and y) separately with a centred 0.5 s moving average and SUM the two
    PSDs before looking for the 3–12 Hz peak.
    Nếu chỉ phân tích tín hiệu radial sẽ bỏ sót run vuông góc trên spiral; do đó
    ta detrend từng trục x,y bằng moving average 0.5 s rồi CỘNG PSD hai trục.

---------------------------------------------------------------------------
 OUTPUTS / ĐẦU RA
---------------------------------------------------------------------------
  data/processed/parkinson_handpd_clean.csv    — clean long-format trajectory
  data/processed/parkinson_handpd_features.csv — per-trial features
  data/processed/spiral_samples.png            — figure 1 (healthy vs PD spiral)
  data/processed/jerk_tremor_boxplots.png      — figure 2 (jerk & tremor boxplots)

USAGE / CÁCH DÙNG
    python src/ai_model/scripts/parkinson_data_loader.py
    python src/ai_model/scripts/parkinson_data_loader.py --data-csv data/raw/handpd.csv
    python src/ai_model/scripts/parkinson_data_loader.py --quick --no-plots

NOTE: ``data/`` is gitignored — the artefacts stay LOCAL, never committed.
      Thư mục ``data/`` đã bị gitignore, artefact chỉ nằm ở máy, KHÔNG commit.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Agg backend: không cần màn hình (headless) — bắt buộc cho CI/server.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must come after matplotlib.use)

# ---------------------------------------------------------------------------
# Constants / Hằng số
# ---------------------------------------------------------------------------
FS_HZ: float = 100.0            # sampling rate, HandPD-style tablet @ 100 Hz
DT_S: float = 1.0 / FS_HZ       # 10 ms between samples
HEAVY_SMOOTH_S: float = 0.5     # 0.5 s moving average (50 samples @100 Hz)
LIGHT_SMOOTH_SAMPLES: int = 5   # light smoothing for jerk (5 samples @100 Hz)
TREMOR_BAND_HZ: Tuple[float, float] = (3.0, 12.0)   # search band for the peak
PD_BAND_HZ: Tuple[float, float] = (3.0, 7.0)        # PD resting-tremor band

# Simulator protocol constants / Hằng số giao thức mô phỏng
N_SUBJECTS_DEFAULT: int = 92
N_HEALTHY_DEFAULT: int = 18     # => 18 healthy / 74 PD => imbalanced (đúng HandPD)
TRIALS_PER_TASK: int = 4        # 4 spiral + 4 meander per subject

# Clinical simulation parameters / Tham số mô phỏng lâm sàng
HEALTHY_DURATION_S = (6.0, 9.0)     # healthy drawing time / thời gian vẽ người khỏe
PD_DURATION_S = (9.0, 13.0)         # bradykinesia / chậm vận động ở PD
PD_TREMOR_HZ = (4.2, 5.8)           # PD resting tremor 4–6 Hz
HEALTHY_TREMOR_HZ = (8.2, 10.2)     # physiological tremor 8–12 Hz (mean ~9 Hz)
PD_TREMOR_AMPLITUDE_MM = (1.0, 1.8)     # run PD: biên độ 1–1.8 mm
HEALTHY_TREMOR_AMPLITUDE_MM = (0.15, 0.35)  # run sinh lý: biên độ nhỏ
PD_NOISE_SIGMA_MM = (0.35, 0.65)    # nhiễu vận động lớn hơn ở PD
HEALTHY_NOISE_SIGMA_MM = (0.15, 0.30)
PD_AMPLITUDE_SCALE = (0.72, 0.88)   # micrographia: biên độ nhỏ hơn
HEALTHY_AMPLITUDE_SCALE = (0.98, 1.05)
PD_INWARD_SHRINK = (0.10, 0.22)     # micrographia: co dần theo tiến trình vẽ
HEALTHY_INWARD_SHRINK = (0.0, 0.02)

GROUP_TO_LABEL: Dict[str, int] = {
    # 0 = Healthy / control, 1 = Parkinson's disease
    "healthy": 0, "control": 0, "hc": 0, "0": 0,
    "pd": 1, "parkinson": 1, "parkinsons": 1, "patient": 1, "1": 1,
}
LABEL_TO_GROUP: Dict[int, str] = {0: "healthy", 1: "pd"}


# ===========================================================================
# Small numeric helpers / Hàm số trợ giúp
# ===========================================================================
def _moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    """Edge-safe centred moving average (same length as input).

    Trung bình trượt căn giữa, xử lý biên bằng cách "edge-pad" nên đầu ra
    cùng độ dài đầu vào và không sinh NaN ở hai đầu tín hiệu.
    """
    sig = np.asarray(signal, dtype=float)
    if sig.size == 0:
        return sig.copy()
    window = int(max(1, window))
    if window % 2 == 0:
        window += 1
    if window > sig.size:                       # window larger than signal
        window = sig.size if sig.size % 2 == 1 else max(1, sig.size - 1)
    if window <= 1:
        return sig.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    pad = window // 2
    padded = np.pad(sig, pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _gradient(y: np.ndarray, dt: float) -> np.ndarray:
    """First derivative with respect to time (centred differences)."""
    return np.gradient(np.asarray(y, dtype=float), dt)


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """numpy 2.x renamed trapz -> trapezoid; support both / hỗ trợ cả hai."""
    fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(fn(y, x))


def _cumulative_arclength(points: np.ndarray) -> np.ndarray:
    """Cumulative arc length of a polyline (N, 2) -> (N,)."""
    deltas = np.diff(points, axis=0)
    seg = np.hypot(deltas[:, 0], deltas[:, 1])
    return np.concatenate([[0.0], np.cumsum(seg)])


def _resample_polyline(points: np.ndarray, n_out: int) -> np.ndarray:
    """Resample a polyline to ``n_out`` points evenly spaced in ARC LENGTH.

    Isolates the geometric shape from the speed profile: the simulator first
    builds a constant-speed path, then warps time to create bradykinesia and
    movement fragmentation (see ``_speed_warp``).
    """
    s = _cumulative_arclength(points)
    if s[-1] <= 1e-12:                          # degenerate (all points equal)
        return np.repeat(points[:1], n_out, axis=0)
    s_uniform = np.linspace(0.0, s[-1], n_out)
    return np.column_stack([
        np.interp(s_uniform, s, points[:, 0]),
        np.interp(s_uniform, s, points[:, 1]),
    ])


# ===========================================================================
# Geometric primitives / Hình học cơ bản
# ===========================================================================
def _spiral_geometry(n_out: int, turns: float = 4.5, radius_mm: float = 55.0) -> np.ndarray:
    """Archimedean spiral / Đường xoắn ốc Archimedes: r = b * theta.

    HandPD's spiral is drawn from the centre outwards; the curvature kappa of an
    Archimedean spiral is (nearly) constant, so the spread of kappa is a clean
    marker of tremor/noise and of the inward micrographic drift.
    Đường xoắn ốc Archimedes vẽ từ trong ra ngoài; độ cong gần như không đổi nên
    độ lệch chuẩn của kappa phản ánh run và hiện tượng co chữ.
    """
    theta = np.linspace(0.0, 2.0 * np.pi * turns, n_out * 3)
    r = radius_mm * theta / theta[-1]
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def _meander_geometry(
    n_out: int, width_mm: float = 90.0, height_mm: float = 45.0, n_rows: int = 6,
    u: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Meander / "Greek key" figure — a smooth serpentine of ``n_rows`` rows.

        x(u) = (W/2) * sin(2*pi*k*u)          k = n_rows / 2 lateral half-cycles
        y(u) = -(H/2) * cos(pi*u)             slow bottom-to-top sweep

    Clinical role: the meander stresses rhythmic distal control; PD patients break
    the pattern, drift vertically, slow down and inject tremor.
    Meander kiểm tra khả năng điều khiển nhịp nhàng của ngón tay; bệnh nhân PD
    hay lệch dòng, gãy nét, chậm lại và xuất hiện run.

    WHY SMOOTH (no sharp corners)? / VÌ SAO LÀM MƯỢT (không có góc nhọn)?
    A polyline with 90-degree corners has a discontinuous tangent; that kink is a
    broadband impulse which spreads energy across the whole 3–12 Hz tremor band
    and would *mask the very tremor we want to measure*. The C-infinity serpentine
    above has essentially a single lateral line (k/T Hz) so the band stays clean.
    Đường gấp khúc có tiếp tuyến không liên tục → tạo xung phổ rộng phủ kín dải
    run 3–12 Hz và che mất chính tín hiệu run cần đo. Serpentine trơn ở trên chỉ
    có một vạch phổ ngang (k/T Hz) nên dải run vẫn sạch.
    """
    if u is None:
        u = np.linspace(0.0, 1.0, n_out * 3)
    else:
        u = np.asarray(u, dtype=float)
    k = max(1, int(round(n_rows / 2.0)))
    x = (width_mm / 2.0) * np.sin(2.0 * np.pi * k * u)
    y = -(height_mm / 2.0) * np.cos(np.pi * u)
    return np.column_stack([x, y])


# ===========================================================================
# Simulator / Mô phỏng
# ===========================================================================
def _speed_warp(
    t: np.ndarray, rng: np.random.Generator, is_pd: bool
) -> np.ndarray:
    """Build the normalised progress ``u(t)`` of the pen along the path.

    Bradykinesia and movement fragmentation are modelled as a *time warp*:
    ``u`` is the normalised cumulative sum of a positive speed weight. Controls
    have an almost flat weight (regular rhythm); PD adds slow oscillatory drift
    and 1.2–2.5 Hz fragmentation (arhythmicity / dyskinesia).
    Mô hình hoá chậm vận động & gãy nhịp bằng biến đổi thời gian: trọng số tốc độ
    gần phẳng ở người khoẻ, còn PD có dao động chậm + gãy nhịp 1.2–2.5 Hz.
    """
    base = rng.uniform(0.25, 0.75)
    warp = 1.0 + 0.10 * np.sin(2.0 * np.pi * base * t + rng.uniform(0, 2 * np.pi))
    if is_pd:
        frag_f = rng.uniform(1.2, 2.5)
        frag_a = rng.uniform(0.25, 0.45)
        warp = warp + frag_a * np.sin(2.0 * np.pi * frag_f * t + rng.uniform(0, 2 * np.pi))
        warp = warp + 0.25 * rng.uniform(0, 1) * np.sin(
            2.0 * np.pi * rng.uniform(0.2, 0.5) * t + rng.uniform(0, 2 * np.pi)
        )
    warp = np.clip(warp, 0.15, None)             # keep progress monotonic
    progress = np.cumsum(warp)
    progress -= progress[0]
    if progress[-1] <= 1e-12:
        return np.linspace(0.0, 1.0, t.size)
    return progress / progress[-1]


def _add_perpendicular_tremor(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    freq_hz: float,
    amplitude_mm: float,
    rng: np.random.Generator,
    burst_modulation: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Inject a sinusoidal tremor PERPENDICULAR to the direction of travel.

    Why perpendicular? On a spiral the radial direction rotates with the drawing;
    a tremor that shows up perpendicular to the instantaneous velocity mimics the
    agonist/antagonist alternation of PD resting/postural tremor and is exactly
    the component a radial-only feature would lose.
    Vì sao vuông góc? Trên spiral, nếu chỉ chiếu lên bán kính thì thành phần run
    vuông góc với hướng vẽ (đúng dạng run tư thế/run nghỉ của PD) sẽ biến mất.
    """
    dx = _gradient(x, 1.0)          # tangent / vector tiếp tuyến (arbitrary dt)
    dy = _gradient(y, 1.0)
    norm = np.hypot(dx, dy)
    norm[norm < 1e-12] = 1e-12
    nx, ny = -dy / norm, dx / norm  # unit normal / vector pháp tuyến đơn vị

    phase = rng.uniform(0.0, 2.0 * np.pi)
    # Slow amplitude modulation (tremor bursts) — chỉ ở PD. / run thành cơn ở PD.
    envelope = np.ones_like(t)
    if burst_modulation:
        envelope = envelope + 0.25 * np.sin(
            2.0 * np.pi * rng.uniform(0.3, 0.6) * t + rng.uniform(0.0, 2.0 * np.pi)
        )
    tremor = amplitude_mm * envelope * np.sin(2.0 * np.pi * freq_hz * t + phase)
    if burst_modulation:
        # small 2nd harmonic to make the PD spectrum realistic / hài bậc 2 nhỏ
        tremor = tremor + 0.18 * amplitude_mm * np.sin(
            4.0 * np.pi * freq_hz * t + phase / 2.0
        )
    return x + nx * tremor, y + ny * tremor


@dataclass
class _TrialSpec:
    """Sampling parameters for one simulated trial / Tham số 1 trial mô phỏng."""
    duration_s: float
    amplitude_scale: float
    inward_shrink: float
    tremor_hz: float
    tremor_amp_mm: float
    noise_sigma_mm: float
    is_pd: bool


def _draw_trial(
    task: str, spec: _TrialSpec, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render one trial -> (t_ms, x_mm, y_mm).

    ``rng`` is consumed in a FIXED order (shape params, time warp, tremor, noise,
    drift) so that a given seed always reproduces the same dataset; reordering
    those calls would change every simulated value.
    ``rng`` được dùng theo thứ tự CỐ ĐỊNH (tham số hình học, biến đổi thời gian,
    run, nhiễu, trôi) nên cùng một seed luôn cho cùng một bộ dữ liệu.
    """
    n = int(round(spec.duration_s * FS_HZ)) + 1
    t = np.arange(n, dtype=float) / FS_HZ                      # seconds
    t_ms = np.round(t * 1000.0).astype(np.int64)               # milliseconds

    # 1) geometric shape with constant speed / hình học tốc độ đều
    if task == "spiral":
        base = _spiral_geometry(n, turns=rng.uniform(4.0, 5.0))
    elif task == "meander":
        base = _meander_geometry(n, n_rows=int(rng.integers(5, 8)))
    else:
        raise ValueError(f"Unknown task '{task}' (expected 'spiral' | 'meander')")

    # 2) micrographia: global scale + progressive inward drift
    #    (co-contraction makes the loops shrink as the trial advances)
    progress_geom = np.linspace(0.0, 1.0, base.shape[0])[:, None]
    base = base * spec.amplitude_scale * (1.0 - spec.inward_shrink * progress_geom)

    # 3) time warp (bradykinesia + fragmentation) -> sample the path
    u = _speed_warp(t, rng, spec.is_pd)
    s = _cumulative_arclength(base)
    x = np.interp(u, np.linspace(0.0, 1.0, s.size), base[:, 0])
    y = np.interp(u, np.linspace(0.0, 1.0, s.size), base[:, 1])

    # 4) tremor perpendicular to the movement direction / run vuông góc
    x, y = _add_perpendicular_tremor(
        x, y, t, spec.tremor_hz, spec.tremor_amp_mm, rng, spec.is_pd
    )

    # 5) motor noise: white noise + low-frequency drift (PD is noisier)
    #    nhiễu vận động: nhiễu trắng + trôi tần số thấp (PD nhiễu hơn)
    x = x + rng.normal(0.0, spec.noise_sigma_mm, size=n)
    y = y + rng.normal(0.0, spec.noise_sigma_mm, size=n)
    if spec.is_pd and rng.random() < 0.5:
        drift_f = rng.uniform(0.15, 0.4)
        drift_a = rng.uniform(0.3, 1.0)
        x = x + drift_a * np.sin(2.0 * np.pi * drift_f * t + rng.uniform(0, 2 * np.pi))
        y = y + drift_a * np.sin(2.0 * np.pi * drift_f * t + rng.uniform(0, 2 * np.pi))

    return t_ms, x, y


def _trial_spec(is_pd: bool, rng: np.random.Generator) -> _TrialSpec:
    """Draw the clinical parameters of one trial / Bốc tham số lâm sàng 1 trial."""
    if is_pd:
        return _TrialSpec(
            duration_s=rng.uniform(*PD_DURATION_S),
            amplitude_scale=rng.uniform(*PD_AMPLITUDE_SCALE),
            inward_shrink=rng.uniform(*PD_INWARD_SHRINK),
            tremor_hz=rng.uniform(*PD_TREMOR_HZ),
            tremor_amp_mm=rng.uniform(*PD_TREMOR_AMPLITUDE_MM),
            noise_sigma_mm=rng.uniform(*PD_NOISE_SIGMA_MM),
            is_pd=True,
        )
    return _TrialSpec(
        duration_s=rng.uniform(*HEALTHY_DURATION_S),
        amplitude_scale=rng.uniform(*HEALTHY_AMPLITUDE_SCALE),
        inward_shrink=rng.uniform(*HEALTHY_INWARD_SHRINK),
        tremor_hz=rng.uniform(*HEALTHY_TREMOR_HZ),
        tremor_amp_mm=rng.uniform(*HEALTHY_TREMOR_AMPLITUDE_MM),
        noise_sigma_mm=rng.uniform(*HEALTHY_NOISE_SIGMA_MM),
        is_pd=False,
    )


def simulate_handpd_strokes(
    n_subjects: int = N_SUBJECTS_DEFAULT,
    n_healthy: int = N_HEALTHY_DEFAULT,
    trials_per_task: int = TRIALS_PER_TASK,
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Simulate the HandPD drawing protocol -> long-format DataFrame.

    Returns columns / Trả về các cột:
        subject_id : int   — 1..n_subjects
        group      : str   — 'healthy' | 'pd'
        label      : int   — 0 = Healthy, 1 = Parkinson  (encode label)
        task       : str   — 'spiral' | 'meander'
        trial      : int   — 1..trials_per_task  (TRÙNG giữa 2 task — xem TRAP 1)
        t_ms       : int   — milliseconds since trial start (0, 10, 20, ...)
        x, y       : float — pen position in millimetres / toạ độ bút (mm)
    """
    if not 0 <= n_healthy <= n_subjects:
        raise ValueError("n_healthy must satisfy 0 <= n_healthy <= n_subjects")

    rng = np.random.default_rng(seed)
    n_pd = n_subjects - n_healthy
    rows = []

    for subject_id in range(1, n_subjects + 1):
        is_pd = subject_id > n_healthy        # first n_healthy subjects are controls
        group = LABEL_TO_GROUP[int(is_pd)]
        label = int(is_pd)
        # Per-subject severity: some PD subjects tremble more than others.
        # Mức độ nặng khác nhau giữa các bệnh nhân PD.
        severity = rng.uniform(0.75, 1.25) if is_pd else 1.0
        for task in ("spiral", "meander"):
            for trial in range(1, trials_per_task + 1):
                spec = _trial_spec(is_pd, rng)
                spec.tremor_amp_mm = max(0.0, spec.tremor_amp_mm * severity)
                t_ms, x, y = _draw_trial(task, spec, rng)
                rows.append(pd.DataFrame({
                    "subject_id": np.full(t_ms.size, subject_id, dtype=np.int64),
                    "group": np.full(t_ms.size, group, dtype=object),
                    "label": np.full(t_ms.size, label, dtype=np.int8),
                    "task": np.full(t_ms.size, task, dtype=object),
                    "trial": np.full(t_ms.size, trial, dtype=np.int16),
                    "t_ms": t_ms,
                    "x": x.astype(np.float32),
                    "y": y.astype(np.float32),
                }))

    df = pd.concat(rows, ignore_index=True)
    if verbose:
        n_trials = n_subjects * 2 * trials_per_task
        print(
            f"[SIMULATE] HandPD protocol: {n_subjects} subjects "
            f"({n_healthy} healthy / {n_pd} PD), {trials_per_task} spiral + "
            f"{trials_per_task} meander each => {n_trials} trials, "
            f"{len(df):,} samples @ {FS_HZ:.0f} Hz"
        )
    return df


# ===========================================================================
# Real-data loading / Nạp dữ liệu thật
# ===========================================================================
def _label_from_group(value) -> int:
    """Map any accepted group token to 0/1 / Chuẩn hoá nhóm về 0 hoặc 1."""
    key = str(value).strip().lower()
    if key not in GROUP_TO_LABEL:
        raise ValueError(
            f"Unknown group '{value}'. Accepted: healthy/control/pd/parkinson/0/1"
        )
    return GROUP_TO_LABEL[key]


def _derive_trials(df: pd.DataFrame, gap_ms: int = 500) -> np.ndarray:
    """Derive trial indices from time gaps when the CSV has no 'trial' column.

    A new trial starts when ``t_ms`` goes BACKWARDS (recording restarted) or jumps
    by more than ``gap_ms`` (long pause between drawings) — computed per
    (subject, task) so spiral and meander trials are counted separately.
    Two samples sharing the same millisecond are NOT a boundary: identical
    timestamps are normal (100 Hz tablett, held pen, duplicate rows) and splitting
    on them would fragment a single stroke.
    Suy ra chỉ số trial khi CSV không có cột 'trial': trial mới khi thời gian LÙI
    LẠI hoặc nhảy > gap_ms — tính riêng theo từng (subject, task). Hai mẫu trùng
    mili-giây KHÔNG được coi là ranh giới trial.

    LIMITATION / HẠN CHẾ: if the CSV restarts timestamps at 0 for every stroke but
    reuses exactly the same values, adjacent strokes are indistinguishable from
    time alone and stay merged. Provide a ``trial`` column for exact boundaries.
    Nếu CSV đánh lại t từ 0 cho từng nét nhưng không có cột ``trial``, các nét
    liền nhau không thể tách chỉ từ thời gian — hãy bổ sung cột ``trial``.

    Returns a plain NumPy array in the CURRENT ROW ORDER (positional), so the
    caller must assign it positionally — never as a pandas Series, because after
    ``sort_values`` the index labels no longer match the row order and pandas
    would silently realign them.
    Trả về mảng NumPy theo ĐÚNG THỨ TỰ DÒNG hiện tại (positional); nếu trả về
    Series thì pandas sẽ căn theo nhãn index (đã bị đảo sau sort_values) và làm
    sai toàn bộ chỉ số trial.
    """
    trials = np.zeros(len(df), dtype=np.int32)
    t_all = df["t_ms"].to_numpy()
    for _, idx in df.groupby(["subject_id", "task"], sort=False).indices.items():
        idx = np.sort(np.asarray(idx))
        t = t_all[idx]
        dt_ms = np.diff(t)
        starts = np.concatenate([[True], (dt_ms < 0) | (dt_ms > gap_ms)])
        trials[idx] = np.cumsum(starts)
    return trials


def load_parkinson_csv(csv_path: str | Path, verbose: bool = True) -> pd.DataFrame:
    """Load a REAL dataset CSV with schema subject_id, group, task, t_ms, x, y.

    Dữ liệu thật cần đúng schema ``subject_id, group, task, t_ms, x, y``.
    Nhóm hợp lệ: healthy / control / pd / parkinson / 0 / 1.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"data-csv not found / không tìm thấy: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"subject_id", "group", "task", "t_ms", "x", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns / thiếu cột: {sorted(missing)}")

    df = df.copy()
    df["subject_id"] = df["subject_id"].astype(str)
    df["task"] = df["task"].astype(str).str.strip().str.lower()
    df["t_ms"] = pd.to_numeric(df["t_ms"], errors="coerce").astype("int64")
    df["x"] = pd.to_numeric(df["x"], errors="coerce").astype(float)
    df["y"] = pd.to_numeric(df["y"], errors="coerce").astype(float)
    df["label"] = df["group"].map(_label_from_group).astype(np.int8)
    df["group"] = df["label"].map(LABEL_TO_GROUP).astype(object)

    if "trial" in df.columns:
        df["trial"] = df["trial"].astype(np.int32)
    else:
        df = df.sort_values(["subject_id", "task", "t_ms"], kind="stable")
        df["trial"] = _derive_trials(df)

    # Basic QC / kiểm tra sơ bộ
    n_before = len(df)
    df = df.dropna(subset=["x", "y", "t_ms"]).reset_index(drop=True)
    df = df.sort_values(["subject_id", "task", "trial", "t_ms"], kind="stable")
    df = df.reset_index(drop=True)

    if verbose:
        print(
            f"[LOAD] {csv_path.name}: {df['subject_id'].nunique()} subjects, "
            f"{df.groupby(['subject_id', 'task', 'trial']).ngroups} trials, "
            f"{n_before - len(df)} bad rows dropped"
        )
    return df


def fetch_parkinson_dataset(
    data_csv: Optional[str | Path] = None,
    n_subjects: int = N_SUBJECTS_DEFAULT,
    n_healthy: int = N_HEALTHY_DEFAULT,
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load the real dataset if ``data_csv`` is given, else simulate HandPD.

    Nếu có ``data_csv`` -> đọc dữ liệu THẬT; nếu không -> GIẢ LẬP đúng giao thức.
    """
    if data_csv:
        return load_parkinson_csv(data_csv, verbose=verbose)
    return simulate_handpd_strokes(
        n_subjects=n_subjects, n_healthy=n_healthy, seed=seed, verbose=verbose
    )


# ===========================================================================
# Feature extraction / Trích xuất đặc trưng theo trial
# ===========================================================================
FEATURE_COLUMNS: list = [
    # canonical output schema of extract_trial_features — every trial gets a row,
    # short/degenerate trials keep NaN instead of dropping the column
    # (lược đồ cột cố định: trial quá ngắn vẫn có dòng với NaN, không mất cột)
    "subject_id", "group", "label", "task", "trial", "n_samples", "duration_s",
    "sampling_hz", "tremor_peak_freq_hz", "tremor_peak_power",
    "tremor_band_ratio_3_7", "spiral_curvature_stability", "curvature_std",
    "path_length_mm", "mean_velocity", "velocity_cv", "mean_accel", "mean_jerk",
    "log10_mean_jerk", "mean_radius_mm", "radius_late_early_ratio",
]


def extract_trial_features(df: pd.DataFrame, fs_hz: float = FS_HZ) -> pd.DataFrame:
    """Extract per-trial kinematic features (one row per task/trial).

    IMPORTANT / QUAN TRỌNG: grouping is ALWAYS ["subject_id", "task", "trial"].
    Grouping by "trial" alone would mix a spiral with a meander whose ``t_ms``
    restarts at 0 -> the finite differences explode (~1e15). See TRAP 1.

    Luôn groupby(["subject_id", "task", "trial"]). Nếu chỉ groupby("trial") thì
    spiral bị trộn với meander, t quay về 0 và đạo hàm nổ 1e15.
    """
    heavy_win = max(3, int(round(HEAVY_SMOOTH_S * fs_hz)) | 1)   # 51 samples @100 Hz
    light_win = LIGHT_SMOOTH_SAMPLES | 1                        # 5 samples
    dt = 1.0 / fs_hz
    records = []

    group_cols = ["subject_id", "task", "trial"]
    for (subject_id, task, trial), part in df.groupby(group_cols, sort=False):
        part = part.sort_values("t_ms", kind="stable")
        t_sec = part["t_ms"].to_numpy(dtype=float) / 1000.0
        x_raw = part["x"].to_numpy(dtype=float)
        y_raw = part["y"].to_numpy(dtype=float)
        n = x_raw.size

        rec = {
            "subject_id": subject_id,
            "group": part["group"].iloc[0],
            "label": int(part["label"].iloc[0]),
            "task": task,
            "trial": int(trial),
            "n_samples": n,
            "duration_s": float(t_sec[-1] - t_sec[0]) if n > 1 else 0.0,
            "sampling_hz": fs_hz,
        }
        if n < max(heavy_win, light_win) + 2:
            records.append(rec)                     # too short -> NaNs
            continue

        # --- HEAVY smoothing (0.5 s) for velocity / acceleration -------------
        # Smoothing nặng để triệt nhiễu trắng trước khi lấy đạo hàm.
        x_heavy = _moving_average(x_raw, heavy_win)
        y_heavy = _moving_average(y_raw, heavy_win)
        vx, vy = _gradient(x_heavy, dt), _gradient(y_heavy, dt)
        speed = np.hypot(vx, vy)
        ax, ay = _gradient(vx, dt), _gradient(vy, dt)
        accel = np.hypot(ax, ay)

        # --- LIGHT smoothing (5 samples) for jerk ---------------------------
        # Jerk = đạo hàm bậc 3 của vị trí = tốc độ thay đổi gia tốc; phản ánh
        # độ "mượt" của chuyển động. SMOOTH NHẸ 5 mẫu để không xoá mất run 5 Hz.
        x_light = _moving_average(x_raw, light_win)
        y_light = _moving_average(y_raw, light_win)
        vx_l, vy_l = _gradient(x_light, dt), _gradient(y_light, dt)
        ax_l, ay_l = _gradient(vx_l, dt), _gradient(vy_l, dt)
        jx, jy = _gradient(ax_l, dt), _gradient(ay_l, dt)
        jerk = np.hypot(jx, jy)

        # --- Tremor via FFT (TRAP 3) ----------------------------------------
        # Detrend EACH axis with a centred 0.5 s moving average, then SUM the two
        # PSDs. Rationale: PD tremor is injected perpendicular to motion, so a
        # radial-only (r(t)) spectrum would miss it.
        # Detrend từng trục rồi CỘNG PSD x,y — tránh bỏ sót run vuông góc.
        res_x = x_raw - _moving_average(x_raw, heavy_win)
        res_y = y_raw - _moving_average(y_raw, heavy_win)
        freq, psd_x = _periodogram(res_x, fs_hz)
        _, psd_y = _periodogram(res_y, fs_hz)
        psd_sum = psd_x + psd_y

        band = (freq >= TREMOR_BAND_HZ[0]) & (freq <= TREMOR_BAND_HZ[1])
        pd_band = (freq >= PD_BAND_HZ[0]) & (freq <= PD_BAND_HZ[1])
        if np.any(band) and _trapz(psd_sum[band], freq[band]) > 0:
            f_band = freq[band]
            p_band = psd_sum[band]
            k = int(np.argmax(p_band))
            rec["tremor_peak_freq_hz"] = float(f_band[k])
            rec["tremor_peak_power"] = float(p_band[k])
            total = _trapz(p_band, f_band)
            rec["tremor_band_ratio_3_7"] = (
                _trapz(psd_sum[pd_band], freq[pd_band]) / total if total > 0 else np.nan
            )
        else:
            rec["tremor_peak_freq_hz"] = np.nan
            rec["tremor_peak_power"] = np.nan
            rec["tremor_band_ratio_3_7"] = np.nan

        # --- Geometry: path length, curvature stability, micrographia -------
        path_len = float(np.sum(np.hypot(np.diff(x_heavy), np.diff(y_heavy))))
        vx_s, vy_s = _gradient(x_heavy, dt), _gradient(y_heavy, dt)
        ax_s, ay_s = _gradient(vx_s, dt), _gradient(vy_s, dt)
        denom = np.power(np.hypot(vx_s, vy_s), 3.0)
        denom[denom < 1e-9] = 1e-9
        kappa = np.abs(vx_s * ay_s - vy_s * ax_s) / denom      # |kappa| [1/mm]
        # spiral_curvature_stability = 1/(1+std(kappa)) -> 1 = perfectly steady
        # Độ ổn định độ cong: đường xoắn ốc khoẻ mạnh có kappa ổn định (std nhỏ);
        # run và hiện tượng co chữ làm kappa dao động -> chỉ số giảm.
        rec["spiral_curvature_stability"] = float(1.0 / (1.0 + np.std(kappa)))
        rec["curvature_std"] = float(np.std(kappa))
        rec["path_length_mm"] = path_len
        rec["mean_velocity"] = float(np.mean(speed))
        rec["velocity_cv"] = float(np.std(speed) / (np.mean(speed) + 1e-9))
        rec["mean_accel"] = float(np.mean(accel))
        rec["mean_jerk"] = float(np.mean(jerk))
        rec["log10_mean_jerk"] = float(np.log10(np.mean(jerk) + 1e-12))

        cx, cy = float(np.mean(x_heavy)), float(np.mean(y_heavy))
        radius_t = np.hypot(x_heavy - cx, y_heavy - cy)          # R(t) around centroid
        rec["mean_radius_mm"] = float(np.mean(radius_t))
        if n >= 8:
            q = max(1, n // 4)
            early = float(np.mean(radius_t[:q]))
            late = float(np.mean(radius_t[-q:]))
            # radius_late_early_ratio < 1 => the drawing shrinks (micrographia)
            rec["radius_late_early_ratio"] = late / (early + 1e-9)
        else:
            rec["radius_late_early_ratio"] = np.nan

        records.append(rec)

    features = pd.DataFrame(records)
    # Guarantee the canonical schema even for very short trials.
    # Đảm bảo đủ cột chuẩn kể cả khi có trial quá ngắn.
    for col in FEATURE_COLUMNS:
        if col not in features.columns:
            features[col] = np.nan
    features = features[FEATURE_COLUMNS]
    return features


def _periodogram(signal: np.ndarray, fs_hz: float):
    """Welch-free PSD via scipy.signal.periodogram, Hann window.

    Fallback to a hand-rolled FFT if scipy is unavailable (should not happen —
    scipy is in requirements.txt).
    """
    try:
        from scipy.signal import periodogram
        freq, psd = periodogram(signal, fs=fs_hz, window="hann", detrend="linear",
                                scaling="density")
        return np.asarray(freq), np.asarray(psd)
    except Exception:  # pragma: no cover - defensive fallback
        n = signal.size
        win = np.hanning(n)
        spec = np.fft.rfft(signal * win)
        freq = np.fft.rfftfreq(n, d=1.0 / fs_hz)
        psd = (np.abs(spec) ** 2) / (fs_hz * np.sum(win ** 2))
        return freq, psd


# ===========================================================================
# Reporting / Báo cáo
# ===========================================================================
def print_group_statistics(features: pd.DataFrame) -> pd.DataFrame:
    """Print mean/std/min/max of jerk & tremor freq per group + class balance.

    In bảng thống kê mean/std/min/max của jerk & tremor freq theo nhóm,
    kèm tỷ lệ cân bằng lớp (class balance).
    """
    print("\n" + "=" * 78)
    print("PER-TRIAL STATISTICS BY GROUP / THỐNG KÊ THEO NHÓM")
    print("=" * 78)
    stats = (
        features.groupby("group")[["mean_jerk", "tremor_peak_freq_hz",
                                   "spiral_curvature_stability", "mean_velocity"]]
        .agg(["mean", "std", "min", "max"])
        .round(4)
    )
    print(stats.to_string() if not stats.empty else "  (no per-trial features — "
          "trials too short? / chưa có đặc trưng, trial quá ngắn?)")

    # Log-scale jerk keeps the table readable / jerk thang log cho dễ đọc
    log_stats = features.groupby("group")["log10_mean_jerk"].agg(
        ["mean", "std", "min", "max"]).round(4)
    print("\nlog10(mean jerk) by group / jerk theo thang log10:")
    print(log_stats.to_string() if not log_stats.empty else "  (n/a)")

    print("\nCLASS BALANCE / CÂN BẰNG LỚP")
    n_by_label = features.drop_duplicates(["subject_id"])["label"].value_counts()
    trial_by_label = features["label"].value_counts().sort_index()
    n_subj_total = int(n_by_label.sum())
    for label, name in LABEL_TO_GROUP.items():
        n_s = int((features.drop_duplicates(["subject_id"])["label"] == label).sum())
        n_t = int(trial_by_label.get(label, 0))
        pct = 100.0 * n_s / max(1, n_subj_total)
        print(f"  {label} = {name:<8} subjects: {n_s:3d} ({pct:5.1f}%)   trials: {n_t}")
    print(f"  imbalanced ratio / tỷ lệ mất cân bằng ≈ 1 : "
          f"{n_by_label.get(1, 0) / max(1, n_by_label.get(0, 0)):.2f}")
    return stats


def check_trial_grouping_integrity(df: pd.DataFrame) -> bool:
    """Guard against TRAP 1 — verify the trial grouping is well-posed.

    Checks that inside every (subject_id, task, trial) group ``t_ms`` never goes
    backwards (excluding the boundary between groups) and reports how many
    trial ids are reused across tasks — the exact condition that makes
    ``groupby("trial")`` silently merge a spiral with a meander.

    Kiểm tra bẫy trùng số trial: trong mỗi nhóm (subject, task, trial) thì t_ms
    không được lùi lại; đồng thời cho biết số trial id bị dùng lại giữa 2 task —
    đúng điều kiện khiến groupby("trial") trộn spiral với meander.
    """
    print("\n" + "=" * 78)
    print("TRAP CHECK #1 — GROUPING BY 'trial' ALONE IS WRONG / BẪY GROUPBY('trial')")
    print("=" * 78)
    n_trial_ids = int(df["trial"].nunique())
    n_task_trial = int(df.groupby(["task", "trial"], sort=False).ngroups)
    n_groups = int(df.groupby(["subject_id", "task", "trial"], sort=False).ngroups)
    print(f"  distinct 'trial' ids              : {n_trial_ids}")
    print(f"  distinct (task, trial) pairs      : {n_task_trial}"
          f"   <-- ids are REUSED across tasks / id bị dùng lại giữa 2 task")
    print(f"  groupby(['subject_id','task','trial']) -> {n_groups} valid trials")

    ordered = df.sort_values(["subject_id", "task", "trial", "t_ms"], kind="stable")
    gid = ordered.groupby(["subject_id", "task", "trial"], sort=False).ngroup()
    dt_ms = ordered["t_ms"].diff()
    same_group = (gid.diff() == 0).fillna(False)
    n_backwards = int(((dt_ms < 0) & same_group).sum())
    print(f"  time going backwards inside a trial: {n_backwards}"
          f"  {'(OK)' if n_backwards == 0 else '(BROKEN / SAI)'}")
    return n_backwards == 0


def demonstrate_grouping_trap(df: pd.DataFrame, subject=None) -> None:
    """Show the 1e15 derivative explosion caused by ``groupby("trial")``.

    Minh hoạ đạo hàm "nổ" (~1e15) khi groupby("trial") trộn spiral + meander:
    t quay về 0 nên dt âm → velocity sai khủng khiếp.
    """
    sub = df if subject is None else df[df["subject_id"] == subject]
    if sub.empty:
        return
    first = sub["subject_id"].iloc[0]
    probe = sub[sub["subject_id"] == first]

    def _speed_diagnostics(frame: pd.DataFrame) -> Tuple[float, int]:
        """Raw finite-difference |dp/dt| of a merged trial group.

        Returns (max FINITE speed, count of non-finite samples). With the wrong
        grouping ``dt`` is 0 at the splice -> division by zero -> inf/1e15.
        """
        worst, n_bad, n_pts = 0.0, 0, 0
        for _, part in frame.groupby("trial", sort=False):
            part = part.sort_values("t_ms")
            if part["t_ms"].nunique() < 3:
                continue
            t = part["t_ms"].to_numpy(float) / 1000.0
            dt = np.diff(t)
            dx = np.diff(part["x"].to_numpy(float))
            dy = np.diff(part["y"].to_numpy(float))
            with np.errstate(divide="ignore", invalid="ignore"):
                speed = np.hypot(dx, dy) / dt
            n_pts += speed.size
            n_bad += int(np.sum(~np.isfinite(speed)))
            finite = speed[np.isfinite(speed)]
            if finite.size:
                worst = max(worst, float(np.max(finite)))
        return worst, (100 * n_bad // max(1, n_pts))

    wrong, bad_pct = _speed_diagnostics(probe)                  # groupby("trial")
    right, right_pct = _speed_diagnostics(
        probe.assign(trial=probe["task"] + "#" + probe["trial"].astype(str))
    )
    print("\n  Demo on subject_id = "
          f"{first} (raw finite differences, no smoothing / đạo hàm thô):")
    print(f"    groupby('trial')          -> max finite |v| = {wrong:.3e} mm/s, "
          f"non-finite: {bad_pct}%   <-- EXPLODES / NỔ")
    print(f"    groupby(['task','trial']) -> max finite |v| = {right:.3e} mm/s, "
          f"non-finite: {right_pct}%   <-- sane / hợp lý")


def run_sanity_checks(features: pd.DataFrame, strict: bool = True) -> bool:
    """Verify the simulated/loaded data shows the expected clinical signatures.

    Kỳ vọng tự kiểm tra (self-check):
        * velocity healthy > PD          (bradykinesia / chậm vận động)
        * jerk PD > healthy              (run + gãy nhịp / movement fragmentation)
        * tremor PD ≈ 5 Hz, healthy ≈ 9 Hz
        * stability healthy > PD         (kappa ổn định hơn)
    """
    g = features.groupby("group")
    v_hc = float(g["mean_velocity"].mean().get("healthy", np.nan))
    v_pd = float(g["mean_velocity"].mean().get("pd", np.nan))
    j_hc = float(g["mean_jerk"].mean().get("healthy", np.nan))
    j_pd = float(g["mean_jerk"].mean().get("pd", np.nan))
    f_hc = float(g["tremor_peak_freq_hz"].mean().get("healthy", np.nan))
    f_pd = float(g["tremor_peak_freq_hz"].mean().get("pd", np.nan))
    s_hc = float(g["spiral_curvature_stability"].mean().get("healthy", np.nan))
    s_pd = float(g["spiral_curvature_stability"].mean().get("pd", np.nan))

    checks = [
        ("velocity: healthy > PD (bradykinesia)", v_hc > v_pd,
         f"{v_hc:.2f} vs {v_pd:.2f} mm/s"),
        ("jerk: PD > healthy (tremor/noise)", j_pd > j_hc,
         f"{j_pd:.3e} vs {j_hc:.3e} mm/s^3"),
        ("tremor peak: PD in 4-6 Hz", 4.0 <= f_pd <= 6.0, f"{f_pd:.2f} Hz"),
        ("tremor peak: healthy ~9 Hz (physiological)", 7.5 <= f_hc <= 12.0,
         f"{f_hc:.2f} Hz"),
        ("curvature stability: healthy > PD", s_hc > s_pd,
         f"{s_hc:.4f} vs {s_pd:.4f}"),
    ]
    print("\n" + "=" * 78)
    print("SELF-CHECK — EXPECTED CLINICAL SIGNATURES / KỲ VỌNG LÂM SÀNG")
    print("=" * 78)
    ok = True
    for name, passed, detail in checks:
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:<52} {detail}")
    if not ok and strict:
        raise AssertionError("Clinical sanity checks FAILED / kiểm tra lâm sàng thất bại")
    return ok


# ===========================================================================
# Figures / Hình vẽ
# ===========================================================================
def plot_spiral_samples(df: pd.DataFrame, out_path: str | Path) -> Path:
    """Figure 1 — one spiral from a healthy control vs one from a PD subject.

    Hình 1 — so sánh 1 spiral của người khoẻ và 1 spiral của bệnh nhân PD
    (thấy rõ biên độ nhỏ hơn + run).
    """
    out_path = Path(out_path)
    spirals = df[(df["task"] == "spiral")]
    pick = {}
    for group in ("healthy", "pd"):
        sub = spirals[spirals["group"] == group]
        if sub.empty:
            continue
        subject = sub["subject_id"].iloc[0]
        trial = sub[sub["subject_id"] == subject]["trial"].iloc[0]
        pick[group] = sub[(sub["subject_id"] == subject) & (sub["trial"] == trial)]

    if not pick:
        print("[WARN] no spiral trials to plot / không có trial spiral để vẽ")
        return out_path

    fig, axes = plt.subplots(1, len(pick), figsize=(11, 5.2))
    axes = np.atleast_1d(axes)
    titles = {
        "healthy": "Healthy control / Người khoẻ\n(regular spiral, smooth)",
        "pd": "Parkinson's / Bệnh nhân PD\n(micrographia + ~5 Hz tremor)",
    }
    for ax, (group, part) in zip(axes, pick.items()):
        t = part["t_ms"].to_numpy(float) / 1000.0
        sc = ax.scatter(part["x"], part["y"], c=t, s=4, cmap="viridis")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(titles.get(group, group), fontsize=10)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.grid(alpha=0.3)
        fig.colorbar(sc, ax=ax, label="time / thời gian (s)")
    fig.suptitle(
        "HandPD-style spiral trajectories / Quỹ đạo spiral kiểu HandPD", fontsize=12
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[PLOT] saved / đã lưu: {out_path}")
    return out_path


def plot_jerk_tremor_boxplots(features: pd.DataFrame, out_path: str | Path) -> Path:
    """Figure 2 — boxplots of jerk (log10) and tremor frequency by group.

    Hình 2 — boxplot jerk (log10) và tần số run theo nhóm: PD jerk cao hơn,
    run dịch về dải 4–6 Hz còn người khoẻ ~9 Hz.
    """
    out_path = Path(out_path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    try:                                    # seaborn optional, mpl fallback
        import seaborn as sns
        sns.boxplot(data=features, x="group", y="log10_mean_jerk", ax=axes[0],
                    hue="group", palette={"healthy": "#2a9d8f", "pd": "#e76f51"},
                    legend=False)
        sns.boxplot(data=features, x="group", y="tremor_peak_freq_hz", ax=axes[1],
                    hue="group", palette={"healthy": "#2a9d8f", "pd": "#e76f51"},
                    legend=False)
    except Exception:                       # pragma: no cover
        for ax, col in zip(axes, ("log10_mean_jerk", "tremor_peak_freq_hz")):
            data = [features.loc[features["group"] == g, col].dropna().to_numpy()
                    for g in ("healthy", "pd")]
            ax.boxplot(data, tick_labels=["healthy", "pd"])

    axes[0].set_title("Movement smoothness / độ mượt\n(log10 mean |jerk|, PD > healthy)",
                      fontsize=10)
    axes[0].set_xlabel("group / nhóm")
    axes[0].set_ylabel("log10 mean |jerk|  (mm/s$^3$)")
    axes[1].axhline(5.0, color="#e76f51", ls="--", lw=1,
                    label="PD tremor 4–6 Hz")
    axes[1].axhline(9.0, color="#2a9d8f", ls="--", lw=1,
                    label="physiological ~9 Hz")
    axes[1].set_title("Tremor frequency / tần số run\n(FFT peak of detrended x,y PSD)",
                      fontsize=10)
    axes[1].set_xlabel("group / nhóm")
    axes[1].set_ylabel("peak frequency / tần số đỉnh (Hz)")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("HandPD tremor & jerk by group / Run & jerk theo nhóm", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[PLOT] saved / đã lưu: {out_path}")
    return out_path


# ===========================================================================
# CLI / Giao diện dòng lệnh
# ===========================================================================
def repo_root() -> Path:
    """<repo>/src/ai_model/scripts/x.py -> <repo> (robust to cwd)."""
    return Path(__file__).resolve().parents[3]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="HandPD-style Parkinson handwriting data loader / simulator "
                    "— Bộ nạp & mô phỏng dữ liệu nét vẽ Parkinson.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-csv", default=None,
                   help="real dataset CSV (subject_id, group, task, t_ms, x, y); "
                        "if omitted the HandPD protocol is SIMULATED / nếu bỏ "
                        "trống sẽ giả lập")
    p.add_argument("--out-dir", default=str(repo_root() / "data" / "processed"),
                   help="output directory / thư mục đầu ra")
    p.add_argument("--n-subjects", type=int, default=N_SUBJECTS_DEFAULT,
                   help="number of subjects when simulating / số người giả lập")
    p.add_argument("--n-healthy", type=int, default=N_HEALTHY_DEFAULT,
                   help="healthy controls among them / số người khoẻ")
    p.add_argument("--seed", type=int, default=42, help="RNG seed / hạt giống")
    p.add_argument("--quick", action="store_true",
                   help="small fast run for smoke tests / chạy nhanh để test")
    p.add_argument("--no-plots", action="store_true",
                   help="skip figures / không vẽ hình")
    p.add_argument("--no-demo", action="store_true",
                   help="skip the groupby('trial') trap demo / bỏ phần minh hoạ "
                        "bẫy groupby('trial')")
    return p


def main(argv: Optional[list] = None) -> int:
    args = build_argparser().parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_subjects = 12 if args.quick else args.n_subjects
    n_healthy = 5 if args.quick else args.n_healthy

    # 1) fetch (real or simulated) / nạp hoặc giả lập
    df = fetch_parkinson_dataset(
        data_csv=args.data_csv, n_subjects=n_subjects, n_healthy=n_healthy,
        seed=args.seed, verbose=True,
    )

    # 2) per-trial features (TRAP 1: groupby ["subject_id","task","trial"])
    check_trial_grouping_integrity(df)
    features = extract_trial_features(df)
    print(f"\n[FEATURES] extracted / đã trích xuất: {len(features)} trials, "
          f"{features.shape[1]} columns")
    if not args.no_demo:
        demonstrate_grouping_trap(df)

    # 3) stats + class balance / thống kê + cân bằng lớp
    print_group_statistics(features)

    # 4) figures / hình
    if not args.no_plots:
        plot_spiral_samples(df, out_dir / "spiral_samples.png")
        plot_jerk_tremor_boxplots(features, out_dir / "jerk_tremor_boxplots.png")

    # 5) export CSVs / xuất CSV
    clean_path = out_dir / "parkinson_handpd_clean.csv"
    feat_path = out_dir / "parkinson_handpd_features.csv"
    df.to_csv(clean_path, index=False, float_format="%.3f")
    features.to_csv(feat_path, index=False, float_format="%.6f")
    print(f"[SAVE] clean trajectories / quỹ đạo sạch: {clean_path}")
    print(f"[SAVE] per-trial features / đặc trưng từng trial: {feat_path}")

    # 6) self-check / tự kiểm tra (real data may legitimately fail -> no strict)
    run_sanity_checks(features, strict=args.data_csv is None)
    print("\n[DONE] parkinson_data_loader finished / hoàn tất.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
