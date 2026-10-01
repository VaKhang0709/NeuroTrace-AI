#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
feature_extractor.py — On-device kinematic feature extraction for Parkinson screening
feature_extractor.py — Trích xuất đặc trưng động học cho tầm soát Parkinson

===========================================================================
 PURPOSE / MỤC ĐÍCH
===========================================================================
Turns ONE raw canvas stroke (a list of pointer samples taken while the user draws
a spiral on the web canvas) into the ~12 clinical features that the NeuroTrace
model consumes. Designed to run BOTH in the browser-adjacent Python training
pipeline and as the reference implementation for the device-side extractor.

Biến MỘT nét vẽ thô trên canvas (danh sách mẫu con trỏ) thành ~12 đặc trưng lâm
sàng mà mô hình NeuroTrace sử dụng. Đây là bản tham chiếu cho bộ trích xuất trên
thiết bị và dùng luôn trong pipeline huấn luyện.

Input format / Định dạng đầu vào::

    canvas_points = [
        {"x": 12.5, "y": -40.2, "timestamp": 1280, "radius": 2.4},   # x,y in mm|px
        {"x": 12.9, "y": -39.8, "timestamp": 1296, "radius": 2.5},   # timestamp in ms
        ...
    ]

``radius`` is the canvas brush/stylus contact radius — a proxy for pen pressure
(xấp xỉ lực đè bút của bệnh nhân, vì canvas không đo được lực thật).

===========================================================================
 MEDICAL RATIONALE PER FEATURE / Ý NGHĨA Y KHOA TỪNG CHỈ SỐ
===========================================================================
 1. micrographia_index — MICROGRAPHIA / chữ nhỏ dần
    Parkinson patients progressively shrink their handwriting (co-contraction of
    flexors/extensors). Index = normalised linear trend of R(t), the distance to
    the stroke centroid. LOW = the drawing shrinks / bệnh nhân càng vẽ càng nhỏ.
    Chỉ số = độ dốc chuẩn hoá của bán kính R(t) quanh tâm nét vẽ. THẤP = co nhỏ.

 2. radius_late_early_ratio — mean R(last 25%) / mean R(first 25%). < 1 means the
    stroke collapses inward over time. Tỷ lệ bán kính cuối/đầu < 1 = teo vào.

 3. mean_jerk / log_unquantized_jerk — movement smoothness / độ mượt vận động
    Jerk is the 3rd derivative of position (rate of change of acceleration,
    mm/s^3). Healthy reaching movements are smooth (low jerk); PD tremor,
    bradykinesia and motor fragmentation raise it. Jerk is reported raw
    (unquantized) and also on a log10 scale because its dynamic range is huge.
    Jerk = đạo hàm bậc 3 của vị trí. Người khoẻ chuyển động mượt (jerk thấp);
    run, chậm vận động và gãy nhịp ở PD làm jerk tăng. Báo cáo cả giá trị thô và
    log10 vì dải giá trị rất rộng.

 4. accel_psd_peak_freq_hz / accel_band_power_ratio — RESTING TREMOR / run
    The acceleration magnitude A(t) is dominated by tremor during a slow drawing
    task. A peak inside 4–6 Hz (the classic PD resting-tremor band) plus the
    fraction of spectral power inside that band quantify the tremor.
    Biên độ gia tốc A(t) chủ yếu phản ánh run khi vẽ chậm. Đỉnh trong dải 4–6 Hz
    (dải run nghỉ kinh điển của PD) và tỷ lệ công suất trong dải đó định lượng run.

    COMPLEMENTARY, MORE ROBUST ESTIMATOR / BỘ ƯỚC LƯỢNG BỔ SUNG CHÍNH XÁC HƠN:
    ``axis_psd_peak_freq_hz`` / ``axis_tremor_band_ratio`` come from the POSITION
    residual: detrend each axis (x and y) with a centred 0.5 s moving average and
    SUM the two PSDs. Why: when the tremor is strong, |a(t)| behaves like a
    RECTIFIED oscillation, so its spectrum moves energy into 2f and into
    intermodulation products, and the true tremor line can be masked (measured:
    the true 5 Hz line shows up as 4.0 Hz in |a| but as 5.3 Hz per axis). The
    per-axis route also cannot cancel a tremor that is perpendicular to travel.
    Khi run mạnh, |a(t)| giống tín hiệu chỉnh lưu → phổ dồn năng lượng vào 2f và
    các tích điều chế, che mất vạch run thật (đo được: 5 Hz thật hiện thành 4.0 Hz
    trên |a| nhưng 5.3 Hz khi phân tích từng trục). Phân tích từng trục cũng không
    triệt tiêu thành phần run VUÔNG GÓC với hướng vẽ.

 5. pause_count — HYPOMETRIA / gãy nét, ngập ngừng
    PD patients freeze mid-stroke (akinesia/freezing). We count contiguous
    intervals with near-zero speed longer than 100 ms.
    Bệnh nhân PD hay dừng giữa nét (mất vận động/thoái hóa). Đếm các đoạn tốc độ
    ~0 kéo dài trên 100 ms.

 6. velocity_cv = std(v)/mean(v) — ARHYTHMICITY / nhịp vận động không đều
    Healthy drawing speed is fairly constant; PD speed is erratic (dyskinesia,
    fatigue, burst-pause pattern), which inflates the coefficient of variation.
    Người khoẻ vẽ với tốc độ khá đều; PD vẽ thất thường nên CV tăng.

 7. virtual_pressure_index = E[radius / (speed + eps)] — VIRTUAL PRESSURE / lực đè
    A proxy for pen pressure: the pen is pressed harder (larger canvas radius) and
    moves slower (longer dwell) in PD subjects. Đại lượng thay thế cho lực đè bút:
    bệnh nhân PD đè mạnh hơn (radius lớn hơn) và di chuyển chậm hơn (dwell lâu).

 8. mean_velocity — BRADYKINESIA / chậm vận động
    Average pen speed. PD patients draw significantly slower.

 9. metadata — sampling/QC info so downstream code can reject bad strokes.
    Thông tin lấy mẫu & kiểm tra chất lượng để lọc nét vẽ hỏng.

===========================================================================
 TRAP HANDLED HERE / "BẪY" ĐÃ XỬ LÝ
===========================================================================
``scipy.signal.savgol_filter`` with ``deriv=3`` (jerk) and ``polyorder=2`` returns
ZEROS for every sample — a cubic derivative cannot be represented by a quadratic.
The pipeline silently produces "perfectly smooth" trajectories and every jerk
feature becomes 0. We therefore ALWAYS use ``polyorder = max(2, deriv)``
(and require ``window_length > polyorder``).

``savgol_filter`` với ``deriv=3`` nhưng ``polyorder=2`` trả về TOÀN SỐ 0 (đạo hàm
bậc 3 không thể biểu diễn bằng đa thức bậc 2) → mọi chỉ số jerk bằng 0 và mô hình
"tưởng" nét vẽ hoàn hảo. Vì vậy luôn dùng ``polyorder = max(2, deriv)``.

USAGE / CÁCH DÙNG
    from feature_extractor import ParkinsonFeatureExtractor
    fx = ParkinsonFeatureExtractor()
    feats = fx.extract_parkinson_features(canvas_points)

    python src/ai_model/feature_extractor.py     # runs the built-in dummy test
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from scipy.signal import periodogram, savgol_filter

logger = logging.getLogger(__name__)

__all__ = ["ParkinsonFeatureExtractor"]

# Feature vector order used by the NeuroTrace model — keep stable!
# Thứ tự vector đặc trưng mà mô hình dùng — KHÔNG đổi thứ tự này.
FEATURE_ORDER: Tuple[str, ...] = (
    "micrographia_index",
    "radius_late_early_ratio",
    "mean_jerk",
    "log_unquantized_jerk",
    "accel_psd_peak_freq_hz",
    "accel_band_power_ratio",
    "pause_count",
    "velocity_cv",
    "virtual_pressure_index",
    "mean_velocity",
)

EPS = 1e-9


class ParkinsonFeatureExtractor:
    """Extract clinically-motivated kinematic features from one canvas stroke.

    Trích xuất đặc trưng động học mang ý nghĩa lâm sàng từ một nét vẽ canvas.

    Parameters / Tham số
    ----------
    target_fs : float
        Resampling frequency in Hz (canvas pointer events are irregular).
        Tần số lấy mẫu lại (sự kiện con trỏ canvas không đều).
    savgol_window : int
        Odd window length (samples) of the Savitzky–Golay filter. Bigger = smoother.
        Độ dài cửa sổ Savitzky–Golay (mẫu), càng lớn càng mượt.
    savgol_poly : int or None
        Polynomial order for position/velocity/acceleration. ``None`` = auto.
        ``None`` = tự chọn. Jerk always uses ``polyorder = max(2, 3)``.
    tremor_savgol_window / tremor_savgol_poly : int
        SHORTER filter used only for the acceleration that feeds the tremor FFT.
        Measured frequency response at 100 Hz (gain of the position filter):
        window=21 attenuates 9 Hz to 0.12 and 5 Hz to 0.72 — it would erase the
        8-12 Hz physiological tremor and leak its energy down into the 4-6 Hz PD
        band. window=13/poly=4 keeps 5-6 Hz at ~1.0 and 9 Hz at ~0.96 while still
        rejecting >15 Hz, which is what a tremor band-power estimator needs.
        Bộ lọc NGẮN hơn chỉ dùng cho gia tốc phục vụ FFT run: cửa sổ 21 làm suy
        giảm 9 Hz xuống 0.12 và 5 Hz còn 0.72 → xoá mất run sinh lý 8–12 Hz và
        "rò" năng lượng xuống dải 4–6 Hz của PD. Cửa sổ 13/poly 4 giữ 5–6 Hz ~1.0
        và 9 Hz ~0.96 mà vẫn chặn >15 Hz.
    pause_savgol_window : int
        SHORT filter window used only for pause detection. A 21-sample (210 ms)
        window smears a 250 ms freeze into a ~20 ms zero crossing, so the freeze
        would never be counted; 7 samples (70 ms) keeps the timing resolution.
        Cửa sổ NGẮN chỉ dùng để phát hiện pause: cửa sổ 21 mẫu (210 ms) làm nhoè
        cú đứng hình 250 ms thành ~20 ms → không đếm được; 7 mẫu giữ độ phân giải
        thời gian.
    pause_speed_threshold : float or None
        Speed (unit/s) below which the pen is considered stopped. ``None`` = auto
        (adaptive to the stroke's own speed profile).
        Ngưỡng tốc độ coi như bút đang dừng; ``None`` = tự thích ứng.
    pause_min_duration_ms : float
        Minimum stop duration counted as a pause (default 100 ms).
        Thời gian dừng tối thiểu để tính là một "pause" (mặc định 100 ms).
    tremor_band : (float, float)
        Tremor band of interest in Hz (PD resting tremor = 4–6 Hz).
        Dải run quan tâm (run nghỉ của PD = 4–6 Hz).
    accel_band : (float, float)
        Total-power reference band for the band-power ratio.
        Dải công suất tham chiếu cho tỷ lệ công suất.
    """

    def __init__(
        self,
        target_fs: float = 100.0,
        savgol_window: int = 21,
        savgol_poly: Optional[int] = None,
        tremor_savgol_window: int = 13,
        tremor_savgol_poly: int = 4,
        pause_savgol_window: int = 7,
        pause_speed_threshold: Optional[float] = None,
        pause_min_duration_ms: float = 100.0,
        tremor_band: Tuple[float, float] = (4.0, 6.0),
        accel_band: Tuple[float, float] = (0.5, 25.0),
        max_gap_ms: float = 400.0,
    ) -> None:
        self.target_fs = float(target_fs)
        # Sample spacing in SECONDS. scipy.signal.savgol_filter defaults to
        # delta=1.0, i.e. derivatives "per SAMPLE"; without delta the velocity is
        # off by a factor 1/fs and the jerk by 1/fs^3 (a 1e6 error at 100 Hz), so
        # every clinical threshold learned downstream would be meaningless.
        # Khoảng cách mẫu (giây). ``savgol_filter`` mặc định delta=1.0 (đạo hàm
        # "theo mẫu"), nếu không truyền delta thì vận tốc lệch 1/fs lần và jerk
        # lệch 1/fs^3 lần (sai 1e6 lần ở 100 Hz) → mọi ngưỡng lâm sàng vô nghĩa.
        self.dt = 1.0 / self.target_fs
        self.savgol_window = int(savgol_window)
        self.savgol_poly = savgol_poly
        self.tremor_savgol_window = int(tremor_savgol_window)
        self.tremor_savgol_poly = int(tremor_savgol_poly)
        self.pause_savgol_window = int(pause_savgol_window)
        self.pause_speed_threshold = pause_speed_threshold
        self.pause_min_duration_ms = float(pause_min_duration_ms)
        self.tremor_band = (float(tremor_band[0]), float(tremor_band[1]))
        self.accel_band = (float(accel_band[0]), float(accel_band[1]))
        # Long sampling gap (finger lifted / tab switched) — not a real pause.
        # Khoảng trống lấy mẫu dài (nhấc tay / đổi tab) — không tính là pause.
        self.max_gap_ms = float(max_gap_ms)

    # =====================================================================
    # 1) Pre-processing / Tiền xử lý
    # =====================================================================
    @staticmethod
    def _points_to_arrays(
        canvas_points: Sequence[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Validate the raw canvas payload -> (t_ms, x, y, radius).

        Kiểm tra dữ liệu thô và chuyển thành mảng (t_ms, x, y, radius).
        """
        if canvas_points is None or len(canvas_points) == 0:
            raise ValueError("canvas_points is empty / danh sách điểm rỗng")

        expected = ("x", "y", "timestamp", "radius")
        for key in ("x", "y", "timestamp"):
            if key not in canvas_points[0]:
                raise ValueError(
                    f"canvas_points[{0}] is missing '{key}' — expected keys "
                    f"{expected} / thiếu khoá '{key}'"
                )

        t_ms = np.array([float(p["timestamp"]) for p in canvas_points], dtype=float)
        x = np.array([float(p["x"]) for p in canvas_points], dtype=float)
        y = np.array([float(p["y"]) for p in canvas_points], dtype=float)
        radius = np.array(
            [float(p.get("radius", 0.0)) for p in canvas_points], dtype=float
        )

        finite = np.isfinite(t_ms) & np.isfinite(x) & np.isfinite(y)
        if not np.all(finite):
            n_bad = int(np.sum(~finite))
            logger.warning("dropped %d non-finite samples / bỏ %d mẫu lỗi", n_bad, n_bad)
        t_ms, x, y, radius = t_ms[finite], x[finite], y[finite], radius[finite]
        radius = np.nan_to_num(radius, nan=0.0)
        if t_ms.size < 3:
            raise ValueError(
                "need at least 3 valid samples / cần ít nhất 3 mẫu hợp lệ"
            )

        # Sort by time and merge duplicate timestamps (Chrome coalesces pointer
        # events; two samples can share the same millisecond).
        # Sắp xếp theo thời gian và gộp timestamp trùng (trình duyệt gộp sự kiện).
        order = np.argsort(t_ms, kind="stable")
        t_ms, x, y, radius = t_ms[order], x[order], y[order], radius[order]
        keep = np.concatenate([[True], np.diff(t_ms) > 0])
        if not np.all(keep):
            t_ms, x, y, radius = t_ms[keep], x[keep], y[keep], radius[keep]
        if t_ms.size < 3:
            raise ValueError(
                "need 3 samples with distinct timestamps / cần 3 mốc thời gian khác nhau"
            )
        return t_ms, x, y, radius

    def resample_to_target_fs(
        self, canvas_points: Sequence[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Resample an irregular stroke onto a uniform ``target_fs`` grid (100 Hz).

        Why resample? Pointer events arrive at uneven intervals (60–240 Hz,
        jittery). Every derivative/frequency estimator below assumes a constant
        ``dt``, so the first thing we do is interpolate onto a fixed grid.

        Vì sao phải resample? Sự kiện con trỏ đến không đều (60–240 Hz, rung).
        Mọi phép đạo hàm/FFT bên dưới đều giả định ``dt`` không đổi, nên bước đầu
        tiên là nội suy về lưới thời gian cố định.

        Returns / Trả về: (t, x, y, radius, resample_meta)
        """
        t_ms, x, y, radius = self._points_to_arrays(canvas_points)
        dt_ms = 1000.0 / self.target_fs

        # A long sampling gap means the finger was lifted or the page was hidden;
        # forward-filling it would create a fake "pause", so we only interpolate
        # and let ``resample_meta`` report the gap.
        # Khoảng trống dài = nhấc tay/đổi tab; ta chỉ nội suy và báo cáo lại.
        gaps_ms = np.diff(t_ms)
        n_long_gaps = int(np.sum(gaps_ms > self.max_gap_ms))

        t_out = np.arange(t_ms[0], t_ms[-1] + dt_ms * 0.5, dt_ms)
        if t_out.size < 3:                       # extremely short stroke
            t_out = np.linspace(t_ms[0], t_ms[-1], 3)
        x_out = np.interp(t_out, t_ms, x)
        y_out = np.interp(t_out, t_ms, y)
        r_out = np.interp(t_out, t_ms, radius)

        meta = {
            "n_raw_points": int(t_ms.size),
            "n_resampled_points": int(t_out.size),
            "raw_duration_ms": float(t_ms[-1] - t_ms[0]),
            "mean_raw_interval_ms": float(np.mean(gaps_ms)) if gaps_ms.size else np.nan,
            "long_gaps": n_long_gaps,
            "target_fs_hz": self.target_fs,
        }
        return t_out, x_out, y_out, r_out, meta

    # =====================================================================
    # 2) Savitzky–Golay derivatives / Đạo hàm Savitzky–Golay
    # =====================================================================
    def _savgol(
        self, signal: np.ndarray, deriv: int = 0, polyorder: Optional[int] = None,
        window: Optional[int] = None,
    ) -> np.ndarray:
        """Savitzky–Golay smoothing/derivative, in PHYSICAL units per second.

        Two deliberate safeguards / hai bảo vệ có chủ đích:

        (a) POLYORDER GUARD — ``deriv=3, polyorder=2`` returns ALL ZEROS: a cubic
            derivative cannot be represented by a quadratic fit, so every jerk
            feature would silently become 0 and the model would "see" perfectly
            smooth handwriting for every patient. We enforce
            ``polyorder = max(polyorder, deriv)`` (and >= 2) plus
            ``window_length > polyorder``.
            BẪY: ``deriv=3`` với ``polyorder=2`` trả về TOÀN SỐ 0 → mọi đặc trưng
            jerk bằng 0. Ta luôn ép ``polyorder = max(polyorder, deriv)``.

        (b) ``delta = 1/target_fs`` — scipy differentiates with respect to the
            SAMPLE INDEX unless ``delta`` is given, so the resulting velocity
            would be in "units per sample" (100x too small) and jerk 1e6x too
            small. Passing ``delta`` makes every derivative a per-SECOND quantity.
            ``delta`` giúp đạo hàm theo GIÂY thay vì theo chỉ số mẫu.
        """
        signal = np.asarray(signal, dtype=float)
        n = signal.size
        poly = self.savgol_poly if polyorder is None else int(polyorder)
        poly = max(2 if poly is None else poly, int(deriv))     # <<< the guard
        window = self.savgol_window if window is None else int(window)
        if window % 2 == 0:
            window += 1
        window = max(window, poly + 2 + (poly + 2) % 2)          # window > poly
        if window > n:
            window = n if n % 2 == 1 else n - 1
        if window <= poly:                                        # tiny strokes
            window = max(poly + 1, 3)
            if window % 2 == 0:
                window += 1
            if window > n:                                        # SG impossible
                out = signal.copy()
                for _ in range(int(deriv)):
                    out = np.gradient(out, self.dt, edge_order=1)
                return out
        return savgol_filter(signal, window_length=window, polyorder=poly,
                             deriv=deriv, delta=self.dt, mode="interp")

    # =====================================================================
    # 3) Feature blocks / Các khối đặc trưng
    # =====================================================================
    @staticmethod
    def _radial_features(x: np.ndarray, y: np.ndarray, t_s: np.ndarray) -> Dict[str, float]:
        """Micrographia features from R(t) = distance to the stroke centroid.

        Đặc trưng micrographia tính từ R(t) = khoảng cách tới tâm nét vẽ.
        """
        cx, cy = float(np.mean(x)), float(np.mean(y))
        radius_t = np.hypot(x - cx, y - cy)
        mean_r = float(np.mean(radius_t))
        if mean_r <= EPS:                     # degenerate (single dot)
            return {"micrographia_index": np.nan,
                    "radius_late_early_ratio": np.nan,
                    "mean_radius": 0.0}

        # Normalised linear trend of R(t): slope * duration / mean(R).
        # ~+2 for a normal outward spiral (R grows 0 -> Rmax),
        # ~0 or negative when the writing shrinks (micrographia).
        # Độ dốc chuẩn hoá: ~+2 với spiral mở rộng bình thường; ~0 hoặc âm khi co nhỏ.
        slope = float(np.polyfit(t_s - t_s[0], radius_t, 1)[0])
        duration = float(t_s[-1] - t_s[0])
        micrographia_index = slope * duration / mean_r

        q = max(1, radius_t.size // 4)
        early = float(np.mean(radius_t[:q]))
        late = float(np.mean(radius_t[-q:]))
        return {
            "micrographia_index": float(micrographia_index),
            "radius_late_early_ratio": float(late / (early + EPS)),
            "mean_radius": mean_r,
        }

    @staticmethod
    def _moving_average(signal: np.ndarray, window: int) -> np.ndarray:
        """Edge-safe centred moving average (same length as the input).

        Trung bình trượt căn giữa, đệm biên kiểu "edge" nên không sinh NaN.
        """
        sig = np.asarray(signal, dtype=float)
        if sig.size == 0:
            return sig.copy()
        window = int(max(1, window))
        if window % 2 == 0:
            window += 1
        if window > sig.size:
            window = sig.size if sig.size % 2 == 1 else max(1, sig.size - 1)
        if window <= 1:
            return sig.copy()
        kernel = np.ones(window) / float(window)
        return np.convolve(np.pad(sig, window // 2, mode="edge"), kernel, mode="valid")

    def _axis_tremor_features(
        self, x: np.ndarray, y: np.ndarray, fs: float,
        search_band: Tuple[float, float] = (3.0, 12.0), detrend_s: float = 0.5,
    ) -> Dict[str, float]:
        """Tremor estimated on the POSITION residual, axis by axis, then summed.

        Complementary estimator to ``accel_psd_peak_freq_hz``. The magnitude A(t)
        of a strongly trembling stroke is a *rectified* oscillation, so its
        spectrum splits the tremor line into harmonics and intermodulation
        products and the true frequency can be masked. Detrending EACH axis with a
        centred 0.5 s moving average and adding the two PSDs keeps the tremor line
        sharp (and, importantly, does not cancel a tremor that is perpendicular to
        the direction of travel).

        Bộ ước lượng BỔ SUNG cho ``accel_psd_peak_freq_hz``. Với nét vẽ run mạnh,
        A(t) là tín hiệu "chỉnh lưu" nên phổ tách thành hài bậc 2 và các tích điều
        chế, có thể che mất tần số run thật. Detrend TỪNG TRỤC bằng trung bình
        trượt 0.5 s rồi CỘNG hai phổ PSD sẽ giữ vạch run sắc nét (và không triệt
        tiêu thành phần run VUÔNG GÓC với hướng vẽ).
        """
        out = {"axis_psd_peak_freq_hz": np.nan,
               "axis_psd_peak_power": np.nan,
               "axis_tremor_band_ratio": np.nan}
        if x.size < 16:
            return out
        win = max(3, int(round(detrend_s * fs)) | 1)
        res_x = x - self._moving_average(x, win)
        res_y = y - self._moving_average(y, win)
        freq, psd_x = periodogram(res_x, fs=fs, window="hann", detrend="linear",
                                  scaling="density")
        _, psd_y = periodogram(res_y, fs=fs, window="hann", detrend="linear",
                               scaling="density")
        psd_sum = psd_x + psd_y
        df = float(freq[1] - freq[0]) if freq.size > 1 else 1.0

        search = (freq >= search_band[0]) & (freq <= search_band[1])
        if not np.any(search):
            return out
        f_s, p_s = freq[search], psd_sum[search]
        k = int(np.argmax(p_s))
        out["axis_psd_peak_freq_hz"] = float(f_s[k])
        out["axis_psd_peak_power"] = float(p_s[k])
        total = float(np.sum(p_s) * df)
        band = (f_s >= self.tremor_band[0]) & (f_s <= self.tremor_band[1])
        out["axis_tremor_band_ratio"] = (
            float(np.sum(p_s[band]) * df / total) if total > 0 else np.nan
        )
        return out

    @staticmethod
    def _pause_features(
        speed: np.ndarray, fs: float, threshold: Optional[float], min_ms: float
    ) -> Dict[str, float]:
        """Count moves with near-zero speed lasting longer than ``min_ms``.

        Đếm các đoạn tốc độ gần bằng 0 kéo dài quá ``min_ms`` (gãy nét / freezing).
        """
        if speed.size == 0:
            return {"pause_count": 0.0, "pause_total_ms": 0.0,
                    "pause_speed_threshold": np.nan}
        if threshold is None:
            # Scale-free adaptive threshold: 2% of the stroke's MEDIAN speed.
            # Why so low? A spiral is drawn 5-10x faster at the outside than near
            # the centre, so a larger fraction would flag the naturally slow
            # centre of every spiral as a "pause". A real freeze keeps the pen
            # still (speed ~0.1% of the median), so 2% separates them cleanly
            # while staying independent of the canvas units (mm or pixels).
            # Vì sao chỉ 2%? Spiral được vẽ nhanh gấp 5–10 lần ở vòng ngoài so với
            # tâm, nên ngưỡng cao hơn sẽ báo nhầm phần tâm (vốn chậm sẵn) thành
            # pause. Một cú đứng hình thật có tốc độ ~0.1% trung vị, nên 2% tách
            # sạch hai trường hợp mà không phụ thuộc đơn vị (mm hay pixel).
            thr = max(0.02 * float(np.median(speed)), 1e-6)
        else:
            thr = float(threshold)

        stopped = speed < thr
        min_samples = max(1, int(round(min_ms / 1000.0 * fs)))
        count, total, run = 0, 0, 0
        for flag in stopped:
            if flag:
                run += 1
            else:
                if run >= min_samples:
                    count += 1
                    total += run
                run = 0
        if run >= min_samples:                # pause running until the last sample
            count += 1
            total += run
        return {
            "pause_count": float(count),
            "pause_total_ms": float(total / fs * 1000.0),
            "pause_speed_threshold": float(thr),
        }

    @staticmethod
    def _spectral_features(
        accel: np.ndarray, fs: float, tremor_band: Tuple[float, float],
        total_band: Tuple[float, float],
    ) -> Dict[str, float]:
        """FFT peak of A(t) inside the PD tremor band + band-power ratio.

        The acceleration magnitude A(t) is dominated by tremor during slow
        drawing, so its spectrum is the natural place to look for the 4–6 Hz PD
        signature. The ratio tells us how much of the movement energy sits inside
        that narrow band (a tremor-dominant stroke has a high ratio).

        A(t) (độ lớn gia tốc) bị chi phối bởi run khi vẽ chậm, nên phổ của nó là
        nơi tự nhiên nhất để tìm dấu hiệu 4–6 Hz của PD. Tỷ lệ công suất cho biết
        bao nhiêu phần năng lượng vận động nằm trong dải hẹp đó.
        """
        out = {
            "accel_psd_peak_freq_hz": np.nan,
            "accel_psd_peak_power": np.nan,
            "accel_band_power_ratio": np.nan,
            "accel_rms": float(np.sqrt(np.mean(accel ** 2))) if accel.size else np.nan,
        }
        if accel.size < 16:
            return out
        signal = accel - float(np.mean(accel))           # remove DC / bỏ thành phần DC
        freq, psd = periodogram(signal, fs=fs, window="hann", detrend="linear",
                                scaling="density")
        df = float(freq[1] - freq[0]) if freq.size > 1 else 1.0
        band = (freq >= tremor_band[0]) & (freq <= tremor_band[1])
        total_mask = (freq >= total_band[0]) & (freq <= total_band[1])
        if not np.any(band):
            return out
        p_band, f_band = psd[band], freq[band]

        # PEAK LOCALISATION ON A WHITENED SPECTRUM / tìm đỉnh trên phổ đã "làm trắng"
        # A(t) of a moving pen has a steeply falling spectrum (the movement itself
        # lives at low frequency), so a naive argmax inside 4-6 Hz would always
        # return the LOWEST bin of the window (a slope, not a tremor peak). We
        # therefore fit and subtract a smooth log-spectral background over 2-10 Hz
        # and look for a genuine bump inside the tremor band.
        # Phổ của A(t) dốc mạnh về phía tần số thấp (bản thân chuyển động nằm ở tần
        # số thấp), nên argmax "thô" trong 4–6 Hz luôn trả về bin THẤP NHẤT của
        # dải — đó là độ dốc phổ chứ không phải đỉnh run. Vì vậy ta khớp và trừ đi
        # nền phổ log10 trong dải 2–10 Hz rồi mới tìm đỉnh thật trong dải run.
        search = (freq >= 2.0) & (freq <= 10.0)
        peak_freq, peak_power = float(f_band[int(np.argmax(p_band))]), float(
            np.max(p_band)
        )
        if np.count_nonzero(search) >= 8:
            log_psd = np.log10(psd[search] + 1e-30)
            try:
                baseline = np.polyval(np.polyfit(freq[search], log_psd, 3), f_band)
            except np.linalg.LinAlgError:               # pragma: no cover
                baseline = np.zeros_like(p_band)
            whitened = np.log10(p_band + 1e-30) - baseline
            k = int(np.argmax(whitened))
            peak_freq = float(f_band[k])
            peak_power = float(p_band[k])
        out["accel_psd_peak_freq_hz"] = peak_freq
        out["accel_psd_peak_power"] = peak_power
        band_power = float(np.sum(p_band) * df)
        total_power = float(np.sum(psd[total_mask]) * df)
        out["accel_band_power_ratio"] = (
            float(band_power / (total_power + EPS)) if total_power > 0 else np.nan
        )
        return out

    # =====================================================================
    # 4) Public API / API công khai
    # =====================================================================
    def extract_parkinson_features(
        self, canvas_points: Sequence[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract the full clinical feature dict from one canvas stroke.

        Trích xuất toàn bộ đặc trưng lâm sàng từ một nét vẽ canvas.

        Returns / Trả về: dict with the 10 keys in ``FEATURE_ORDER`` plus
        ``metadata`` and a few diagnostic extras (``stroke_id`` is passed through
        when present). Từ điển gồm 10 đặc trưng chính + ``metadata`` + vài chỉ số
        chẩn đoán phụ.
        """
        t_ms, x, y, radius, resample_meta = self.resample_to_target_fs(canvas_points)
        fs = self.target_fs
        t_s = (t_ms - t_ms[0]) / 1000.0

        # --- derivatives / đạo hàm -----------------------------------------
        # Position is smoothed once (deriv=0) so that curvature/radius features
        # are not dominated by sensor noise; velocity and acceleration reuse the
        # same Savitzky–Golay family so all quantities are mutually consistent.
        # Vị trí được làm mượt một lần để đặc trưng độ cong không bị nhiễu chi phối.
        x_s = self._savgol(x, deriv=0)
        y_s = self._savgol(y, deriv=0)
        vx = self._savgol(x, deriv=1)
        vy = self._savgol(y, deriv=1)
        ax = self._savgol(x, deriv=2)
        ay = self._savgol(y, deriv=2)
        # JERK — deriv=3 => the polyorder guard inside _savgol() matters here.
        # JERK — deriv=3 => cần đúng "bẫy" polyorder bên trong _savgol().
        jx = self._savgol(x, deriv=3)
        jy = self._savgol(y, deriv=3)

        speed = np.hypot(vx, vy)                        # unit/s
        accel = np.hypot(ax, ay)                        # unit/s^2
        jerk = np.hypot(jx, jy)                         # unit/s^3

        # Dedicated SHORT window for the tremor FFT: the 21-sample general filter
        # attenuates 9 Hz to ~0.12 while keeping 5 Hz at 0.72, which would erase
        # the physiological tremor of controls and leak its energy into the PD
        # band. The band-power estimator must see both bands with flat gain.
        # Cửa sổ NGẮN riêng cho FFT run: bộ lọc 21 mẫu làm suy giảm 9 Hz còn 0.12
        # nhưng giữ 5 Hz ở 0.72 → xoá run sinh lý của người khoẻ và "rò" năng
        # lượng xuống dải PD. Bộ ước lượng công suất dải cần độ lợi phẳng.
        ax_t = self._savgol(x, deriv=2, polyorder=self.tremor_savgol_poly,
                            window=self.tremor_savgol_window)
        ay_t = self._savgol(y, deriv=2, polyorder=self.tremor_savgol_poly,
                            window=self.tremor_savgol_window)
        accel_tremor = np.hypot(ax_t, ay_t)

        # Dedicated SHORT window for pause detection: pausing is a *timing*
        # feature, so smoothing must not smear a 250 ms freeze.
        # Cửa sổ NGẮN riêng cho phát hiện pause: đây là đặc trưng THỜI GIAN nên
        # không được làm nhoè cú đứng hình 250 ms.
        vx_p = self._savgol(x, deriv=1, polyorder=2, window=self.pause_savgol_window)
        vy_p = self._savgol(y, deriv=1, polyorder=2, window=self.pause_savgol_window)
        speed_pause = np.hypot(vx_p, vy_p)

        mean_velocity = float(np.mean(speed))
        velocity_cv = float(np.std(speed) / (mean_velocity + EPS))
        mean_jerk = float(np.mean(jerk))

        feats: Dict[str, Any] = {}
        feats.update(self._radial_features(x_s, y_s, t_s))
        feats.update(self._pause_features(speed_pause, fs, self.pause_speed_threshold,
                                          self.pause_min_duration_ms))
        feats.update(self._spectral_features(accel_tremor, fs, self.tremor_band,
                                             self.accel_band))
        feats.update(self._axis_tremor_features(x, y, fs))

        feats["mean_jerk"] = mean_jerk
        # Raw (unquantized) jerk on a log scale — the dynamic range of jerk is
        # ~1e3..1e7 mm/s^3, so log10 keeps the network input well-conditioned.
        # Jerk thô (chưa lượng tử hoá) trên thang log10 — dải giá trị quá rộng nên
        # log10 giúp đầu vào mạng ổn định số học.
        feats["log_unquantized_jerk"] = float(np.log10(abs(mean_jerk) + 1e-12))
        feats["mean_velocity"] = mean_velocity
        feats["velocity_cv"] = velocity_cv

        # virtual_pressure_index = E[radius / (speed + eps)]: large contact radius
        # + slow motion => high "pressure". / bán kính tiếp xúc lớn + di chuyển
        # chậm => "lực đè" cao.
        feats["virtual_pressure_index"] = float(
            np.mean(radius / (speed + 1e-6))
        )

        # Diagnostic extras (not part of the model input, but very useful in EDA)
        # Chỉ số chẩn đoán thêm (không đưa vào mô hình nhưng hữu ích khi phân tích).
        feats["mean_accel"] = float(np.mean(accel))
        feats["path_length"] = float(np.sum(np.hypot(np.diff(x_s), np.diff(y_s))))
        feats["stroke_duration_s"] = float(t_s[-1])
        feats["mean_radius"] = feats.get("mean_radius", float(np.mean(radius)))

        metadata = {
            **resample_meta,
            "savgol_window": int(self.savgol_window),
            "savgol_poly_used": int(
                max(2 if self.savgol_poly is None else self.savgol_poly, 3)
            ),                                   # jerk polyorder (the trap guard)
            "pause_min_duration_ms": self.pause_min_duration_ms,
            "tremor_band_hz": self.tremor_band,
            "tremor_savgol_window": int(self.tremor_savgol_window),
            "pause_savgol_window": int(self.pause_savgol_window),
            "target_fs_hz": self.target_fs,
            "delta_seconds": self.dt,
            "n_samples_analyzed": int(x.size),
            "extractor": "ParkinsonFeatureExtractor/v1",
        }
        feats["metadata"] = metadata
        feats["feature_vector"] = self.to_feature_vector(feats)
        return feats

    @staticmethod
    def to_feature_vector(features: Dict[str, Any]) -> np.ndarray:
        """Flatten the feature dict into a fixed-order float32 vector (no NaN)."""
        vec = np.array([features.get(k, np.nan) for k in FEATURE_ORDER], dtype=np.float32)
        return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

    @classmethod
    def feature_names(cls) -> Tuple[str, ...]:
        """Model input order / thứ tự đầu vào mô hình."""
        return FEATURE_ORDER


# ===========================================================================
# Built-in dummy test / Test giả lập kèm theo file
# ===========================================================================
def _make_spiral_points(
    duration_s: float,
    n_turns: float = 4.0,
    r_start: float = 3.0,
    r_end: float = 30.0,
    fs_pts: float = 60.0,
    tremor_hz: Optional[float] = None,
    tremor_amp: float = 0.0,
    shrink: float = 0.0,
    noise: float = 0.1,
    pause_specs: Sequence[Tuple[float, float]] = (),
    timestamp_jitter_ms: float = 0.6,
    speed_irregularity: float = 0.0,
    irregularity_band_hz: Tuple[float, float] = (0.35, 0.95),
    radius_value: float = 2.5,
    radius_noise: float = 0.05,
    seed: int = 0,
) -> List[Dict[str, float]]:
    """Synthesise one canvas stroke (spiral) as a list of canvas_points dicts.

    Sinh một nét vẽ spiral dưới dạng danh sách ``canvas_points``.

    Parameters
    ----------
    tremor_hz / tremor_amp : sinusoidal tremor injected PERPENDICULAR to motion
        (mm). Run được cấy theo phương VUÔNG GÓC với hướng di chuyển.
    shrink : micrographia — the radius shrinks progressively (0..1).
        Micrographia — bán kính co dần theo tiến trình vẽ.
    pause_specs : list of (start_s, duration_s) where the pen HOLDS its position
        while the timestamp keeps running (that is what a real freeze looks like).
        Danh sách (thời điểm bắt đầu, độ dài) mà bút GIỮ NGUYÊN vị trí trong khi
        thời gian vẫn chạy — đúng như một cú "đứng hình" thật.
    speed_irregularity : ARHYTHMICITY / nhịp vận động thất thường
        Amplitude of the burst-stall speed modulation of PD. The path shape is
        unchanged (progress is re-parameterised in time), only the speed profile
        becomes erratic, which is what raises ``velocity_cv``.
        Biên độ dao động tốc độ kiểu "bùng nổ - khựng lại" của PD. Hình dạng nét
        vẽ không đổi, chỉ có tiến trình theo thời gian thay đổi → ``velocity_cv``
        tăng.
    """
    rng = np.random.default_rng(seed)
    # 60 Hz pointer events, irregular like a real browser (~16.7 ms ± jitter)
    # Sự kiện con trỏ 60 Hz, có jitter như trình duyệt thật.
    dt_ms = 1000.0 / fs_pts
    base_t = np.arange(0.0, duration_s * 1000.0, dt_ms)
    if timestamp_jitter_ms > 0:
        # Browser pointer events are irregular — resampling must regularise them.
        # Keep the jitter small: large jitter + linear interpolation acts as a
        # random phase modulation that smears the tremor spectrum.
        # Sự kiện con trỏ có jitter — việc resample phải chuẩn hoá lại. Giữ jitter
        # nhỏ vì jitter lớn + nội suy tuyến tính sẽ làm nhoè phổ run.
        base_t = base_t + rng.uniform(-timestamp_jitter_ms, timestamp_jitter_ms,
                                      base_t.size)
    base_t = np.clip(base_t, 0.0, None)
    base_t = np.unique(np.round(base_t, 3))                   # strictly increasing

    u = base_t / base_t[-1]                                   # normalised progress
    if speed_irregularity > 0:
        # BURST-STALL SPEED PROFILE / tốc độ kiểu "bùng nổ - khựng lại".
        # Re-parameterise the progress in time: the geometric path stays the same
        # but the speed becomes erratic, the hallmark of parkinsonian dysrhythmia.
        # Tái tham số hoá tiến trình theo thời gian: hình dạng nét vẽ giữ nguyên
        # nhưng tốc độ thất thường — dấu hiệu loạn nhịp vận động của PD.
        t_s = base_t / 1000.0
        lo, hi = irregularity_band_hz
        weight = np.ones_like(u)
        for _ in range(3):                      # 3 incommensurate components
            f = rng.uniform(lo, hi)
            weight = weight + speed_irregularity * np.sin(
                2.0 * np.pi * f * t_s + rng.uniform(0.0, 2.0 * np.pi)
            )
        weight = np.clip(weight, 0.05, None)
        u = np.cumsum(weight)
        u = (u - u[0]) / (u[-1] - u[0])
    # constant-speed spiral / spiral tốc độ đều
    theta = 2.0 * np.pi * n_turns * u
    r = (r_start + (r_end - r_start) * u) * (1.0 - shrink * u)
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # tremor PERPENDICULAR to the direction of travel / run vuông góc hướng vẽ
    if tremor_hz is not None and tremor_amp > 0:
        dx, dy = np.gradient(x), np.gradient(y)
        norm = np.hypot(dx, dy)
        norm[norm < 1e-9] = 1e-9
        nx, ny = -dy / norm, dx / norm
        osc = tremor_amp * np.sin(2.0 * np.pi * tremor_hz * base_t / 1000.0)
        x = x + nx * osc
        y = y + ny * osc

    x = x + rng.normal(0.0, noise, x.size)
    y = y + rng.normal(0.0, noise, y.size)

    # FREEZE LAST: hold the pen perfectly still while the timestamp keeps running.
    # Must come AFTER tremor/noise, otherwise the injected oscillation keeps
    # moving the pen during the "freeze" and the pause is never detected.
    # ĐỨNG HÌNH SAU CÙNG: giữ nguyên vị trí trong khi thời gian vẫn chạy. Phải làm
    # SAU bước cấy run/nhiễu, nếu không bút vẫn "rung" trong lúc pause → không phát
    # hiện được pause.
    for start_s, dur_s in pause_specs:
        held = (base_t >= start_s * 1000.0) & (base_t < (start_s + dur_s) * 1000.0)
        idx = np.flatnonzero(held)
        if idx.size == 0:
            continue
        first = idx[0]
        x[idx] = x[first]
        y[idx] = y[first]

    radius = np.clip(radius_value + rng.normal(0.0, radius_noise, x.size), 0.2, None)

    return [
        {"x": float(xi), "y": float(yi), "timestamp": float(ti), "radius": float(ri)}
        for xi, yi, ti, ri in zip(x, y, base_t, radius)
    ]


def _run_dummy_test(verbose: bool = True) -> bool:
    """Built-in clinical sanity test / Test lâm sàng tích hợp.

    Builds TWO synthetic spiral strokes — a healthy control and a Parkinson
    patient (5 Hz tremor ~1.4 mm, micrographia, 3 pauses of 250 ms) — and asserts
    9 clinical inequalities between their feature vectors.

    Tạo HAI nét spiral tổng hợp — một người khoẻ và một bệnh nhân PD (run 5 Hz
    biên độ ~1.4 mm, chữ nhỏ dần, 3 lần dừng 250 ms) — rồi kiểm tra 9 bất đẳng
    thức lâm sàng giữa hai vector đặc trưng.
    """
    extractor = ParkinsonFeatureExtractor()

    healthy_points = _make_spiral_points(
        duration_s=7.0, tremor_hz=9.0, tremor_amp=0.12, shrink=0.0,
        noise=0.08, radius_value=2.6, seed=1,
    )
    # PD: bradykinesia (10.5 s), 5 Hz tremor, micrographia, erratic speed and
    # 3 freezes of 250 ms / PD: chậm vận động, run 5 Hz, chữ nhỏ dần, tốc độ thất
    # thường và 3 lần đứng hình 250 ms
    pd_points = _make_spiral_points(
        duration_s=10.5, tremor_hz=5.0, tremor_amp=1.4, shrink=0.28,
        noise=0.35, pause_specs=[(2.0, 0.25), (4.5, 0.25), (7.0, 0.25)],
        speed_irregularity=0.55, radius_value=2.2, seed=2,
    )

    health = extractor.extract_parkinson_features(healthy_points)
    park = extractor.extract_parkinson_features(pd_points)

    if verbose:
        print("=" * 78)
        print("DUMMY TEST — healthy control vs Parkinson stroke / người khoẻ vs PD")
        print("=" * 78)
        header = f"{'feature / đặc trưng':<30}{'healthy':>16}{'PD':>16}   expected"
        print(header)
        print("-" * len(header))
        expectations = {
            "mean_velocity": "healthy > PD  (bradykinesia)",
            "mean_jerk": "PD > healthy  (tremor)",
            "log_unquantized_jerk": "PD > healthy",
            "accel_psd_peak_freq_hz": "PD in 4-6 Hz",
            "accel_band_power_ratio": "PD > healthy  (tremor power)",
            "pause_count": "PD >= 3, healthy == 0",
            "velocity_cv": "PD > healthy  (arrhythmia)",
            "virtual_pressure_index": "reported / tham khảo",
            "micrographia_index": "healthy > PD  (shrink)",
            "radius_late_early_ratio": "healthy > PD  (shrink)",
            "mean_accel": "diagnostic / chẩn đoán",
            "stroke_duration_s": "PD > healthy  (bradykinesia)",
        }
        for key in FEATURE_ORDER + ("mean_accel", "stroke_duration_s"):
            h, p = health.get(key, np.nan), park.get(key, np.nan)
            print(f"{key:<30}{h:>16.4f}{p:>16.4f}   {expectations.get(key, '')}")
        print(f"\nresampled points: healthy={health['metadata']['n_resampled_points']}, "
              f"PD={park['metadata']['n_resampled_points']}  "
              f"@ {extractor.target_fs:.0f} Hz")
        print(f"raw points      : healthy={health['metadata']['n_raw_points']}, "
              f"PD={park['metadata']['n_raw_points']}\n")

    # --- 9 clinical inequalities / 9 bất đẳng thức lâm sàng -----------------
    jam_ok = park["mean_jerk"] > health["mean_jerk"]
    checks: List[Tuple[str, bool, str]] = [
        ("1.  bradykinesia: healthy velocity > PD velocity",
         health["mean_velocity"] > park["mean_velocity"],
         f"{health['mean_velocity']:.2f} vs {park['mean_velocity']:.2f} unit/s"),

        ("2.  movement smoothness: PD jerk > healthy jerk",
         jam_ok,
         f"{park['mean_jerk']:.3e} vs {health['mean_jerk']:.3e} unit/s^3"),

        ("3.  log10(raw jerk): PD > healthy",
         park["log_unquantized_jerk"] > health["log_unquantized_jerk"],
         f"{park['log_unquantized_jerk']:.2f} vs "
         f"{health['log_unquantized_jerk']:.2f}"),

        ("4.  tremor detected in the 4-6 Hz PD band",
         4.0 <= park["accel_psd_peak_freq_hz"] <= 6.0
         and 4.0 <= park["axis_psd_peak_freq_hz"] <= 6.0,
         f"A(t) peak {park['accel_psd_peak_freq_hz']:.2f} Hz, "
         f"axis peak {park['axis_psd_peak_freq_hz']:.2f} Hz"),

        ("5.  tremor band-power ratio: PD > healthy",
         park["accel_band_power_ratio"] > health["accel_band_power_ratio"],
         f"{park['accel_band_power_ratio']:.3f} vs "
         f"{health['accel_band_power_ratio']:.3f}"),

        ("6.  freezes: PD has 3 pauses of 250 ms, healthy none",
         park["pause_count"] >= 3 and health["pause_count"] == 0,
         f"PD={int(park['pause_count'])}, healthy={int(health['pause_count'])}"),

        ("7.  arrhythmic speed: PD velocity_cv > healthy",
         park["velocity_cv"] > health["velocity_cv"],
         f"{park['velocity_cv']:.3f} vs {health['velocity_cv']:.3f}"),

        ("8.  micrographia: healthy R-trend > PD R-trend",
         health["micrographia_index"] > park["micrographia_index"],
         f"{health['micrographia_index']:.3f} vs "
         f"{park['micrographia_index']:.3f}"),

        ("9.  inward collapse: healthy late/early radius ratio > PD",
         health["radius_late_early_ratio"] > park["radius_late_early_ratio"],
         f"{health['radius_late_early_ratio']:.3f} vs "
         f"{park['radius_late_early_ratio']:.3f}"),
    ]

    # --- TRAP DEMO: polyorder=2 with deriv=3 annihilates the jerk ----------
    # Minh hoạ bẫy: gọi THẲNG scipy với polyorder=2 -> jerk = 0 toàn bộ.
    x_raw = np.asarray([p["x"] for p in healthy_points], dtype=float)
    x_grid = np.interp(
        np.arange(x_raw.size), np.arange(x_raw.size), x_raw
    )                                     # already uniformly sampled by the test
    naive_jerk = savgol_filter(x_grid, window_length=21, polyorder=2, deriv=3,
                               delta=extractor.dt, mode="interp")
    guarded_jerk = extractor._savgol(x_raw, deriv=3)     # polyorder auto-raised
    naive_max = float(np.max(np.abs(naive_jerk)))
    guarded_max = float(np.max(np.abs(guarded_jerk)))
    trap_ok = naive_max == 0.0 and guarded_max > 0.0
    if verbose:
        print("TRAP DEMO — polyorder must be >= deriv / polyorder phải >= bậc đạo hàm")
        print(f"   savgol_filter(deriv=3, polyorder=2) RAW scipy -> "
              f"max|jerk| = {naive_max:.3e}   <-- ALL ZEROS / TOÀN SỐ 0")
        print(f"   ParkinsonFeatureExtractor._savgol(deriv=3)  -> "
              f"max|jerk| = {guarded_max:.3e}   <-- correct / đúng")
        print()

    all_ok = trap_ok
    if verbose:
        print("-" * 78)
    for name, passed, detail in checks:
        all_ok &= bool(passed)
        if verbose:
            print(f"  [{'PASS' if passed else 'FAIL'}] {name:<52} {detail}")
    if verbose:
        print("=" * 78)

    if not all_ok:
        if verbose:
            print("SOME TESTS FAILED / MỘT SỐ TEST THẤT BẠI")
        return False
    if verbose:
        print("ALL TESTS PASSED / TẤT CẢ TEST ĐỀU ĐẠT")
    return True


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return 0 if _run_dummy_test(verbose=True) else 1


if __name__ == "__main__":
    sys.exit(main())
