#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_augmentor.py — Kinematic data augmentation for the NeuroTrace Parkinson model
data_augmentor.py — Tăng cường dữ liệu động học cho mô hình Parkinson NeuroTrace

===========================================================================
 WHY AUGMENT? / VÌ SAO PHẢI TĂNG CƯỜNG DỮ LIỆU?
===========================================================================
Handwriting datasets are small (HandPD: 92 subjects) and heavily IMBALANCED
(18 controls / 74 patients). A CNN trained on 736 strokes overfits the subject
identity instead of the disease, and the class prior biases it toward PD.

Dữ liệu nét vẽ rất nhỏ (HandPD: 92 người) và MẤT CÂN BẰNG nặng (18 khoẻ / 74
bệnh). CNN huấn luyện trên 736 nét sẽ học thuộc danh tính người vẽ thay vì học
bệnh, và lệch về lớp PD. Bộ tăng cường này tạo 10 000 mẫu CÂN BẰNG 50/50.

===========================================================================
 THE FIVE AUGMENTATIONS AND THEIR CLINICAL MEANING
 NĂM PHÉP TĂNG CƯỜNG VÀ Ý NGHĨA LÂM SÀNG
===========================================================================
1. time_warp (×0.85–1.15 on the time axis) — BRADYKINESIA INVARIANCE
   The same shape drawn faster or slower must stay the same disease. Scaling
   time by 0.85–1.15 also multiplies every velocity by 0.87–1.18, teaching the
   model that small speed differences are not diagnostic on their own.
   Cùng một hình vẽ nhanh/chậm hơn vẫn phải là cùng một bệnh. Co giãn thời gian
   còn nhân vận tốc với 0.87–1.18 → dạy mô hình rằng chênh lệch tốc độ nhỏ không
   tự nó là chẩn đoán.

2. gaussian_jitter N(0, σ) on (x, y) — SENSOR / HAND TREMOR NOISE
   Tablet quantisation, finger micro-shakes and canvas sampling jitter. σ is a
   fraction of the stroke's spatial extent so it transfers between mm and pixels.
   Nhiễu lượng tử hoá của bảng vẽ, rung tay và jitter lấy mẫu canvas. σ tính theo
   tỷ lệ kích thước nét vẽ nên dùng được cho cả mm lẫn pixel.

3. random_rotation (0–360°) — DEVICE ORIENTATION INVARIANCE
   The patient may draw the spiral in any orientation; the disease is rotation
   invariant. Xoay 0–360°: bệnh nhân có thể vẽ theo bất kỳ hướng nào; bệnh không
   phụ thuộc hướng.

4. random_scale (×0.85–1.15) — DEVICE / ZOOM INVARIANCE
   Screen size and zoom change the amplitude. Note: BIG scaling factors must NOT
   be used, because MICROGRAPHIA (smaller writing) is itself a clinical sign.
   Kích thước màn hình và mức zoom thay đổi biên độ. LƯU Ý: không được scale quá
   mạnh vì MICROGRAPHIA (chữ nhỏ) chính là một dấu hiệu lâm sàng.

5. synthetic_tremor(stroke) — GENERATIVE TREMOR AUGMENTATION (relabels to PD)
   Implants a 4–6 Hz sinusoid PERPENDICULAR to the direction of travel with the
   clinical amplitude 1.0–1.8 mm. This is how healthy strokes are turned into
   realistic PD strokes when real patients are scarce.
   Cấy sóng sin 4–6 Hz VUÔNG GÓC với hướng di chuyển, biên độ lâm sàng 1–1.8 mm.
   Đây là cách biến nét vẽ người khoẻ thành nét vẽ PD khi thiếu bệnh nhân thật.

===========================================================================
 WHY PERPENDICULAR? / VÌ SAO PHẢI VUÔNG GÓC?
===========================================================================
A tremor injected along the direction of travel only speeds the pen up and down
(it would be absorbed by the time warp); injected perpendicular it makes the
line wobble, which is exactly what a resting/postural tremor looks like on
paper. The perpendicular direction is also the component a radial-only feature
(the distance to the spiral centre) would MISS.

Run cấy dọc theo hướng vẽ chỉ làm bút nhanh/chậm hơn (bị phép time_warp "nuốt"
mất); cấy VUÔNG GÓC mới tạo nét run ngoằn ngoèo — đúng dạng run khi nghỉ/tư thế
trên giấy, và cũng chính là thành phần mà đặc trưng chỉ-theo-bán-kính sẽ BỎ SÓT.

===========================================================================
 TRAP / BẪY: groupby("trial") MERGES SPIRAL AND MEANDER
===========================================================================
The loader emits trial index 1..4 for spiral AND 1..4 for meander, so grouping a
long-format DataFrame by "trial" alone silently splices two different drawings
together; inside the merged group ``t_ms`` restarts at 0 and the derivative
explodes (~1e15). We ALWAYS group by ["subject_id", "task", "trial"].
Bộ nạp sinh trial 1..4 cho spiral VÀ 1..4 cho meander; nếu groupby("trial") thì
hai nét vẽ khác nhau bị ghép chung, t quay về 0 và đạo hàm nổ (~1e15). Luôn
groupby(["subject_id", "task", "trial"]).

===========================================================================
 OUTPUTS / ĐẦU RA (all in ``data/processed`` by default, gitignored)
===========================================================================
  neurotrace_augmented_10k_X.npy   float32 (N, 512, 3) = [t (s), x, y]
  neurotrace_augmented_10k_y.npy   int8    (N,)         = 0 Healthy / 1 PD
  neurotrace_augmented_10k_meta.csv (sample_id, label, source_id, ops)
  augmentation_comparison.png       original vs the 5 augmentations

USAGE / CÁCH DÙNG
    python src/ai_model/scripts/data_augmentor.py
    python src/ai_model/scripts/data_augmentor.py --quick --target-count 2000
    python src/ai_model/scripts/data_augmentor.py --data-csv data/raw/handpd.csv
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# Agg backend — headless safe / backend Agg, không cần màn hình.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:                                        # same-folder import when run as a script
    from parkinson_data_loader import (     # type: ignore
        fetch_parkinson_dataset,
        simulate_handpd_strokes,
    )
except ImportError:                         # pragma: no cover - fallback below
    # Allow ``python scripts/data_augmentor.py`` from any working directory.
    # Cho phép chạy từ bất kỳ thư mục nào.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from parkinson_data_loader import (     # type: ignore
        fetch_parkinson_dataset,
        simulate_handpd_strokes,
    )

# ===========================================================================
# Augmentation constants / Hằng số tăng cường
# ===========================================================================
TIME_WARP_RANGE: Tuple[float, float] = (0.85, 1.15)      # × on the time axis
JITTER_FRACTION_RANGE: Tuple[float, float] = (0.004, 0.015)  # × spatial extent
ROTATION_RANGE_DEG: Tuple[float, float] = (0.0, 360.0)
SCALE_RANGE: Tuple[float, float] = (0.85, 1.15)
TREMOR_FREQ_HZ: Tuple[float, float] = (4.0, 6.0)         # PD resting tremor
# Clinical PD tremor amplitude is 1.0–1.8 mm. A HandPD spiral spans ~60 mm, so
# this equals 1.8–3.0 % of the stroke extent. Expressing it as a fraction keeps
# the augmentor unit-agnostic (mm on the tablet, pixels on the web canvas).
# Biên độ run lâm sàng 1.0–1.8 mm; spiral HandPD rộng ~60 mm → 1.8–3.0 % kích
# thước. Dùng tỷ lệ để không phụ thuộc đơn vị (mm trên bảng vẽ, pixel trên canvas).
TREMOR_AMPLITUDE_FRACTION: Tuple[float, float] = (0.018, 0.030)
N_POINTS_DEFAULT: int = 512
TARGET_COUNT_DEFAULT: int = 10_000
PD_REAL_FRACTION: float = 0.60            # 60 % real patients, 40 % synthetic
N_OPS_RANGE: Tuple[int, int] = (3, 4)     # ops combined per sample


# ===========================================================================
# Stroke container / Cấu trúc một nét vẽ
# ===========================================================================
@dataclass
class Stroke:
    """One drawing stroke / Một nét vẽ.

    ``t`` is in SECONDS, ``x``/``y`` in canvas units (mm or px). The arrays are
    always the same length and strictly increasing in ``t``.
    ``t`` tính bằng GIÂY, ``x``/``y`` theo đơn vị canvas (mm hoặc px).
    """

    stroke_id: str
    label: int                                   # 0 = Healthy, 1 = Parkinson
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    source: str = "real"                         # "real" | "synthetic_tremor"

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=float).ravel()
        self.x = np.asarray(self.x, dtype=float).ravel()
        self.y = np.asarray(self.y, dtype=float).ravel()
        if not (self.t.size == self.x.size == self.y.size):
            raise ValueError(
                f"t/x/y must have equal length / độ dài t,x,y phải bằng nhau: "
                f"{self.t.size}/{self.x.size}/{self.y.size}"
            )
        if self.t.size < 3:
            raise ValueError("a stroke needs >= 3 samples / cần >= 3 mẫu")

    # --- geometry helpers / tiện ích hình học ---------------------------
    @property
    def n_samples(self) -> int:
        return int(self.t.size)

    @property
    def duration(self) -> float:
        """Stroke duration in seconds / thời lượng nét vẽ (giây)."""
        return float(self.t[-1] - self.t[0])

    @property
    def extent(self) -> float:
        """Spatial scale = bbox diagonal (unit-agnostic) / kích thước nét vẽ."""
        w = float(np.max(self.x) - np.min(self.x))
        h = float(np.max(self.y) - np.min(self.y))
        return float(np.hypot(w, h)) or 1.0

    def copy(self, **changes: Any) -> "Stroke":
        """Return a modified copy (ops never mutate their input).

        Trả về bản sao đã sửa — các phép biến đổi không bao giờ sửa dữ liệu gốc.
        """
        data = {"stroke_id": self.stroke_id, "label": self.label, "t": self.t,
                "x": self.x, "y": self.y, "source": self.source}
        data.update(changes)
        return Stroke(**data)


# ===========================================================================
# The augmentor / Bộ tăng cường
# ===========================================================================
class KinematicDataAugmentor:
    """Kinematic augmentations for handwriting-based PD screening.

    All ``apply_*``/single-op methods take a :class:`Stroke` and return a NEW
    :class:`Stroke`; none of them mutate the input.

    Mọi phương thức biến đổi nhận và trả về :class:`Stroke` mới, không sửa bản gốc.

    Parameters / Tham số
    ----------
    seed : int
        Seed of the internal RNG for reproducibility / hạt giống ngẫu nhiên.
    n_points : int
        Number of points every exported stroke is resampled to (512).
    """

    OP_NAMES: Tuple[str, ...] = (
        "time_warp", "gaussian_jitter", "random_rotation",
        "random_scale", "synthetic_tremor",
    )

    def __init__(self, seed: int = 42, n_points: int = N_POINTS_DEFAULT) -> None:
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.n_points = int(n_points)

    # =====================================================================
    # 1) Individual operations / Các phép biến đổi riêng lẻ
    # =====================================================================
    def time_warp(
        self, stroke: Stroke, factor: Optional[float] = None
    ) -> Tuple[Stroke, Dict[str, Any]]:
        """Rescale the time axis by ×0.85–1.15 (bradykinesia invariance).

        ``t' = t0 + (t - t0) * factor``. factor > 1 = drawn slower (all velocities
        divided by factor), factor < 1 = faster. Handwriting remains the same
        drawing, so the label is unchanged.
        Co giãn trục thời gian: factor > 1 = vẽ chậm hơn (mọi vận tốc chia cho
        factor), factor < 1 = nhanh hơn. Hình vẽ không đổi nên nhãn giữ nguyên.
        """
        if factor is None:
            factor = float(self.rng.uniform(*TIME_WARP_RANGE))
        factor = float(np.clip(factor, 0.5, 2.0))
        t0 = float(stroke.t[0])
        t_new = t0 + (stroke.t - t0) * factor
        return stroke.copy(t=t_new), {"factor": round(factor, 4)}

    def gaussian_jitter(
        self, stroke: Stroke, sigma: Optional[float] = None
    ) -> Tuple[Stroke, Dict[str, Any]]:
        """Add N(0, σ) white noise to (x, y) (sensor + finger micro-shake).

        σ defaults to 0.4–1.5 % of the stroke extent. Independent noise per axis
        (a real stylus has no reason to prefer a direction). It is applied to
        POSITION only — the timestamps stay untouched, so it does not fake a
        drawing-speed change.
        σ mặc định bằng 0.4–1.5 % kích thước nét vẽ; nhiễu độc lập trên từng trục
        và chỉ tác động lên VỊ TRÍ, không đổi thời gian → không tạo tốc độ giả.
        """
        if sigma is None:
            sigma = float(self.rng.uniform(*JITTER_FRACTION_RANGE)) * stroke.extent
        sigma = float(max(0.0, sigma))
        x = stroke.x + self.rng.normal(0.0, sigma, stroke.x.size)
        y = stroke.y + self.rng.normal(0.0, sigma, stroke.y.size)
        return stroke.copy(x=x, y=y), {
            "sigma": round(sigma, 4),
            "sigma_fraction": round(sigma / stroke.extent, 5),
        }

    def random_rotation(
        self, stroke: Stroke, angle_deg: Optional[float] = None
    ) -> Tuple[Stroke, Dict[str, Any]]:
        """Rotate the stroke by 0–360° around its centroid (orientation invariance).

        Xoay nét vẽ 0–360° quanh tâm của nó (bất biến theo hướng vẽ).
        """
        if angle_deg is None:
            angle_deg = float(self.rng.uniform(*ROTATION_RANGE_DEG))
        theta = np.deg2rad(float(angle_deg) % 360.0)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        cx, cy = float(np.mean(stroke.x)), float(np.mean(stroke.y))
        dx, dy = stroke.x - cx, stroke.y - cy
        x = cx + dx * cos_t - dy * sin_t
        y = cy + dx * sin_t + dy * cos_t
        return stroke.copy(x=x, y=y), {"angle_deg": round(float(angle_deg) % 360.0, 3)}

    def random_scale(
        self, stroke: Stroke, factor: Optional[float] = None
    ) -> Tuple[Stroke, Dict[str, Any]]:
        """Uniformly scale by ×0.85–1.15 around the centroid (zoom invariance).

        Deliberately NARROW: micrographia (abnormally small writing) is itself a
        Parkinson sign, so large scale changes would destroy the label.
        Cố ý chỉ 0.85–1.15: micrographia (chữ nhỏ bất thường) cũng là dấu hiệu
        Parkinson nên scale mạnh sẽ phá nhãn.
        """
        if factor is None:
            factor = float(self.rng.uniform(*SCALE_RANGE))
        factor = float(np.clip(factor, 0.5, 2.0))
        cx, cy = float(np.mean(stroke.x)), float(np.mean(stroke.y))
        x = cx + (stroke.x - cx) * factor
        y = cy + (stroke.y - cy) * factor
        return stroke.copy(x=x, y=y), {"factor": round(factor, 4)}

    def synthetic_tremor(
        self, stroke: Stroke, freq_hz: Optional[float] = None,
        amplitude: Optional[float] = None, relabel: bool = True,
    ) -> Tuple[Stroke, Dict[str, Any]]:
        """Implant a 4–6 Hz tremor PERPENDICULAR to the direction of travel.

        Amplitude defaults to 1.8–3.0 % of the stroke extent (≈1.0–1.8 mm for a
        HandPD spiral — the clinical PD range). The label becomes 1 (Parkinson)
        because a 4–6 Hz resting tremor is the defining parkinsonian sign; this
        is the generative step that manufactures PD strokes from healthy ones.

        Biên độ mặc định 1.8–3.0 % kích thước (≈1.0–1.8 mm với spiral HandPD —
        đúng dải lâm sàng). Nhãn chuyển thành 1 vì run nghỉ 4–6 Hz là dấu hiệu
        đặc trưng của Parkinson; đây là bước "sinh" nét vẽ PD từ người khoẻ.
        """
        if freq_hz is None:
            freq_hz = float(self.rng.uniform(*TREMOR_FREQ_HZ))
        freq_hz = float(np.clip(freq_hz, 3.0, 12.0))
        if amplitude is None:
            amplitude = (float(self.rng.uniform(*TREMOR_AMPLITUDE_FRACTION))
                         * stroke.extent)
        amplitude = float(max(0.0, amplitude))

        # Unit normal of the travel direction / pháp tuyến đơn vị của hướng vẽ
        dx = np.gradient(stroke.x)
        dy = np.gradient(stroke.y)
        norm = np.hypot(dx, dy)
        norm[norm < 1e-12] = 1e-12
        nx, ny = -dy / norm, dx / norm

        phase = float(self.rng.uniform(0.0, 2.0 * np.pi))
        wave = amplitude * np.sin(2.0 * np.pi * freq_hz * stroke.t + phase)
        x = stroke.x + nx * wave
        y = stroke.y + ny * wave
        label = 1 if relabel else stroke.label
        out = stroke.copy(x=x, y=y, label=label, source="synthetic_tremor")
        return out, {
            "freq_hz": round(freq_hz, 3),
            "amplitude": round(amplitude, 4),
            "amplitude_fraction": round(amplitude / stroke.extent, 5),
            "phase_rad": round(phase, 3),
        }

    # =====================================================================
    # 2) Pipeline / Quy trình tăng cường
    # =====================================================================
    @staticmethod
    def strokes_from_long_dataframe(df: "Any", verbose: bool = True) -> List[Stroke]:
        """Split a long-format loader DataFrame into :class:`Stroke` objects.

        Groups by ["subject_id", "task", "trial"] — NEVER by "trial" alone
        (see the TRAP note at the top of this file): trial ids are reused by the
        spiral and the meander, and merging them makes ``t_ms`` restart at 0.

        Nhóm theo ["subject_id", "task", "trial"] — TUYỆT ĐỐI không groupby
        "trial" (xem ghi chú BẪY ở đầu file): id trial bị dùng lại cho cả spiral
        và meander, gộp chung sẽ khiến t_ms quay về 0.
        """
        import pandas as pd  # local import: keeps the module importable without pandas

        if not isinstance(df, pd.DataFrame):
            raise TypeError("input_samples must be a long-format pandas DataFrame "
                            "/ phải là DataFrame dạng long")
        required = {"subject_id", "task", "x", "y"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"missing columns / thiếu cột: {sorted(missing)}")

        label_col = "label" if "label" in df.columns else None
        group_col = "group" if "group" in df.columns else None
        if label_col is None and group_col is None:
            raise ValueError("need a 'label' or 'group' column / thiếu cột nhãn")
        t_col = "t_ms" if "t_ms" in df.columns else "t"
        if t_col not in df.columns:
            raise ValueError("need a 't_ms' (or 't') column / thiếu cột thời gian")
        scale = 0.001 if t_col == "t_ms" else 1.0     # ms -> s / ms sang giây

        group_cols = ["subject_id", "task", "trial"] if "trial" in df.columns else \
            ["subject_id", "task"]
        strokes: List[Stroke] = []
        for key, part in df.groupby(group_cols, sort=False):
            part = part.sort_values(t_col, kind="stable")
            if label_col is not None:
                label = int(part[label_col].iloc[0])
            else:
                raw = str(part[group_col].iloc[0]).strip().lower()
                label = 0 if raw in ("healthy", "control", "hc", "0") else 1
            if part[t_col].to_numpy().size < 3:
                continue                              # too short to be a stroke
            stroke_id = "-".join(str(k) for k in (key if isinstance(key, tuple)
                                                  else (key,)))
            strokes.append(Stroke(
                stroke_id=stroke_id, label=label,
                t=part[t_col].to_numpy(float) * scale,
                x=part["x"].to_numpy(float), y=part["y"].to_numpy(float),
                source="real",
            ))
        if verbose:
            n_pd = sum(s.label == 1 for s in strokes)
            print(f"[STROKES] {len(strokes)} strokes "
                  f"({len(strokes) - n_pd} healthy / {n_pd} PD) "
                  f"grouped by {group_cols}")
        return strokes

    def _random_ops(
        self, allowed: Sequence[str], n_ops: int, force: Sequence[str] = (),
    ) -> List[str]:
        """Pick ``n_ops`` distinct ops (plus forced ones), in random order."""
        pool = [op for op in allowed if op not in force]
        n_ops = int(np.clip(n_ops, 1, len(allowed)))
        n_extra = max(0, n_ops - len(force))
        chosen = list(force) + list(
            self.rng.choice(pool, size=min(n_extra, len(pool)), replace=False)
        )
        order = self.rng.permutation(len(chosen))
        return [chosen[i] for i in order]

    def apply_op(self, stroke: Stroke, op: str) -> Tuple[Stroke, Dict[str, Any]]:
        """Dispatch one named operation / Thực thi một phép biến đổi theo tên."""
        if op == "time_warp":
            return self.time_warp(stroke)
        if op == "gaussian_jitter":
            return self.gaussian_jitter(stroke)
        if op == "random_rotation":
            return self.random_rotation(stroke)
        if op == "random_scale":
            return self.random_scale(stroke)
        if op == "synthetic_tremor":
            return self.synthetic_tremor(stroke)
        raise ValueError(f"unknown op '{op}' / phép biến đổi không hợp lệ")

    @staticmethod
    def resample_stroke(stroke: Stroke, n_points: int) -> Tuple[np.ndarray, np.ndarray]:
        """Resample onto ``n_points`` samples evenly spaced IN TIME.

        Returns ``(t, xy)`` where t is in seconds and xy is (n_points, 2). The
        duration is preserved (t[-1] - t[0] = duration) so bradykinesia stays
        visible to the model; only the sampling grid is regularised.
        Đưa về ``n_points`` mẫu chia đều theo THỜI GIAN, giữ nguyên tổng thời
        lượng để mô hình vẫn "thấy" chứng chậm vận động.
        """
        t_uniform = np.linspace(float(stroke.t[0]), float(stroke.t[-1]), int(n_points))
        x = np.interp(t_uniform, stroke.t, stroke.x)
        y = np.interp(t_uniform, stroke.t, stroke.y)
        return t_uniform, np.column_stack([x, y])

    def _build_sample(
        self, source_stroke: Stroke, ops: Sequence[str]
    ) -> Tuple[np.ndarray, Stroke, str]:
        """Apply the op chain then resample -> (X (n,3) float32, stroke, ops str)."""
        stroke = source_stroke
        log: List[str] = []
        for op in ops:
            stroke, params = self.apply_op(stroke, op)
            param_str = ",".join(f"{k}={v}" for k, v in params.items())
            log.append(f"{op}({param_str})" if param_str else op)
        t_out, xy = self.resample_stroke(stroke, self.n_points)
        sample = np.column_stack([t_out, xy]).astype(np.float32)   # [t, x, y]
        return sample, stroke, "|".join(log)

    def augment_pipeline(
        self,
        input_samples: "Any",
        target_count: int = TARGET_COUNT_DEFAULT,
        out_dir: Union[str, Path] = "data/processed",
        file_prefix: str = "neurotrace_augmented_10k",
        make_plot: bool = True,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Build a balanced augmented dataset from long-format loader output.

        Composition / Thành phần (for ``target_count = 10000``):
          * 5000 Healthy  — augmented healthy strokes (never tremor-injected, so
            the label stays clean / không bao giờ cấy run để nhãn sạch)
          * 5000 PD       — 60 % (3000) augmented REAL PD strokes,
                            40 % (2000) healthy strokes turned PD by
                            ``synthetic_tremor``. If the input has no real PD
                            stroke the PD half is 100 % synthetic.

                Every sample gets a random chain of 3–4 ops and is resampled to
        ``n_points`` (512) samples of ``[t, x, y]``.

        KNOWN LIMITATION / HẠN CHẾ ĐÃ BIẾT — read before training:
        the synthetic PD samples are built from HEALTHY strokes, so they carry
        the healthy drawing SPEED (verified: mean velocity 91 vs 46 units/s for
        augmented real PD strokes, while their tremor is a correct 5.0 Hz with an
        axis tremor ratio of 0.61 vs 0.17 for healthy). A model trained on this
        dataset can therefore learn "fast => healthy" from the synthetic half and
        under-use the tremor. If that shows up in validation, either lower
        ``PD_REAL_FRACTION``'s synthetic share, or bias the ``time_warp`` of the
        synthetic samples toward > 1 (slower) to imitate bradykinesia.
        Mẫu PD tổng hợp được sinh từ nét vẽ NGƯỜI KHOẺ nên vẫn giữ TỐC ĐỘ của
        người khoẻ (đo được: vận tốc trung bình 91 so với 46 đơn vị/s ở mẫu PD
        thật, trong khi run đúng 5.0 Hz và tỷ lệ run theo trục là 0.61 so với
        0.17). Vì vậy mô hình có thể học "nhanh ⇒ khoẻ" từ nửa tổng hợp mà dùng
        ít thông tin run. Nếu thấy hiện tượng này khi validation, hãy giảm tỷ lệ
        mẫu tổng hợp hoặc thiên vị ``time_warp`` của mẫu tổng hợp về phía > 1
        (chậm hơn) để mô phỏng chứng chậm vận động.

        Returns / Trả về: dict with ``X``, ``y``, ``meta`` and the written paths.
        """
        import pandas as pd

        if target_count % 2 != 0:
            raise ValueError("target_count must be even / target_count phải chẵn")
        if isinstance(input_samples, pd.DataFrame):
            strokes = self.strokes_from_long_dataframe(input_samples, verbose=verbose)
        elif isinstance(input_samples, (list, tuple)) and input_samples and \
                isinstance(input_samples[0], Stroke):
            strokes = list(input_samples)
        else:
            raise TypeError("input_samples must be a long-format DataFrame or a "
                            "list of Stroke objects / phải là DataFrame long hoặc "
                            "danh sách Stroke")

        healthy = [s for s in strokes if s.label == 0]
        pd_real = [s for s in strokes if s.label == 1]
        if not healthy:
            raise ValueError("no healthy strokes to augment / không có nét vẽ khoẻ")

        n_per_class = target_count // 2
        n_pd_real = int(round(n_per_class * PD_REAL_FRACTION)) if pd_real else 0
        n_pd_synth = n_per_class - n_pd_real
        if verbose:
            print(f"[PLAN] healthy={n_per_class} (from {len(healthy)} strokes), "
                  f"PD: {n_pd_real} augmented real + {n_pd_synth} synthetic-tremor "
                  f"(from {len(pd_real)} real PD strokes)")

        healthy_ops_pool = ("time_warp", "gaussian_jitter", "random_rotation",
                            "random_scale")
        pd_ops_pool = healthy_ops_pool + ("synthetic_tremor",)

        X_list: List[np.ndarray] = []
        y_list: List[int] = []
        meta_rows: List[Dict[str, Any]] = []

        def _emit(source: Stroke, forced: Sequence[str], pool: Sequence[str],
                  label: int, origin: str) -> None:
            n_ops = int(self.rng.integers(N_OPS_RANGE[0], N_OPS_RANGE[1] + 1))
            ops = self._random_ops(pool, n_ops, force=forced)
            sample, out_stroke, ops_str = self._build_sample(source, ops)
            idx = len(X_list)
            X_list.append(sample)
            y_list.append(int(label))
            meta_rows.append({
                "sample_id": f"nt_{idx:06d}",
                "label": int(label),
                "source_id": str(source.stroke_id),
                "origin": origin,                  # 'real' | 'synthetic_tremor'
                "ops": ops_str,
                "duration_s": round(float(out_stroke.duration), 4),
            })

        # --- Healthy half / nửa khoẻ mạnh -----------------------------------
        # NOTE: no synthetic_tremor here — injecting a 4-6 Hz tremor into a
        # healthy stroke would create a PD-looking stroke with label 0 and teach
        # the model to ignore the tremor. / Không cấy run vào nửa khoẻ: nếu không
        # sẽ tạo ra nét "giống PD" nhưng nhãn 0 → dạy mô hình bỏ qua run.
        for _ in range(n_per_class):
            src = healthy[int(self.rng.integers(len(healthy)))]
            _emit(src, forced=(), pool=healthy_ops_pool, label=0, origin="real")

        # --- PD half / nửa PD ------------------------------------------------
        for _ in range(n_pd_real):
            src = pd_real[int(self.rng.integers(len(pd_real)))]
            _emit(src, forced=(), pool=pd_ops_pool, label=1, origin="real")
        for _ in range(n_pd_synth):
            src = healthy[int(self.rng.integers(len(healthy)))]
            _emit(src, forced=("synthetic_tremor",), pool=pd_ops_pool, label=1,
                  origin="synthetic_tremor")

        X = np.stack(X_list).astype(np.float32)              # (N, 512, 3)
        y = np.asarray(y_list, dtype=np.int8)
        meta = pd.DataFrame(meta_rows)
        # pandas 3.x: keep text columns as ``object`` — never let a string end up
        # in a float column. / pandas 3.x: cột chuỗi phải là ``object``, không để
        # chuỗi lọt vào cột số thực.
        for col in ("sample_id", "source_id", "origin", "ops"):
            meta[col] = meta[col].astype(object)

        # --- self-check / tự kiểm tra ---------------------------------------
        self._self_check(X, y, meta)

        # --- export / xuất file ---------------------------------------------
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        x_path = out_path / f"{file_prefix}_X.npy"
        y_path = out_path / f"{file_prefix}_y.npy"
        meta_path = out_path / f"{file_prefix}_meta.csv"
        np.save(x_path, X)
        np.save(y_path, y)
        meta.to_csv(meta_path, index=False)

        plot_path = None
        if make_plot:
            plot_path = self.plot_augmentation_comparison(
                healthy[0], out_path / "augmentation_comparison.png"
            )

        if verbose:
            n_synth = int((meta["origin"] == "synthetic_tremor").sum())
            print(f"[SAVE] X {X.shape} {X.dtype} -> {x_path}")
            print(f"[SAVE] y {y.shape} {y.dtype} -> {y_path}  "
                  f"(0: {int((y == 0).sum())} / 1: {int((y == 1).sum())})")
            print(f"[SAVE] meta {meta.shape} -> {meta_path}  "
                  f"(synthetic-tremor samples: {n_synth})")
            if plot_path:
                print(f"[PLOT] {plot_path}")

        return {"X": X, "y": y, "meta": meta, "x_path": x_path, "y_path": y_path,
                "meta_path": meta_path, "plot_path": plot_path}

    @staticmethod
    def _self_check(X: np.ndarray, y: np.ndarray, meta: "Any") -> None:
        """Assert the dataset contract / Kiểm tra các ràng buộc của bộ dữ liệu."""
        print("\n" + "=" * 78)
        print("SELF-CHECK — AUGMENTED DATASET / KIỂM TRA BỘ DỮ LIỆU TĂNG CƯỜNG")
        print("=" * 78)
        n = X.shape[0]
        checks = [
            ("shape is (N, n_points, 3)", X.ndim == 3 and X.shape[2] == 3,
             f"{X.shape}"),
            ("0 NaN / Inf in X", bool(np.isfinite(X).all()),
             f"nan={int(np.isnan(X).sum())}, inf={int(np.isinf(X).sum())}"),
            ("X dtype float32", X.dtype == np.float32, str(X.dtype)),
            ("y length matches X", y.shape[0] == n, f"{y.shape[0]} vs {n}"),
            ("meta rows match X", len(meta) == n, f"{len(meta)} vs {n}"),
            ("class balance is 50/50", int((y == 0).sum()) == int((y == 1).sum()),
             f"{int((y == 0).sum())} vs {int((y == 1).sum())}"),
            ("labels are only {0, 1}", set(np.unique(y).tolist()) <= {0, 1},
             str(np.unique(y).tolist())),
            ("timestamps increase within each sample",
             bool(np.all(np.diff(X[:, :, 0], axis=1) > 0)),
             "monotonic t / t tăng dần"),
            ("every sample has non-zero spread",
             bool(np.all(X[:, :, 1:].std(axis=(1, 2)) > 1e-6)),
             "no frozen/constant stroke / không có nét vẽ 'chết'"),
        ]
        ok = True
        for name, passed, detail in checks:
            ok &= bool(passed)
            print(f"  [{'PASS' if passed else 'FAIL'}] {name:<42} {detail}")
        print("=" * 78)
        if not ok:
            raise AssertionError("augmented dataset failed self-check / "
                                 "bộ dữ liệu không đạt kiểm tra")

    # =====================================================================
    # 3) Visual comparison / So sánh trực quan
    # =====================================================================
    def plot_augmentation_comparison(
        self, source: Stroke, out_path: Union[str, Path]
    ) -> Path:
        """Figure: the original stroke next to the 5 individual augmentations.

        Hình: nét vẽ gốc cạnh 5 phép biến đổi riêng lẻ — thấy rõ run 5 Hz vuông
        góc ở ``synthetic_tremor``.
        """
        variants: List[Tuple[str, Stroke]] = [("original / gốc", source)]
        titles = {
            "time_warp": "1. time_warp (×0.85–1.15 t)",
            "gaussian_jitter": "2. gaussian_jitter N(0,σ)",
            "random_rotation": "3. random_rotation 0–360°",
            "random_scale": "4. random_scale ×0.85–1.15",
            "synthetic_tremor": "5. synthetic_tremor 4–6 Hz\n(perpendicular, label→1)",
        }
        for op in self.OP_NAMES:
            variant, _ = self.apply_op(source, op)
            variants.append((titles[op], variant))

        fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
        for ax, (title, stroke) in zip(axes.ravel(), variants):
            # colour by time so the drawing direction is visible
            # tô màu theo thời gian để thấy hướng vẽ
            sc = ax.scatter(stroke.x, stroke.y, c=stroke.t, s=3, cmap="viridis")
            ax.set_title(f"{title}\nlabel={stroke.label}, "
                         f"duration={stroke.duration:.2f}s", fontsize=10)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.3)
            fig.colorbar(sc, ax=ax, label="t (s)", shrink=0.8)
        fig.suptitle(
            "Augmentation variants / Các biến thể tăng cường "
            "(HandPD-style kinematic augmentor)", fontsize=13,
        )
        # hspace: the second row of titles sits close to the first row's ticks
        # / tăng khoảng cách dọc để tiêu đề hàng 2 không đè lên nhãn hàng 1
        fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=3.0, w_pad=2.0)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return out_path


# ===========================================================================
# Demo / fallback data source
# ===========================================================================
def _fallback_strokes(n_subjects: int = 12, seed: int = 0) -> "Any":
    """Tiny self-contained generator used ONLY if parkinson_data_loader import fails.

    Bộ sinh nhỏ và tự chứa, CHỈ dùng khi không import được parkinson_data_loader.
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    rows = []
    for subject in range(1, n_subjects + 1):
        is_pd = subject % 3 != 1                       # ~2/3 of subjects are PD
        for task in ("spiral", "meander"):
            for trial in range(1, 3):
                duration = rng.uniform(9.0, 12.0) if is_pd else rng.uniform(6.0, 8.5)
                n = int(duration * 100) + 1
                t = np.arange(n) / 100.0
                u = t / t[-1]
                theta = 2 * np.pi * 4 * u
                r = 40.0 * u * (0.8 if is_pd else 1.0)
                x = r * np.cos(theta) + rng.normal(0, 0.3 if is_pd else 0.1, n)
                y = r * np.sin(theta) + rng.normal(0, 0.3 if is_pd else 0.1, n)
                if is_pd:
                    osc = 1.4 * np.sin(2 * np.pi * 5.0 * t)
                    d = np.gradient(x), np.gradient(y)
                    nrm = np.hypot(*d) + 1e-9
                    x = x + (-d[1] / nrm) * osc
                    y = y + (d[0] / nrm) * osc
                rows.append(pd.DataFrame({
                    "subject_id": np.full(n, subject, dtype=np.int64),
                    "group": np.full(n, "pd" if is_pd else "healthy", dtype=object),
                    "label": np.full(n, int(is_pd), dtype=np.int8),
                    "task": np.full(n, task, dtype=object),
                    "trial": np.full(n, trial, dtype=np.int16),
                    "t_ms": (t * 1000).astype(np.int64),
                    "x": x.astype(np.float32), "y": y.astype(np.float32),
                }))
    return pd.concat(rows, ignore_index=True)


def repo_root() -> Path:
    """<repo>/src/ai_model/scripts/x.py -> <repo>."""
    return Path(__file__).resolve().parents[3]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Kinematic data augmentor for NeuroTrace / Bộ tăng cường dữ "
                    "liệu động học cho NeuroTrace.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-csv", default=None,
                   help="real dataset CSV (subject_id, group, task, t_ms, x, y); "
                        "if omitted the HandPD protocol is simulated / nếu bỏ "
                        "trống sẽ giả lập")
    p.add_argument("--target-count", type=int, default=TARGET_COUNT_DEFAULT,
                   help="total samples to generate (even) / tổng số mẫu (chẵn)")
    p.add_argument("--n-points", type=int, default=N_POINTS_DEFAULT,
                   help="samples per stroke / số điểm mỗi nét")
    p.add_argument("--out-dir", default=str(repo_root() / "data" / "processed"),
                   help="output directory / thư mục đầu ra")
    p.add_argument("--seed", type=int, default=42, help="RNG seed / hạt giống")
    p.add_argument("--quick", action="store_true",
                   help="small fast run for smoke tests / chạy nhanh để test")
    p.add_argument("--no-plot", action="store_true",
                   help="skip the comparison figure / không vẽ hình")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    target = args.target_count
    n_subjects = 92
    if args.quick:
        target = min(target, 2000)
        n_subjects = 12

    # --- data source / nguồn dữ liệu ---------------------------------------
    # Real CSV when provided, otherwise the HandPD protocol simulator from
    # parkinson_data_loader / CSV thật nếu có, nếu không thì dùng bộ giả lập
    # giao thức HandPD của parkinson_data_loader.
    try:
        if args.data_csv:
            df = fetch_parkinson_dataset(data_csv=args.data_csv, verbose=True)
        else:
            df = simulate_handpd_strokes(
                n_subjects=n_subjects,
                n_healthy=max(1, int(round(n_subjects * 0.2))),
                seed=args.seed, verbose=True,
            )
        if df.empty:
            raise ValueError("empty dataset / dữ liệu rỗng")
    except Exception as exc:                                # pragma: no cover
        print(f"[WARN] loader unavailable ({exc}) -> using the built-in fallback "
              f"generator / dùng bộ sinh dự phòng")
        df = _fallback_strokes(n_subjects=12, seed=args.seed)

    augmentor = KinematicDataAugmentor(seed=args.seed, n_points=args.n_points)
    result = augmentor.augment_pipeline(
        df, target_count=target, out_dir=args.out_dir,
        file_prefix=f"neurotrace_augmented_{target // 1000}k" if target % 1000 == 0
        else "neurotrace_augmented",
        make_plot=not args.no_plot,
    )

    y = result["y"]
    print(f"\n[DONE] balanced dataset / bộ dữ liệu cân bằng: "
          f"{len(y)} samples, {int((y == 0).sum())} healthy / "
          f"{int((y == 1).sum())} PD, X.shape={result['X'].shape}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
