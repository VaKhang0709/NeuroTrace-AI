# TremorFormer V6 — research architecture, not a validated diagnostic

## Implemented network

Default **15,779,536 trainable parameters** (measured with PyTorch 2.5.1).

```text
explicit-seconds trajectory, native x/y + optional pressure/azimuth
  │ cubic canonical 100Hz + source-rate anti-alias when >100.5Hz
  │ device lowpass / geometry normalization / derivative + db4 features
  ▼
[N, 19, 512] ── four CNN branches k={3,7,15,31}, stride=4
                       concat 256 → projection → [N,128,256]
                               + learned + sinusoidal position
                                              │
one-hot device(5) → linear(32) ──────────────► FiLM at each block
                                              │
                         6 × Conformer blocks:
                      ½ FFN residual (expansion8)
                      8-head attention residual
                      depthwise Conv31 residual
                      ½ FFN residual → LayerNorm
                                              │
                                 attention pooling
                                              │
          ┌─────────────┬─────────────┬─────────────┬───────────────┐
       PD logits     UPDRS/132      task(3)       age(5)       GRL→device(5)
       (sigmoid only at inference)                          (+CE in loss)
```

No softmax/sigmoid in training heads. Clinical forward returns a dict including
raw logits. `predict_proba()` is inference-only. UPDRS is normalized by 132 in the
loss; multiply the regressor output by 132 for raw score interpretation. Missing
UPDRS/task/age labels are masked rather than imputed.

FiLM deliberately conditions on known device and could make device identity MORE
recoverable. GRL is not a mathematical guarantee of invariance. Prespecified
ablation: device_film=false, GRL disabled, and both enabled. Gate the measured
held-out frozen-feature HC device probe (<0.70 desired) rather than asserting it.

## Signal contract (NEW V6, not the V5 contract)

Explicit increasing second timestamps; native coordinate units recorded, no pixel
inference. Source rates >100.5Hz are reconstructed uniformly at median rate and
45Hz anti-aliased before cubic interpolation to canonical 100Hz. Other sources
are cubic-interpolated directly. Target fourth-order zero-phase Butterworth:
Wacom/stylus15Hz, touch/unknown10Hz, mouse12Hz (capped to 0.45 source median rate).
Whole-stroke median/r95 geometry normalization precedes derivatives. Source timing
jitter and cubic overshoot need measurement on real devices.

Channels, in order:

```text
x y pressure vx vy speed ax ay acceleration jerk curvature
pressure_derivative azimuth_derivative
wavelet_x_4_8 wavelet_y_4_8 wavelet_x_8_12 wavelet_y_8_12
pressure_observed azimuth_observed
```

All derivatives use the canonical 0.01s grid. Curvature denominator floor1e-8,
clamp±1000. Azimuth is unwrapped radians. Missing pressure is the documented
V5-inspired full-stroke velocity/radius proxy; missing azimuth derivative is zero.
No biomechanical claim attaches to proxy force. Observed flags prevent silent
assumption of measured sensors. Per-window population z-scores (+1e-6 epsilon)
apply to the 17 continuous channels; binary flags are retained. **This differs
from V5's pressure/dt and shared-radius window transforms**, despite the mission's
“as per V5” wording. V6 has no raw dt channel after explicit canonical resampling.

Full512 windows, stride128, include final tail. <512 samples are rejected, not
padded or duration-stretched. Ingestion/QC operators must document excluded short
trials to avoid silent selection bias. Native loaders must segment pen-up strokes
before calling this pipeline; unified records are single continuous trajectories.

`db4` level5 alone does NOT isolate 4–8Hz/8–12Hz at100Hz. Implementation reconstructs
overlapping DWT details and applies FFT frequency selection to those bands. This is
a wavelet-informed band channel with finite-stroke boundary artifacts, not a claim
that exact dyadic DWT coefficients correspond to these physiological bands.

## Compute and deployment

4× temporal stride reduces attention length512→128, enabling the requested
15–40M scale without attention over all512 tokens. AdamW parameters/gradients/
optimizer state alone require roughly 16 bytes/parameter (~252MB), excluding
activations, kernels, AMP master states and runtime. Proposed initial T4 microbatch
2–4 (effective256 by accumulation); A100 may allow larger microbatch. **GPU VRAM
and throughput are NOT measured here**, and must be profiled before long runs.
CPU full-width forward/backward and full-width ONNX parity were exercised.

Five full ONNX models are about300MB total; web-client feasibility is UNKNOWN and
likely costly. Single-member deployment is available; distillation is not
implemented or claimed. Exported device ID is a second int64 input. Final
patient classification is not a calibrated clinical probability by architecture.
