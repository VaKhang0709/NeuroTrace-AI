# V5 JavaScript source-reference preprocessor

`src/export/v5_preprocessor.js` ports `src/data/preprocessor.py`'s
`V5Preprocessor.transform` and `extract_trial_windows`, whose signal implementation
was extracted from `src/ai_model/scripts/train_pipeline_v5_diagnostic.py`.
See `v5_input_contract.md` for original-code evidence and unresolved artifact gaps.

## Important correction to the requested stage list

Bounding-box [0,1] scaling, resampling directly to 256 before filtering, and min-max
pressure normalization would **not** reproduce the audited Python code. This module
deliberately does not implement those transformations. It preserves:

1. Raw coordinates; finite-point masking and repaired-second dt clipping.
2. Approximate 100 Hz interpolation, fourth-order 12 Hz Butterworth SOS with
   SciPy-compatible forward/backward odd padding (15 points) and steady-state
   initial conditions. Filtering happens **before** window resampling.
3. Trial-median / 95th-percentile radius spatial normalization for velocity,
   `gradient(..., 0.01)` speed; filtered real pressure or full-trial proxy.
4. Overlapping 256-point windows (stride 64, final tail included), linear
   index interpolation to 256; window x/y mean subtraction and shared RMS radius,
   pressure mean/std, dt relative to its median.
5. Float32 intermediate rounding, supplied scaler mean/scale, channel-first
   Float32Array packing in `[x, y, pressure, velocity, dt]` order.

## API

Node:

```js
const {preprocessV5} = require('./src/export/v5_preprocessor.js');
const windows = preprocessV5({x, y, timestamps, pressure}, scalerConfig);
// windows is an array, NOT a single tensor for an arbitrarily long recording.
// Each windows[i] is Float32Array(1280), logical shape [1, 5, 256].
```

Browser classic script: load the file, then call
`NeuroTraceV5.preprocessV5(input, scalerConfig)`. No bundler or npm dependency is
required. Load scaler JSON separately; the module does not fetch network resources.
Optional `{window: 256, stride: 64}` third argument matches Python defaults.

Timestamps must **already** be repaired, strictly increasing seconds. This API does
not infer source units, flatten stroke structures, or reconstruct participant IDs.
Coordinates must already follow loader numeric conversion; no DPI or canvas-unit
assumptions are made. Omit `pressure` or pass `null` for missing pressure; individual
nonfinite/missing readings in an otherwise varying pressure channel use its median.
Scaler `feature_names`, `mean`, and `scale` are required; statistics are never fitted
on the submitted trajectory. Unusable/short/stationary input throws an error.
Caller supplies appropriate resource limits for untrusted very long recordings.

All windows must be handled explicitly. Original evaluation averages per-window
sigmoid probabilities for each parent, not raw logits. No threshold, classification,
UI, ONNX execution, or clinical claims are part of this module.

## Reproducible verification

```bash
python -m venv .venv
.venv/bin/pip install numpy scipy scikit-learn pytest
PYTHON="$PWD/.venv/bin/python" node tests/test_js_preprocessor.js
.venv/bin/python -m pytest tests/test_v5_modules.py tests/test_v5_input_contract.py
```

The Node test invokes Python on the same synthetic inputs/scaler, comparing every
channel/time element of every window. It covers short and long recordings,
irregular timestamps, real/missing/constant pressure, partial missing pressure,
tail windows, non-default window/stride, packing, and rejection of invalid input.
It allows `2e-6 + 2e-6*abs(reference)` float32 error, not bitwise equality for all
possible inputs. Filter coefficients and initial states are fixed from
`scipy.signal.butter(4,12,fs=100,output='sos')` and `sosfilt_zi`.

**Scope:** This is numerical source-reference compatibility on synthetic fixtures,
not proof of the deployed model's input contract, browser score equivalence, or
clinical validity. Production artifacts remain unavailable. No production ONNX
parity or real held-out participant test is claimed. Python raw loader float32
conversion and timestamp repair must still occur upstream as audited.
