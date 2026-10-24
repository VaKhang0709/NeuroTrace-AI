# V5 training-to-browser input contract audit

Version `v5-source-audit-1` · 2026-10-24 · source provenance hashes in companion JSON.

**Conclusion:** The source-defined transformation is recoverable. The EXACT deployed
production contract is **UNKNOWN** until weights, scaler, graph and a licensed real
sample are inspected together. No retraining, threshold change or production parity
claim is made. Mouse/one healthy phone observations are not clinical validation.

## Artifact availability

Searched the repository and `/home/user`, `/content`, `/tmp` for checkpoint `.pth`,
ONNX `.onnx`, virtual `.npz`, scaler JSON and CSV data. None of the production
artifacts or real input trajectories were found. `/content/drive/MyDrive/NEUROTRACE`
is not mounted here. Prior conversation artifact claims are not artifact evidence.
Original training, virtual expansion, export and parity source scripts ARE present
under `src/ai_model/scripts/`; their line-1 Colab magics must be removed for execution.
No external Drive data or live browser deployment was fetched in this local audit.
The inventory is absence in this workspace, not a claim of absence in remote storage.
No raw identifiable trajectories are included in this PR.

## Findings and evidence

### raw_coordinates — UNKNOWN

Loaders copy numeric x/y directly; no source-unit, origin, axis direction, DPI or canvas-size conversion is specified. Physical Wacom units and actual app coordinate convention cannot be established without raw metadata.

Evidence: `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L337-411`; `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L414-465`.

### time_repair — VERIFIED

repair_time_axis chooses scale from median positive raw increment: >50000 -> 1e-6; otherwise >0.5 -> 1e-3; else 1. It subtracts first timestamp and replaces nonpositive or >20*median increments by median. This is heuristic seconds conversion, not proof of original units.

Evidence: `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L311-334`.

### time_fallback — VERIFIED

UCI missing timestamp column uses index/133; app completely unusable timestamps use index/60 and synthetic_time flag. Partial NaNs may propagate through repair_time_axis and be removed later.

Evidence: `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L370-380`; `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L450-465`; `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L488-490`.

### uci_segmentation — VERIFIED

UCI splits only contiguous test-ID runs (SST/DST/STCP); runs need >=24 samples and repaired duration >=0.4s. No explicit pen-up split is used. Subject identity comes from normalized file stem.

Evidence: `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L204-208`; `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L384-411`.

### app_strokes — VERIFIED

Nested stroke points are flattened in input order into one record/trial, losing pen-up boundaries. No explicit sort by time or pen-up gap segmentation; gaps are time-repaired then interpolated. Records require >=20 points, duration >=0.4s.

Evidence: `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L414-465`; `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L467-525`.

### sample_trace — UNKNOWN

No real raw trajectory is available in this workspace. A numerical REAL-sample trace (raw to final tensor) cannot be produced; synthetic regression fixtures are not real participant evidence.

Evidence: `workspace inventory recorded in docs/v5_input_contract.md#artifact-availability`.

### finite_and_dt — VERIFIED

Drop samples whose x/y/t are nonfinite; require >=20. Raw dt=diff(t), first dt copied from second. Require >=16 positive dt values; nonpositive dt replaced by median. Rebuilt t_clean cumulatively clips subsequent dt to [0.05*median,10*median]. Reject duration <0.4s.

Evidence: `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L610-633`.

### grid — MISMATCH

Nominal 100 Hz grid uses n=max(48,round(duration*100)+1), tg=linspace(0,duration,n). Actual interval is duration/(n-1), not necessarily 0.01 seconds (especially minimum 48 points). A fixed exact 100 Hz browser grid is not identical.

Evidence: `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L633-635`.

### filter — VERIFIED

Fourth-order 12 Hz low-pass Butterworth SOS designed with fs=100; sosfiltfilt zero-phase applied to linearly interpolated x,y,raw repaired dt on tg. Real pressure also filtered later. z is interpolated but not filtered. Proxy pressure and velocity are not explicitly low-pass filtered after derivation.

Evidence: `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L63`; `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L635-657`.

### trial_spatial — VERIFIED

Filtered x/y centered by whole-trial medians for radius r0. sc0=95th percentile radius (fallback 1 if <=1e-9/nonfinite). This trial scale is used in velocity and z; final window x/y use separate normalization.

Evidence: `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L638-650`.

### velocity — VERIFIED

vx,vy=gradient((filtered_position-trial_median)/sc0,0.01); speed=hypot(vx,vy). It is computed AFTER interpolation/filtering, BEFORE window extraction/256 interpolation, with fixed 0.01 derivative spacing even if tg spacing differs. Units are trial-scale-normalized displacement per nominal second, NOT raw pixels/s.

Evidence: `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L638-652`.

### pressure — VERIFIED

Use real pressure only if trial.has_real_pressure and finite pressure with >=4 distinct rounded(4dp) values; loaders impose finite-count requirements. Fill missing pressure by median, interpolate and filter. Else proxy uses full-trial speed min-max (epsilon 1e-6): 0.65*(1-0.35*s_norm)+0.15*clip(r0/sc0,0,1.5). No pointer-type check or event-local speed normalization is used.

Evidence: `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L399-402`; `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L514-515`; `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L583-586`; `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L651-657`.

### windows — VERIFIED

Stack [x,y,z,pressure,speed,filtered_dt] as (time,6). Window length default 256, stride 64; >=24 samples, short windows retained, append last full window at n-window if not aligned. Linear interpolation by normalized sample index to 256, not a second timestamp-aware interpolation.

Evidence: `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L136-139`; `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L589-608`; `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L658-663`.

### window_normalization — VERIFIED

After 256 interpolation: x/y subtract window mean, divide both by sqrt(mean(wx^2+wy^2)); skip <=1e-8. Pressure=(p-mean(p))/(std(p)+1e-5). dt=(filtered_dt-median(filtered_dt))/(median(filtered_dt)+1e-6). Speed unchanged; z retained if selected. Final dt is a dimensionless relative interval, not raw seconds or milliseconds.

Evidence: `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L663-674`.

### features — UNKNOWN

Code canonical order [x,y,z,pressure,velocity,dt], selection drops configured channel indices preserving order. A five-channel drop-z run implies [x,y,pressure,velocity,dt], but actual production feature order requires unavailable checkpoint feature_names/scaler JSON. Defaults do NOT drop z.

Evidence: `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L58`; `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L125`; `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L724-725`; `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L1339-1356`.

### scaler — UNKNOWN

Actual production mean and scale values are unavailable. Code fits sklearn StandardScaler on training-only windows transposed (N,L,C) and flattened (N*L,C), then restores (N,C,L) float32. No per-browser-drawing fit is permitted by this algorithm.

Evidence: `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L943-954`; `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L1027-1029`; `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L1354-1355`.

### tensor_layout — VERIFIED

Training arrays are logical N,C,L float32. Export dummy uses (1,checkpoint in_channels,checkpoint seq_len); batch dynamic. Browser Float32Array must be channel-major: offset=(batch*C+channel)*L+time. Logical axes, not a requirement that all NumPy views be physically contiguous.

Evidence: `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L663-674`; `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L757-768`; `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L943-954`; `src/ai_model/scripts/export_onnx_v5.py:L72-94`.

### onnx_contract — UNKNOWN

Original exporter requests input named input, output logits, dynamic batch and fixed C,L from checkpoint; model.forward emits (B,) raw logits. Actual serialized ONNX graph names/shapes/weights are unavailable and not inspected. Requested export: opset17, dynamo=False, do_constant_folding=True, eval before/after.

Evidence: `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L872-886`; `src/ai_model/scripts/export_onnx_v5.py:L65-101`; `src/ai_model/scripts/export_onnx_v5.py:L121-138`.

### aggregation_threshold — VERIFIED

Code evaluates mean per-window sigmoid probability per parent subject at cfg.threshold default 0.50; this is not sigmoid(mean(logits)). Training shifts clinical bias by a training-set-only calibrated logit boundary, so output is not guaranteed clinically calibrated probability. Actual production metadata threshold UNKNOWN; not changed here.

Evidence: `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L140`; `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L966-975`; `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L1129-1145`; `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L1356`.

### parent_mapping — VERIFIED

Virtual session writer stores parent_subject_id and map; loader takes map entry (fallback embedded parent). Dataset buckets all windows by parent, caps real and virtual separately, drops virtual-only parents without real windows. This is code intent, not validation of unavailable actual parent map.

Evidence: `src/ai_model/scripts/expand_virtual_sessions_v5.py:L229-245`; `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L528-575`; `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L707-753`.

### split_guards — VERIFIED

Holdout stratifies parent IDs by (source,label). CV excludes holdout; GroupKFold operates on parent IDs. Training masks select all windows of training parents, validation/holdout select only real windows of held-out parents. Assertions check parent disjointness. Scalers fit only training fold; online rotations happen only to training batches.

Evidence: `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L906-954`; `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L1027-1029`; `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L1074-1082`; `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L1261-1281`; `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L1306-1318`.

### split_limitations — UNKNOWN

Actual executed partitions and aliases cannot be verified without reports/maps/data. Filename/record-derived IDs may alias one person. No cross-source identity linkage. GroupKFold is not stratified; shuffling subject order need not yield different repeated partitions with default GroupKFold. Window overlap is acceptable only with correct parent identity. Validation selects epochs for nonadversarial runs, so CV is not an untouched test estimate.

Evidence: `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L204-208`; `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L384`; `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L493-506`; `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L926-940`; `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L1109-1128`.

### probe_limitations — VERIFIED

Device probe trains logistic regressions on frozen final-model embeddings using subject-grouped 5-fold CV, primarily HC-only. Encoder/scaler are NOT refit per probe fold; bundle includes final-model training and holdout parents, and probe AUC is computed over windows. Thus this is not fully nested unseen-subject representation evaluation and cannot establish domain invariance.

Evidence: `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L1148-1201`; `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L1328`.

### device_label — UNKNOWN

Loader supports app PD labels but defaults absent/unrecognized labels to HC. Actual device-by-label counts are unavailable. If observed PD are all Wacom and mobile data only HC, mobile-PD sensitivity cannot be measured regardless of device-probe AUC; HC-only device discrimination is not mobile-PD clinical validation.

Evidence: `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L508-515`; `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L1170-1184`.

### browser_comparison — UNKNOWN

No deployed browser bundle was found or audited. A browser using raw pixel velocity, raw dt Z-score only, per-event pressure normalization, or no filtering/window normalization would mismatch the source algorithm; this is conditional, not a finding about the live app.

Evidence: `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L610-674`.

### module_scope — MISMATCH

Modular V5Preprocessor accepts already-repaired strictly increasing seconds, whereas original raw loaders repair/reset times and may synthesize them. It is not an end-to-end raw loader replacement. Passing its limited synthetic comparison cannot establish artifact or browser parity. Modular loader enforces five channels; original architecture/export are configurable.

Evidence: `src/data/preprocessor.py:L111-148`; `src/export/export_onnx.py:L12-25`; `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L311-334`.

### parity — UNKNOWN

No production held-out parity run performed: production weights, ONNX, scaler, real held-out inputs, PyTorch and ONNX Runtime unavailable at audit start. Original parity script samples 25 windows/device, not necessarily 50 distinct subjects or held-out parents. New module optionally reads standardized real windows but does not validate holdout provenance.

Evidence: `src/ai_model/scripts/verify_onnx_parity.py:L69-105`; `src/eval/verify_parity.py:L14-33`.

## Real sample trace status

**BLOCKED / UNKNOWN.** No real input sample is locally available. Therefore raw
coordinate ranges, timestamp increments, pressure range, window count, scaler
values, final tensor and logits for a REAL sample are not reported. Synthetic test
fixtures exercise source semantics only and are explicitly not substituted for this
missing deliverable. Once authorized data are mounted, retain per-stage trace values
locally (loader → time repair → uniform grid → filter → velocity/proxy → windows →
window normalization → scaler → tensor → logits); publish only nonidentifying
aggregate diagnostics after license/privacy review.

## Reproducible parity procedure (not executed)

Missing prerequisites: production `.pth`, `.onnx`, scaler JSON, verified held-out
parent list and corresponding licensed real windows; PyTorch/ONNX Runtime/ONNX.
Install dependencies in an isolated environment and run from repository root:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r src/requirements-v5.txt scikit-learn
python -m pytest tests/test_v5_input_contract.py tests/test_v5_modules.py
# File must contain >=50 real windows from held-out parent IDs, already processed
# by the original training transform and standardized with checkpoint scaler.
# Preserve provenance locally; do not upload raw trajectories or private IDs.
python -m src.eval.verify_parity \
  --checkpoint /secure/models/neurotrace_v5_production.pth \
  --onnx /secure/models/neurotrace_v5.onnx \
  --real-windows /secure/heldout_standardized_windows.npy \
  --report /secure/parity_report_v5.json
```

The command compares 50 synthetic multi-batch tensors and 50 supplied real windows;
it does not itself prove they are held out. First compare checkpoint `feature_names`,
`scaler_mean`, `scaler_scale`, `classification_threshold`, `in_channels`, `seq_len`
with the original scaler JSON, and inspect ORT `get_inputs()/get_outputs()` names,
shapes/types. Require matching metadata and max absolute raw-logit error `<1e-5`.
Restricted checkpoint loading can reject legacy metadata; do not blindly enable
unsafe pickle loading. Resolve with a trusted, isolated artifact conversion if needed.
Do not re-export before auditing the ORIGINAL ONNX: re-export parity alone cannot
establish identity to the existing browser model.

## Acceptance and limitations

Source findings carry per-finding references and source hashes. Regression tests
verify timestamp heuristic, flattened strokes, nominal-rate grid discrepancy,
per-window normalization and parent-mask semantics using synthetic fixtures.
They do not establish clinical performance, actual production configuration, real
parent identity, nested probe validity or browser equivalence. Previously reported
0.923 accuracy / 0.976 AUC / 0.679 device probe AUC are not independently verified
by this audit. No architecture, preprocessing, threshold or training changes were made.

## Additional mismatch: timestamp-unit ambiguity

**MISMATCH:** raw increments of 10000 are converted to 10 seconds by the
heuristic, although an upstream microsecond stream could mean 0.01 seconds.
This demonstrates ambiguity, not evidence that any real source was misconverted.
Evidence: `src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L311-334`;
`tests/test_v5_input_contract.py:test_timestamp_heuristic_seconds_and_ms`.

## Executed validation

`python -m pytest tests/test_v5_input_contract.py tests/test_v5_modules.py -q`:
**8 passed, 1 skipped** in the isolated audit environment. PyTorch is absent, so
the architecture check skipped. Source-only signal tests use NumPy/SciPy/sklearn;
no production checkpoint, model inference or browser execution was involved.
