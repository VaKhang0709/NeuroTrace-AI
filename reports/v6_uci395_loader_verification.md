# UCI395 loader/schema boundary correction — 2026-10-25

Branch `arena/4edc99ce-neurotrace-ai`; PR #4 remains OPEN. **Partial acceptance:
real preparation passes; required fresh GPU execution remains BLOCKED.** No V5
artifacts or production threshold were changed. This report does not claim disease
performance, mobile validation, a clinically trained model or SOTA.

## Root cause confirmed from repository and original-format evidence

- `scripts/train_v6_gpu_standalone.py` previously read `args.records` whenever the
  option string was truthy, including a nonexistent path, before constructing any
  real records. It required a prebuilt partition. Fixed: real loader builds both
  before reading; existing records validate every index and report failures.
- `Trajectory.from_dict` requires the explicit version/time flags and exactly five
  columns `[x,y,seconds,pressure,azimuth]`. Unknown age is integer -1,not `'unknown'`.
  Missing metadata flags,four-column arrays,and nonmonotone column2 explain the
  reported errors; flags alone cannot repair a mismapped raw array.
- The old strict seven-column adapter ALREADY mapped raw timestamp5 to canonical2.
  We did not find a time-at-index3 defect there. It did not support named six-column
  variants and silently rejected any duplicate/reversal segment. The new dedicated
  adapter removes duplicates last-in-acquisition-order,then rejects remaining
  reversals with counters. It never manufactures timestamps or sorts resets away.
- Old prepare-only re-saved raw records,not features. It now actually computes and
  saves finite `[N,19,512]` features with the explicit DWT19 contract. New raw QC is
  completed before partition creation; existing locked cohorts cannot silently lose
  short/invalid parents during training or evaluation.

## Real source and layout

Mirror `https://github.com/irenemao/Handwriting-Analysis`, pinned commit
`25ca56231bc161b39b4e896f126de89284041b95`, dataset subfolder `hw_dataset/`.
GitHub API downloads: 78 text blobs,16,143,597 bytes,all Git blob SHA-1 checked.
77 actual recorded files + README; **not** a claim of equivalence to the user's
129-file Kaggle packaging. Full mirror numeric files remain private under
`/tmp/neurotrace_real_uci`,not committed. No Firebase or mobile dataset was fetched.

README and actual `P_26060001.txt` agree on headerless semicolon columns:
`X;Y;Z;Pressure;GripAngle;Timestamp;Test ID`.
First recorded row: `270;202;24;8;1270;4142080;0`.
The full source file has1303rows,TestID0=746,TestID1=557. README specifies0 static
spiral,1 dynamic spiral,2 circular motion; no real meander is established here.
Named-header task inference preserves explicit filenames without splitting parent
IDs. Task-alias tests are unit metadata adaptations,not new recorded meander cases.

Canonical mapping: raw0→x,raw1→y,raw5→seconds,raw3→pressure,azimuth→missing.
Raw2 is Z,NOT pressure. Raw4 is GripAngle,NOT verified azimuth. Named six-column
headers map by name,not by this seven-column positional mapping. No headerless
six-column interpretation is guessed. The six-column test projects actual recorded
numbers under an explicitly adapted header; it does not misrepresent that header
or its azimuth semantics as verified original data.

**Clock limitation:** README does not declare units. Explicit configured ms→s
factor0.001 is a documented assumption,not an independently established Wacom
clock measurement. No rate is guessed from values. Explicit unit headers must agree
with configuration. Unknown azimuth units remain missing. Source coordinate
calibration is unknown; no invented mm/pixels. These limitations are printed and
saved alongside ingestion counts.

Dataset attribution: Isenkul,M.,Sakar,B.,Kursun,O.(2014),UCI DOI10.24432/C5Q01S,
CC BY4.0 per the dataset citation/license card referenced in fixture README.
Original UCI portal could not be retrieved in this restricted environment;
source time-unit documentation was NOT independently available. Fixture provenance
records commit,source Git/SHA256 hashes,line ranges,fixture SHA256 and modifications.
Four bounded publicly pseudonymized source excerpts are committed as requested;
no private mobile traces or participant names are published. One fixture prepends
the exact documented7column header; numeric rows remain unchanged. The original
source was headerless. Never reuse this fixture subset as a clinical study.

## Measured full real local preparation

Command (with private output paths):

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m scripts.train_v6_gpu_standalone \
  --local-uci-dir /tmp/neurotrace_real_uci/hw_dataset \
  --records /tmp/uci395_runner/records.json \
  --partition /tmp/uci395_runner/partition.json \
  --prepared-records /tmp/uci395_runner/prepared.json \
  --ingestion-report /tmp/uci395_runner/summary.md --prepare-only
```

| Measurement | Actual result |
|---|---:|
| Parsed recordings / source metadata files |77 /1|
| Duplicate timestamp rows removed (LAST retained) |88|
| Remaining nonmonotone segments rejected |14|
| Schema-valid trajectories before feature QC |188|
| Short trajectories rejected by feature QC |6|
| Feature-ready trajectories |182|
| Real HC / PD subjects |15 /62|
| Development / holdout / unlabeled subjects |62 /15 /0|
| Saved finite DWT19 feature tensor |**[2588,19,512]**|

All77parents remain. Subject-stratified partition uses seed42 and approximately20%
holdout per class; all records have exactly one partition membership. Distinct full
codes such as P_26060001/P_26060002 and H_P000-0001/H_P000-0002 are NOT collapsed.
Explicit task suffixes preserve the parent; no V4 numeric-block truncation.
The audited V5 suffix-only normalizer is reused without executing its Colab script
or editing V5. TestID boundaries remain separate trajectories,not concatenated.

## Fresh tests and notebook changes

`OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q` → **41 passed**,one existing
GroupNorm ONNX TracerWarning. 11new tests use recorded fixtures/provenance and
exercise column order,header handling,explicit conversion,duplicate-last policy,
clock reversal rejection,nonfinite removal,full-code identity/task grouping,
stratified disjoint coverage,schema roundtrip,and real local CLI prepare-only with
MISSING records+partition. Rerun preserves source records; invalid existing records
report their indices and are not replaced. Legacy tests include synthetic
engineering inputs; we do NOT call all41 tests clinical or real-recorded cases.

Notebook now installs from the GPU requirements file,streams subprocess stdout AND
stderr live,creates records/partition from local/discovered/downloaded real UCI,
prints conversion/QC/counts before training,and computes actual feature cache.
Default smoke is2epochs/fold,SSL0,full model. Baseline skips pseudo-PD generation;
explicit opt-in still enforces real parents and fold guards. Epoch logs stream.
Member export/parity and locked holdout evaluation/subject-primary CSV follow.
No holdout is used for SSL,donors,model selection or calibration. Raw ensemble
threshold stays0.50; member isotonic is not silently applied to the ensemble.

## Additional limited REAL-recorded CPU execution check — not GPU acceptance

After successful full-cohort preparation,a deliberately limited **full15,779,536
parameter** member was actually trained for2epochs on8real development parents:
4windows/parent (32windows total),first eligible trial per parent. Validation used
4different development parents/35windows for selection. No artificial subjects,
random checkpoint substitution,toy architecture,SSL or clinical-score claim.
Both losses were finite. This is a bounded execution diagnostic,NOT full-data
training or a substitute for the demanded GPU sequence.

The genuinely updated member passed dynamic ONNX parity on100tensors:
**50random engineering tensors +50actual recorded member-validation tensors**,
maxabsolute logit error **7.748603820800781e-7**,below1e-5. The guarded
single-member10TTA inference path (NOT fivefold50TTA) predicted the locked15real
holdout parents,and the subject-primary aggregate metrics CSV was written
(50bootstrap replicates for this bounded diagnostic,not the default1000).
Calibration/statistics used only permitted development parents;holdout was never
used for selection/calibration/threshold search. No disease-performance numbers
are promoted from this tiny training diagnostic. All weights,predictions,raw
records and fitted maps remain private under `/tmp/uci395_cpu_diagnostic`.

A safe aggregate execution summary (no participant IDs,predictions or weights) is
`reports/v6_uci395_cpu_execution_summary.json`. Its ONNX hash identifies only this
private CPU diagnostic,NOT a model release. Elapsed time is not a GPU benchmark.
The capped32-window one-member diagnostic does NOT meet full fivefold/GPU acceptance.

## GPU acceptance and unresolved science

Actual environment: Python3.11,Torch2.5.1+cu124,CUDA unavailable,device_count0,
`nvidia-smi` absent. No fresh Colab/Kaggle GPU runtime can be launched from here.
Therefore required GPU install→real prepare→2–3epochs→trained ONNX→locked CSV is
**NOT VERIFIED**. CPU execution and notebook syntax cannot substitute for it.
Exact external commands are in `docs/v6_gpu_execution_guide.md`.

No real mobile PD cohort: mobile/touch PD sensitivity is **UNMEASURABLE**. Wacom-only
preparation/evaluation is not mobile validation. Healthy-phone high-PD web behavior
remains an unresolved domain gap; this notebook does NOT fix it. No clinical
threshold,calibration,SOTA or diagnostic readiness claim is made. Required real
HC touch/mouse for recorded-source pseudo-PD generation is still absent.

## Final regression / immutable production artifacts

Fresh full suite:41passed,onelegacy ONNX tracer warning. Read-only V5 JS/Python
oracle regression:6synthetic cases,25windows,maxabsolute difference0 (using the
installed scientific Python via `PYTHON=/home/user/.venv/bin/python`). These remain
engineering checks,not disease performance. `git diff --check` passed.

V5 SHA256 remains exactly:

- `neurotrace_v5.onnx`: `a672a165cfb24bcd62c54a72a3bd13c1595632d0e2a69116914200d2d09b8d22`
- `scaler_config_v5.json`: `ce7f992a175a8d8a3ca2c7639946ef7efc79724dc5c8722555679694e8e7f42e`

Changed implementation:dedicatedUCI395loader + compatibility wrappers,GPUrunner,
notebook,streamed training epoch logs,and subject-primary evaluation CSV writer.
Recordedfixtures/provenance,11newtests,thisreport,executionguide,correctness/history
notes,README and BLOCKERS describe the measured boundary. No raw full cohort,
private identifiers,credentials,model weights or prediction files are committed.
