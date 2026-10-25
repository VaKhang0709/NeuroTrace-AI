# V6 real UCI395 / Colab / Kaggle execution guide

**2026-10-25 — loader and real-data preparation verified; GPU acceptance BLOCKED.**
Work only on `arena/4edc99ce-neurotrace-ai`. Keep PR #4 OPEN for Tech Lead.
No V5 changes, production threshold changes, clinical validation or SOTA claims.

## Confirmed source format and root cause

The pinned public mirror's original README and actual `P_26060001.txt` agree:

| Raw seven-column field | Index | Canonical field/index |
|---|---:|---|
| X | 0 | x / 0 |
| Y | 1 | y / 1 |
| Z | 2 | Not pressure; not used |
| Pressure | 3 | pressure / 3 |
| GripAngle | 4 | Not verified azimuth; not used |
| Timestamp | 5 | seconds / **2**, after explicit conversion |
| Test ID | 6 | Contiguous protocol segmentation, not a feature |

These originals are **headerless semicolon files**, not the six-column
`X,Y,Pressure,Azimuth,Tilt,Time` hypothesis. The new adapter also accepts that
six-column variant **only with an explicit named header**, including a leading
unnamed/index column. It maps time to canonical index 2 and pressure to index 3.
Do not positionally reinterpret an unlabelled six-column table as original UCI.

The prior seven-column parser already put time at index 2; an index-3 bug in that
code is **not confirmed**. Confirmed boundary defects: missing-record paths were
read before records could be built; a manually created partition was required;
named headers were unsupported; duplicate rows caused silent trajectory loss;
`--prepare-only` did not actually compute features. Merely adding schema flags to
a four-column or wrongly ordered array does not fix these problems.

The original mirror README **does not declare timestamp units**. Default
`--uci-time-unit milliseconds` is an exposed assumption, multiplying raw time by
`0.001` after subtracting the first retained timestamp. It is NOT independent
hardware unit verification or sampling-rate inference. Supported alternatives:
seconds/microseconds. Explicit unit headers must agree with the configured unit.
Raw coordinate calibration is also unknown: `coordinate_units='unknown'`, not an
invented mm/pixel calibration. Native numeric x/y are preserved. GripAngle is NOT
substituted for azimuth. Named Azimuth fields need an explicit known angle unit
(`--uci-azimuth-unit degrees|radians`); unknown units are represented as missing.

Canonical records are `[x,y,seconds,pressure,azimuth_radians]`, with
`schema_version='neurotrace-trajectory-1'`, `time_units='seconds'`. Missing azimuth
and UPDRS serialize to JSON null; schema reconstructs NaN. Unknown age is **-1**,
not the string `'unknown'`. Device/source are Wacom/uci395; clinical labels derive
from recognized research codes with folder conflict checks, never default HC.

## Actual local verification (not GPU or clinical acceptance)

Pinned mirror: `irenemao/Handwriting-Analysis`, commit
`25ca56231bc161b39b4e896f126de89284041b95`. All 78 fetched blob hashes were checked
against Git blob SHA-1; this verifies mirror content, **not** an independent UCI
release checksum. This mirror has 77 recording text files + README; it is not
claimed identical in file/image counts to every Kaggle package.

| Real recorded preparation result | Observed |
|---|---:|
| Parsed recording files / subjects | 77 / 77 |
| HC / PD subjects | 15 / 62 |
| Duplicate rows removed, keep LAST | 88 |
| Segments rejected for remaining clock reversal | 14 |
| Schema-valid trajectories | 188 |
| Feature-short trajectories excluded before partition | 6 |
| Feature-ready trajectories / actual DWT19 windows | 182 / **2,588** |
| Development / locked holdout / unlabeled subjects | 62 / 15 / 0 |

The raw loader never resamples. It removes nonfinite source rows, preserves last
rows at repeated timestamps in acquisition order, and rejects remaining clock
reversals; it never sorts a reset away or fabricates time. Feature readiness uses
the existing 100-Hz/512-step feature pipeline, **without stretching short trials**.
All 77 parents remain after QC. New partitions stratify **subjects**, with seed 42;
all records' parents occur in exactly one disjoint set. Existing locked partitions
are checked and preserved, not silently regenerated after seeing scores.

The committed bounded public excerpts in `tests/fixtures/uci395/` come from four
DIFFERENT recorded research codes. Attribution, hashes, exact source line ranges
and the one prepended documented-header adaptation are recorded there. They are
not a training cohort or invented participants. Full raw downloads stay outside
Git. The six-column header adapter test is explicitly a mechanical projection,
not evidence that the original README contained such a header or azimuth sensor.

## Fresh Colab / Kaggle run

Open `notebooks/v6_gpu_beast_training.ipynb`. Select an A100/T4 GPU runtime and
Internet. P100 requires a driver/CUDA stack compatible with pinned Torch 2.5.1;
its name alone does not establish compatibility. Use fresh supported Python
(3.10–3.12; sandbox verification used 3.11), installing before importing Torch.

The notebook clones/fast-forwards ONLY this branch, prints the exact commit,
installs `requirements-v6-gpu.txt`, checks CUDA, discovers a conventional unique
Kaggle `control/` + `parkinson/` folder or downloads the official UCI zip. To use
an arbitrary mounted directory/authorized zip, set `LOCAL_UCI` in the config cell.
No existing unified JSON or partition is required. No mocks or Drive authentication
are required for public UCI. A blocked official download needs a real local input;
there is no silent synthetic substitute.

Default `SMOKE=True` means **2 supervised epochs/fold, SSL=0**, full architecture,
five real-data folds. This ends during warmup: it checks execution, not convergence.
All subprocess stdout/stderr and epoch CSV metrics stream live. The notebook runs
prepare → train → trained-member ONNX/parity → locked subject metrics CSV in order.
`SMOKE=False` selects 200/100 supervised/SSL epochs. This is a large research run,
not a promise of accuracy or VRAM fit. Microbatch 2/target accumulation 256 is a
starting point, not measured T4/A100 memory guidance. Keep several GB available
for optimizer state/top-3/last checkpoints and private outputs.

### Exact standalone equivalent (repository root, Bash)

```bash
python -m pip install -r requirements-v6-gpu.txt
python -c "import torch; print(torch.__version__,torch.cuda.is_available()); print(torch.cuda.get_device_properties(0))"

ROOT=data/processed/v6_uci395
COMMON=(python -u -m scripts.train_v6_gpu_standalone
  --records "$ROOT/v6_unified_records.json" --partition "$ROOT/partition.json"
  --prepared-records "$ROOT/v6_prepared_records.json"
  --output models/v6/smoke_checkpoints --feature-contract v6-dwt19-v1
  --uci-time-unit milliseconds --epochs 2 --ssl-epochs 0 --batch-size 2)
# OPTIONAL: set UCI_DIR to the real mounted/extracted folder before these commands.
# Colab example: export UCI_DIR=/content/authorized_uci/hw_dataset
# Kaggle example: export UCI_DIR=/kaggle/input/your-uci-dataset/hw_dataset
if [ -n "${UCI_DIR:-}" ]; then COMMON+=(--local-uci-dir "$UCI_DIR"); fi
"${COMMON[@]}" --prepare-only
"${COMMON[@]}"

CHECKPOINTS=(models/v6/smoke_checkpoints/fold_{0..4}_best.pth)
python -u -m scripts.export_v6_onnx_pipeline \
  --checkpoints "${CHECKPOINTS[@]}" --records "$ROOT/v6_prepared_records.json" \
  --partition "$ROOT/partition.json" --output models/v6/smoke_export \
  --threshold-policy fixed_0.5
python -u -m src.inference.v6_ensemble_inference \
  --checkpoints "${CHECKPOINTS[@]}" --records "$ROOT/v6_prepared_records.json" \
  --partition "$ROOT/partition.json" --output reports/private/v6_smoke_predictions.npz
python -u -m src.evaluation.v6_eval \
  --predictions reports/private/v6_smoke_predictions.npz \
  --partition "$ROOT/partition.json" --output reports/private/v6_smoke_holdout.md
```

With no local flag the runner attempts the official source. For an authorized zip,
use `--local-uci /secure/uci.zip` instead. A supplied missing local path is an error,
not permission to download into or overwrite your source location. Extracted local
inputs are read directly and never copied/modified. Acquisition reports record
network failure. Nonstandard filenames require `--subject-map` CSV fields
`relative_path,subject_id,label`, reviewed by the data owner.

Expected outputs: versioned raw-canonical records, locked partition,
`v6_prepared_records.json`, actual `v6_prepared_records.features.npz` `[N,19,512]`,
`v6_unified_records.ingestion.json` audit counters, five checkpoints/epoch CSVs,
`neurotrace_v6_tremorformer.onnx` + `.parity.json`, `scaler_config_v6.json`, private
holdout predictions, `v6_smoke_holdout.md` + **`v6_smoke_holdout.csv`**. CSV metrics
are computed on subject-mean probabilities; they are not window-level clinical
claims or public patient-level prediction rows. All raw/processed/caches/weights/
private reports remain ignored; preserve privately before closing the runtime.

Existing record files validate EVERY entry through `Trajectory.from_dict`; invalid
indices/reasons are printed and no mock replacement or silent column repair occurs.
Feature failures after a locked partition abort rather than exclude holdout parents.
Review raw QC in a new prespecified run, not after looking at test performance.

## Calibration, augmentation and interpretation limits

- Default real-only baseline **does not generate** pseudo-PD. `--use-synthetic`
  explicitly requests the prior experimental ablation and needs real HC touch/mouse
  plus real fold-permitted PD donors. Both donor and recipient must be in training;
  holdout/validation/SSL guards remain. No raw synthetic variants enter validation.
- DWT19 `v6-dwt19-v1` differs from `v6-bands19-v1` despite equal channel count.
  Export and inference enforce the checkpoint contract. Raw logits out; sigmoid
  only for inference. ONNX opset 17/dynamo=False/eval/dynamic batch and time,
  100 parity tensors across lengths 256/384/512/768, absolute tolerance <1e-5.
- Member selection and isotonic calibration use member-validation subjects only.
  Member calibration is NOT silently applied to the fivefold ensemble. The locked
  ensemble report uses raw mean probabilities at 0.50. Global train-only mean/std
  in the scaler JSON are diagnostics, not a second input-normalization transform.
- **Wacom-only UCI does not validate mobile/touch PD. Mobile PD sensitivity remains
  UNMEASURABLE without real independent mobile PD. Healthy-phone high-PD web
  behavior remains an unresolved domain gap. This notebook does not fix it.**
- No clinical/SOTA, fairness, calibration, uncertainty or production readiness claim.
  V5 and its 0.50 threshold remain immutable. Do not promote smoke weights.

## Acceptance checklist / blockers

Fresh full suite: `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q`:
**41 passed**, including 11 new recorded-fixture tests, one legacy export tracer
warning. Notebook schema/Python syntax and output sanitization pass. Full 77-subject
real local preparation succeeded; raw source snapshots and generated tensors were
kept under `/tmp` for this verification.

A separate deliberately bounded CPU diagnostic actually trained the full15.78M
architecture for2epochs on8real development parents (32windows,4windows/parent),
validated on4other parents,exported one trained member (100tensor parity,
50real validation tensors,maxerror7.748603820800781e-7),and wrote a single-member
locked15subject metrics CSV. This is NOT fivefold or full-data/GPU acceptance and
no clinical score is claimed. Safe aggregate evidence is in the loader report;
weights/predictions remain private. It does not change the blocker below.

**Required acceptance still missing:** actual fresh GPU runtime install → real
prepare → 2–3 epoch CUDA smoke → trained member export/parity → locked subject CSV.
Sandbox Torch 2.5.1+cu124 reports CUDA=False, device_count=0, no nvidia-smi. CPU-only
execution cannot satisfy this requirement. See `BLOCKERS.md` and
`reports/v6_uci395_loader_verification.md` for measured evidence and limitations.
