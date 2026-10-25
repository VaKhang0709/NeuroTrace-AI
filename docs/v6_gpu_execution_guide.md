# V6 GPU execution guide — Tech Lead handoff

## Status: workflow implemented; real-data training still BLOCKED

Work is only on `arena/4edc99ce-neurotrace-ai`; keep PR #4 OPEN. Do not merge,
force-push or replace V5. This guide is a research execution plan, **not a claim
of SOTA, calibrated medical risk or ROC-AUC>0.97**. No real V6 weights are released.

| Item | Evidence in this sandbox |
|---|---|
| Official UCI download | Attempted requested direct zip; blocked by TLS/network policy |
| Other mirrors | Fetch/discovery functions implemented; no verified handwriting DOI/file/license supplied |
| Recorded-source10k synthesis | BLOCKED: no real UCI-PD or touch/mouse HC trajectories mounted |
| Generator scale test | **10002 mock-source materialized NPZ trajectories**,5001touch+5001mouse,3formula-source parents |
| GPU training | NOT EXECUTED: CUDA unavailable |
| Local correctness | **30Python tests passed**; V5 JS6cases/25windows still diff0 |
| Full15.78M dynamic-time ONNX |100random tensors, lengths256/384/512/768,batches1/2/4,maxlogit error**1.0728836059570312e-6** |

Mock data stayed under `/tmp`, tagged `MOCK_SOURCE`, and cannot be consumed by the
production runner. This count is neither10002people nor10002real PD diagnoses.
No raw participant trajectories or generated dataset binaries were committed.

## 1. Runtime and dependencies

Open `notebooks/v6_gpu_beast_training.ipynb` in Colab (GPU runtime A100/T4) or
Kaggle (P100,Internet enabled,private dataset attached). The notebook clones the
existing branch if needed and verifies it; it has NO push/merge commands.
Use a fresh Python runtime before installing PyTorch2.5.1. P100 needs a CUDA/driver
stack that still supports compute6.0; compatibility is checked by executing the
GPU code, not inferred from the GPU name alone.

Standalone equivalent, from repository root:

```bash
python -m pip install -r requirements-v6-gpu.txt
python -c "import torch; print(torch.__version__,torch.cuda.is_available()); print(torch.cuda.get_device_properties(0))"
```

The notebook prints GPU name,VRAM,compute capability and Torch version. Start
microbatch2 on T4/P100,accumulate toward256; A100 may support larger batches.
These are conservative starting points, **not measured VRAM/throughput results**.
The six-block15.78M model uses stride4 before attention. Checkpoints with AdamW
state/top3/last/5folds can consume several GB; reserve disk and private backups.
`requirements-v6-gpu-tested.txt` snapshots this CPU test environment, not a promise
of driver compatibility. Notebook installation needs network/package access.

## 2. Authorized inputs and partitioning

Prepare a private unified JSON list with the schema from `src/data_loaders/schema.py`:
explicit second timestamps,raw native coordinates,optional pressure/azimuth,source,
real clinical labels and stable HUMAN subject_id. Each record must be one continuous
stroke/trajectory; do not concatenate pen-up boundaries silently. Missing diagnosis
is-1,not HC. All training trajectories must yield ≥512canonical100Hz samples;
short/stationary/invalid inputs fail loudly and require a documented QC decision
before a partition is locked. No silent holdout exclusion is permitted.

Supply a private partition JSON with development_subjects,holdout_subjects and
optional unlabeled_subjects. Sets must be disjoint. Device UUID is only an identity
fallback; link the same person across devices and datasets BEFORE locking folds.
Five GroupKFold splits use parent IDs. Validation/test/SSL/donor pools never overlap.
Each validation fold needs both diagnoses or training aborts rather than inventing AUC.

The record list must include genuine touch/mouse HC baselines. UCI alone is NOT a
real mobile dataset. All available UCI records may already be in this list; otherwise
the runner uses the downloaded UCI strict parser. Nonstandard filenames need
`--subject-map` CSV with relative_path,subject_id,label. Only published seven-column
semicolon layout is supported automatically; timestamps are explicitly ms→seconds,
nonmonotone times are rejected,grip angle is NOT assumed azimuth. Review raw units,
consent and clinical labels rather than treating a parser as scientific verification.

## 3. Acquisition, mirrors and local fallback

```bash
python -m src.data_ingestion.auto_downloader
# Restricted network fallback (copy, not modification of originals):
python -m src.data_ingestion.auto_downloader --local-uci /secure/uci_395.zip
# Or --local-uci /secure/extracted_raw_directory
```

The exact requested UCI URL is in `src/data_ingestion/auto_downloader.py`. Downloads
are bounded,hashed,atomically completed and safely extracted. A computed UCI hash
is NOT an independently verified official release checksum. Existing local files
are explicitly marked unverified. Network failure produces a BLOCKED report and
CLI exit2; this is intentional, not a fake successful acquisition.

`fetch_open_mirror()` accepts reviewed JSON entries with name,doi,url,sha256,license,
modality=`dynamic_trajectory`. It supports public Zenodo/Figshare/PhysioNet zip file
hosts; `resolve_zenodo_files()` and `resolve_figshare_files()` discover metadata by
explicit record/article ID. API MD5 checksums must not be passed as SHA-256.
PhysioNet requires a reviewed public direct file URL; no credentials/gates are
bypassed. No real mirror DOI was guessed or image modality substituted. A generic
mirror adapter consumes reviewed versioned unified JSON; unknown raw formats need
conversion review, not assumed timestamp/diagnosis mappings.

Use `--mirrors /secure/reviewed_mirrors.json`. Acquisition and parsed aggregate
metadata go to `reports/v6_ingested_datasets_summary.md`; no IDs or raw points are
published. PaHaW/HandPD/mPower access remains subject to prior documented gates.

## 4. Generator and >10000 variants

```bash
python -m src.augmentation.tremor_injector \
  --records /secure/unified_records.json --partition /secure/partition.json \
  --output data/processed/v6_synthetic_10k \
  --manifest data/processed/v6_expanded_10k_manifest.csv --per-domain 5001
```

Defaults choose5001/domain because5000+5000 is exactly10000,not strictly greater.
The manifest and NPZ files are private/ignored. Immutable outputs are not overwritten;
use a new private run directory to change the seed or source pool.

Generator recipe:

1. Real train/development UCI-Wacom PD donor,normalized by95th-percentile radius,
   fourth-order4–8Hz zero-phase Butterworth component. This band is **not pure
   pathology**: healthy/other movement can also contain it.
2. Recorded HC touch/mouse path is smoothed at3Hz. Monotone micro-hesitation time
   warp,amplitude multiplier.3–1.2,slow±10% envelope and≤5% frequency modulation.
   Mirror extension avoids a donor-repeat discontinuity. Modulation can spread
   energy outside the original band; this is a research hypothesis,not a patient model.
3. Simulate60Hz touch or100Hz mouse with small coordinate noise; omit native pressure
   so the pipeline applies its documented proxy. If only one recipient domain exists,
   retargeting is explicit in `baseline_domain` vs `domain`; it is not a recording
   from the missing hardware.
4. Recipient subject_id stays anchored; donor_subject_id is also stored. Per-file
   SHA-256,source sessions,parameters,synthetic flag and unvalidated pseudo-label
   status are written. Synthetic PD is NEVER called a measured diagnosis.

**Two-parent rule:** Grouping only by recipient leaks PD donor information. The lazy
training loader admits a row only when BOTH recipient and donor belong to the
current fold's training parents. All synthetic validation/test/calibration is
forbidden. Invalid donor coverage can leave a fold with fewer/no synthetic rows;
log counts,do not relax the identity constraint to meet an arbitrary sample count.

The production generator refuses missing real donor/HC sources. Only a test command
with `--mock-fixtures` writes formula data,and that CLI restricts outputs to `/tmp`.
Production loaders reject `MOCK_SOURCE`; it is not a workaround for missing access.

## 5. End-to-end preparation and GPU training

```bash
python -m scripts.train_v6_gpu_standalone \
  --records /secure/unified_records.json --partition /secure/partition.json \
  --local-uci /secure/uci_395.zip --prepare-only

# Prespecified baseline (generates variants but does NOT optimize on pseudo labels):
python -m scripts.train_v6_gpu_standalone \
  --records /secure/unified_records.json --partition /secure/partition.json \
  --epochs 200 --ssl-epochs 100 --batch-size 2 \
  --feature-contract v6-dwt19-v1

# Separately named experimental run,do not overwrite the baseline checkpoints:
python -m scripts.train_v6_gpu_standalone \
  --records /secure/unified_records.json --partition /secure/partition.json \
  --epochs 200 --ssl-epochs 100 --batch-size 2 \
  --feature-contract v6-dwt19-v1 --use-synthetic --synthetic-fraction 0.25 \
  --output models/v6_ablation/checkpoints
```

Prepare-only can run on CPU. Actual training fails closed without CUDA. To run a
within-Wacom baseline when synthesis is blocked,use explicit `--skip-generation`
without `--use-synthetic`; never claim a generated10k dataset for that run.
The runner preserves a private prepared-record snapshot at
`data/processed/v6_unified_records.json` for export/replay. No holdout is used by
an optimizer,SSL,calibrator or synthetic recipe. Source records claiming `mock`
provenance are rejected by this GPU runner.

Synthetic use is **opt-in**,not automatically justified by scarcity. A mixed sampler
uses25%synthetic probability by default (cap≤50%),rather than letting10k pseudo-PD
variants overwhelm a small real cohort. Epoch length is anchored to the real pool;
not every recipe must be drawn every epoch. Train-only repeated sampling is not
new independent data. Compare real-only and injection ablations on the same locked
REAL cohort; do not tune recipe strengths against holdout.

Training reuses AMP,AdamW3e-4/wd.05,5epoch warmup,cosine restarts,uncertainty-weighted
BCE/masked auxiliary losses and **positive device CE through GRL**. A negative CE
would reverse twice. Per-fold CSV/TensorBoard record subject ROC-AUC,PR-AUC (average
precision),sensitivity,specificity,ECE at.50 and synthetic-window count. Top3+last
remain in fold folders; selected copies are
`models/v6/checkpoints/fold_{k}_best.pth`. Fold manifests,SSL and donor parent
provenance are checkpointed privately. Do not publish patient IDs in checkpoints.

## 6. Two incompatible19-channel contracts (explicitly versioned)

| Contract | Meaning |
|---|---|
| `v6-bands19-v1` | Previous default:13base channels +4axis tremor/physiology bands +2sensor-observed flags |
| `v6-dwt19-v1` | Requested workflow default:13base +db4 detail magnitudesL1..L5 +V5-inspired proxy pressure |

Both widths are19 but they are **not interchangeable**. Checkpoints carry the
contract; ensemble loading rejects mixed contracts. DWT levels are reconstructed
x/y detail vectors summarized by their magnitude,not five copies of4–8Hz power.
At100Hz ideal dyadic bands areD1≈25–50,D2≈12.5–25,D3≈6.25–12.5,D4≈3.125–6.25,
D5≈1.5625–3.125Hz. All DWT19 channels get per-window population z-score (+1e-6).
The older contract retains two binary flags without z-scoring. Missing-pressure
and proxy channels can be redundant; ablate rather than claiming novelty benefit.

The raw schema is x,y,seconds,pressure,azimuth. Feature derivation is NOT a parser
that guesses physical units from pixels. Cubic100Hz resampling,anti-aliasing,
device filtering,velocity/acceleration/jerk/curvature andwavelets are recomputed
from augmented paths. No dt unit or V5 threshold changes are introduced.

## 7. Dynamic-time ONNX,measured statistics and calibration

```bash
python -m scripts.export_v6_onnx_pipeline \
  --checkpoints models/v6/checkpoints/fold_0_best.pth \
    models/v6/checkpoints/fold_1_best.pth models/v6/checkpoints/fold_2_best.pth \
    models/v6/checkpoints/fold_3_best.pth models/v6/checkpoints/fold_4_best.pth \
  --records data/processed/v6_unified_records.json --partition /secure/partition.json \
  --output models/v6 --threshold-policy fixed_0.5
```

Outputs `models/v6/neurotrace_v6_tremorformer.onnx` and`scaler_config_v6.json`.
Member choice uses validation AUC only. Calibration uses the selected member's
complete REAL validation cohort; training/SSL/test overlap is rejected. Synthetic
or incomplete validation records are forbidden. A tiny mock export is tested but
not released as a model. Need real trained checkpoints for a real-data export.

- Opset17,dynamo=False,dynamic batch/time. Time support required replacing legacy
  MHA constant-length export reshapes with explicit QKV attention that preserves
  parameter names,plus interpolation of learned positions and dynamic sinusoids.
  Full parameter count stays15,779,536. Test lengths256/384/512/768;batches1/2/4.
  Training/reference preprocessing is STILL512steps. Other lengths are engineering
  support,not validated alternative clinical protocols. No promise for all lengths.
- 100tensor parity must have max|Δlogit|<1e-5. Supplied validation tensors are counted
  separately from random tests. No external sigmoid is baked into the ONNX graph.
- JSON mean/scale are MEASURED from the selected member's REAL training windows,
  not invented constants. However,the model was trained with PER-WINDOW z-scores:
  `apply_global_standardization=false` and statistics_role mark these values as
  **diagnostics only**,NOT another `(x-mean)/scale` transform. Zero-std diagnostic
  channels use safe scale1,with unmodified measured_std recorded separately.
- Isotonic map applies AFTER subject-mean sigmoid probabilities,not each logit/window.
  `fixed_0.5` leaves the V6 operating threshold prespecified. Optional
  `validation_youden` chooses a maximum-Youden point on the SAME calibrated member
  validation data; it is prone to optimism and is NOT a clinical “optimal threshold.”
  No holdout labels are used. V5 root threshold stays0.50 in either policy.
- Member calibration is NOT ensemble calibration. Need separate all-member-unseen
  calibration data before making ensemble calibrated-risk claims. No student
  distillation or web application change is claimed by this workflow.

## 8. Tests and known blockers

```bash
python -m pytest tests/test_v6_gpu_workflow.py tests/test_v6.py \
  tests/test_v5_modules.py tests/test_v5_input_contract.py -q
PYTHON="$(command -v python)" node tests/test_js_preprocessor.js
```

Tests cover blocked network,local zip fallback,checksum failure cleanup,zip-slip,
strict raw units,versioned channels,two-parent filtering,mock rejection,synthetic
training mechanics,notebook schema/cell syntax,CUDA fail-closed and dynamic export.
One GroupNorm tracer boolean warning remains; the tested nonempty shapes pass.
The notebook was structurally validated,NOT executed on Colab/Kaggle hardware here.
See `reports/v6_gpu_workflow_verification.md` and `BLOCKERS.md` for evidence boundaries.

Archive weights/reports to authorized private storage after a GPU run. Do not commit
credentials,manifests,raw trajectories or pseudo-label data to the public repo.
Keep PR #4 open for manual Tech Lead review; never auto-merge after notebook success.
