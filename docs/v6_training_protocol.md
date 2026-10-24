# V6 reproducible research protocol

## Status and environment

Engineering scaffold with tested CPU paths; **not clinically trained**. No real
Firebase/UCI/PaHaW/HandPD sources or GPU available. See `BLOCKERS.md` and
`reports/v6_correctness_report.md`. Default pipeline is ready to attempt training
on full authorized data when GPU and reviewed partitions are available; this is
not a claim of convergence, domain robustness or >0.97 AUC.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements-v6.txt
# CUDA-enabled PyTorch must match the operator's GPU/driver environment.
python -m pytest tests/test_v6.py tests/test_v5_modules.py tests/test_v5_input_contract.py
python -m scripts.v6_smoke_test --output /tmp/v6_smoke --full-export
```

Smoke fixtures are explicitly artificial. Tiny width32/oneblock models exercise
all five folds and SSL for one epoch only; full15.78M network is separately checked
for forward/backward and100-tensor random-initialization ONNX parity. Neither
produces trained Parkinson models or publishable performance estimates.

## Acquisition and private input schema

Use `configs/v6_ingestion.example.json` after copying to a private local path and
providing a service account outside Git; SDK auth is NEVER requested in chat.

```bash
python -m scripts.fetch_v6_firebase --config /secure/ingestion.json
```

Firebase contract: each collection document has device_uuid, session_id, optional
stable cross-device subject_id, tasks array. Each task contains task_id and either
an inline `trajectory` or bucket-relative `storage_path` JSON blob. All tasks must
explicitly use `schema_version="neurotrace-trajectory-1"`, `time_units="seconds"`.
Each unified record contains subject_id, session_id, task_id, domain, samples,
label(-1/0/1), optional updrs/age_bucket, source, coordinate_units. Samples are
`[x,y,t_seconds,pressure_or_null,azimuth_radians_or_null]`. Device domains:
wacom/touch/mouse/stylus/unknown. Tasks: spiral/line/free_write; unknown task is
masked. Proposed age bins0..4: <45,45–54,55–64,65–74,≥75; use ONLY documented ages,
otherwise -1. These bins are a research choice, not a medical severity score.

Private manifest links device→session→task→label; per-device rate/duration/point
QC remains beside ignored raw data. Public ingestion report is aggregate-only.
Ingestion drops malformed sessions, not silently labeled HC. Do not use a device
UUID as proof of human identity across phones/devices or shared-device users.

Review all raw acquisition units, stroke boundaries, diagnosis provenance, consent,
license and patient aliases BEFORE conversion. External gated adapters consume
reviewed unified records only, except the original-UCI adapter with documented V5
timestamp heuristic limitations. No unverified native .svc/image loader is claimed.
Create a private JSON list of unified records for training (JSON NaN should be
encoded as null). Never commit datasets, manifests, UUIDs or private patient IDs.

## Lock partitions before any training

Private partition JSON has `development_subjects`, `holdout_subjects`, optional
`unlabeled_subjects`. These sets must be mutually disjoint and labels consistent
per parent. Five GroupKFold splits use development HUMAN parent IDs, not drawings
or sessions. All real, augmented and synthetic descendants stay with their parent.
Unlabeled parents can enter fold SSL only; never fold validation or test. At least
five development subjects are required; every validation fold must have both
clinical labels or training aborts. GroupKFold itself is not diagnosis-stratified.

Holdout is excluded from every fold, SSL, calibration, donor pool and optimizer.
**Do not “collect all data” into shared SSL before folds**: it would invalidate
unseen-subject claims. Pretrain fold-specific backbones with each fold's training
parents plus independently partitioned unlabeled parents; checkpoint provenance
lists actual SSL subjects. OOF predictions may use ONLY the model that excluded
that development parent. Never evaluate a five-member ensemble on development
parents (most members saw each parent). Ensemble clinical evaluation is holdout-only.

```bash
python -m src.training.v6_train --records /secure/unified_records.json \
  --partition /secure/partition.json --config configs/v6_training.json \
  --output models/v6_runs --epochs 200 --ssl-epochs 100 --batch-size 4 --seed 42
```

## Prespecified hyperparameters

| Component | Default and rationale |
|---|---|
| Canonical rate/window/stride |100Hz/512/128: 5.12s tremor context, overlap retained within parent |
| Source filters |15Hz Wacom/stylus,10Hz touch/unknown,12Hz mouse; engineering priors to validate, not optimized test choices |
| Backbone |256 width,6 blocks,8 heads, FF expansion8,dropout.1;15.78M target-scale |
| SSL |16-step patches,50% mask,2-layer decoder,AdamW3e-4/wd.05,100epochs,training-loss plateau10 |
| Supervised loss |uncertainty weighting of active BCE,UPDRS MSE,task CE,age CE,positive device CE through GRL |
| GRL |lambda.1 default; positive CE because GRL already reverses encoder gradient |
| Optimizer |AdamW3e-4,wd.05,clip norm1 |
| Scheduler |5epochs linear warmup,cosine warm restarts T0=25,Tmult=2 |
| AMP / accumulation |CUDA autocast + GradScaler;microbatch4,effective target256; last incomplete group may be smaller |
| Duration |200epochs/fold default (target150–300),early stop patience20 on SUBJECT AUC |
| Checkpoints |top3 validation-AUC + last,CSV + TensorBoard,optimizer/scheduler/AMP state saved |
| Seeds |Python/NumPy/Torch/CUDA42;per-fold42+fold,deterministic algorithms warn-only |
| Inference |ten raw rotations±15°/time offsets±.04s × fivefold;mean per-window probabilities |
| Evaluation |subject-primary mean across windows;1000 subject bootstrap replicates;fixed.50 threshold |

Seeds are set, but floating-point differences across hardware/kernels remain.
Checkpoints are not advertised as bitwise resumable mid-epoch; a complete
DataLoader/augmentation RNG restore CLI is not implemented. `num_workers=0`
avoids worker RNG divergence. The cache stores baseline windows; training
transforms recompute kinematics/wavelets after raw-coordinate changes.

## Augmentation and ablation gates

Each probability is configurable in `configs/v6_training.json`:
DTW-based monotonic anchor warp.2,magnitude.2,jitter.2,slice.2,rotation.3,
elastic.2,pressure dropout.2,device simulation.2,Mixup.2,CutMix.2.
Mixup(alpha.4)/CutMix use SAME-class/domain training tensor donors; auxiliary
UPDRS/task/age losses are masked for mixed rows. Feature-space CutMix/Mixup do
not claim physically derived derivatives at segment boundaries. Raw transformations
are followed by fresh feature extraction. Device simulation uses60Hz downsampling
plus noise before re-extraction and is flagged as an augmentation, not a real touch cohort.

Cross-domain tremor injection defaults OFF. Opt in only with
`--synthetic-injection`: fold-local real Wacom-PD FFT4–6Hz components are added to
fold-local touch/mouse HC at scale.1–.4 relative spatial radius. Recipients retain
parent provenance and are flagged synthetic, with pseudo-PD targets **only inside
this ablation**. They are not clinically labeled PD. Report baseline with injection
OFF and paired experimental ON. No injected sample may enter clinical test,
calibration or a claim of measured mobile-PD sensitivity. Donor+recipient IDs must
both be training parents. Synthetic window counts are logged by epoch; private
checkpoint hyperparameters identify enabled injection. Prespecify augmentation
ablations instead of asserting each method improves small-data generalization.

## Holdout prediction and reporting

```bash
python -m src.inference.v6_ensemble_inference --records /secure/unified_records.json \
  --partition /secure/partition.json --checkpoints /secure/fold0.pth \
  /secure/fold1.pth /secure/fold2.pth /secure/fold3.pth /secure/fold4.pth \
  --output reports/private/v6_predictions.npz
python -m src.evaluation.v6_eval --predictions reports/private/v6_predictions.npz \
  --partition /secure/partition.json --v5-predictions /secure/v5_same_holdout.npz
```

Ensemble checks checkpoint training/validation/SSL provenance and rejects known
parents. Prediction NPZ and embeddings are private. Probe uses concatenated
identity-variant encoder features across five members; HC-only subject-domain
means, grouped5fold logistic regression with train-fold-only feature scaling.
If a probe fold lacks a domain it returns INCONCLUSIVE, not a passing gate.
Clinical CI resampling is by subject; one-class bootstrap AUCs are undefined and
valid-replicate counts are reported. Sensitivity@95%specificity is descriptive ROC
performance, not a test-chosen production threshold. McNemar requires the identical
V5/V6 held-out parent cohort and diagnoses. No paired data means INCONCLUSIVE.

## Export and calibration

```bash
python -m scripts.v6_export_onnx --checkpoints /secure/fold0.pth /secure/fold1.pth \
  /secure/fold2.pth /secure/fold3.pth /secure/fold4.pth \
  --calibration /secure/selected_member_validation_windows.npz --output models/v6_exports
```

Selected member is the highest validation-AUC checkpoint, not selected by test.
Private calibration NPZ: X,domain,label,subject,optional synthetic flags; parents
must be its own validation set and not train/SSL. Isotonic map fits SUBJECT-mean
sigmoid probabilities. It is not a validated calibration of the five-member
ensemble. An independent all-member-unseen calibration cohort is needed for that.
Calibration is V6-only and has NOT been fitted on real data. V5 is untouched.
ONNX uses opset17,dynamo=False,dynamic batch,device_id input,raw logits;100tensor
strict max-error gate<1e-5. PyTorch native-MHA fastpath is disabled only during
legacy tracing and restored afterwards to avoid unsupported fused operators.

Identity mean0/scale1 in V6 JSON describes inputs AFTER per-window normalization,
not invented dataset scaler statistics. Browser integration is not part of this
repository task; five-member deployment must be profiled before release.
