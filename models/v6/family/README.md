# Automatic teacher recovery update — 2026-11-05

The latest explicit instruction authorizes a NEW teacher when the original runtime
files are lost. The notebook no longer fails for missing checkpoint/partition/records.
It automatically downloads real UCI395 (or uses LOCAL_UCI),trains ONE full grouped
PRO member for50epochs (not5x50),then continues NANO40 + BASE40. Batch8 throughout,
synthetic=False,AMP,cached19channel features,no online raw augmentation/SSL in recovery.
No samples/epochs are capped. Existing surviving original partitions are preserved;
if the partition is lost,a new seed42subject-stratified recovery split is clearly
labelled NEW lineage,NOT restoration of the original experiment/holdout.

All recovered inputs + matching NEW ONNX/scaler are saved under
`NEUROTRACE/v6_production/family/teacher_recovery_v1/`:
`fold_0_best.pth`,`partition.json`,`prepared_records.json`,ONNX/scaler,
`teacher_epochs.csv`,`lineage.json`,`recovery_manifest.json`.
`family/teacher_bundle.json` points the remaining cells at that matched bundle.
Original production exports are NEVER overwritten or mixed with new weights.
Hashes gate one-time reuse. If50epochs finish but export is interrupted,the saved
complete checkpoint is reused for export,not retrained. Records/partition are saved
BEFORE training. Large optimizer/top3 writes go to local SSD;only compact final
weights/identity metadata and completion hashes go to Drive. The family directory
is created automatically. No Drive credentials are embedded.

35–40minutes is an unmeasured T4 target. The notebook has a40minute wall-time guard,
replacing the earlier~1CU/student-only allowance;setup/IO/export also consume time.
If the target cannot be met,it stops rather than faking completion;the completed
teacher survives for a later run. No CUDA/T4 training was executed in this sandbox.
Wacom-only training is not clinically established touchscreen PD sensitivity.

Recovery helper and notebook JSON/AST were checked with7filesystem/orchestration
unit tests (no mocked training represented as real training). The earlier57test
family suite belongs to the previous implementation verification;it was NOT rerun
in this refreshed dependency-free sandbox. V5 remains unchanged;PR#4 remains OPEN.

---

# TremorFormer family — DWT19 only

**Research artifacts; not clinically established touchscreen-PD detection.**
PRO is the existing teacher, never retrained. NANO/BASE require genuine completed
40epoch distillation before they are called releases. No random engineering
artifact or CPU smoke is a substitute. All tiers are Wacom-trained; touchscreen
PD sensitivity, mobile calibration and the healthy-phone domain gap remain
UNESTABLISHED. Cascade decisions below are resource/UX heuristics,not medical risk.

| Tier | Parameters | Deployment artifact | Size gate (decimal bytes) |
|---|---:|---|---:|
| NANO |718,404|`neurotrace_v6_nano.onnx`,mixed dynamic INT8 linear weights|≤2,000,000|
| BASE |3,648,400|`neurotrace_v6_base.onnx`,FP16 internals,float32 IO|≤12,000,000|
| PRO |15,779,536|`neurotrace_v6_pro.onnx`,byte-identical FP32 teacher|original63,112,269|

## One frontend contract — do NOT fork the extractor

All use **`v6-dwt19-v1`**, inputs `trajectory: float32[batch,19,sequence_length]`,
`device_id: int64[batch]`,output `logits: float32[batch]`; batch/time are dynamic.
Reference windows512 at100Hz; other lengths are engineering support,not clinical
validation. ONNX input features are ALREADY normalized by the unchanged existing
extractor. Same byte-identical production `scaler_config_v6.json` for all tiers;
per-window zscore epsilon1e-6,no global mean/std pass. No tier-specific features,
scale fitting,pressure reinterpretation or front-end FP16/INT8 input conversion.

Use sigmoid(raw logit),then subject/session window mean per the fixed protocol;
0.50 for holdout reporting. The scaler's isotonic map was fitted for PRO only:
**do NOT apply that map to students**. The shared normalization file remains
unchanged,not refitted. Family raw-score reporting deliberately does NOT apply
teacher isotonic to ANY tier. If your app currently applies that map,review the
postprocessing integration; do not solve it by changing feature extraction.

NANO is mixed dynamic INT8 for constant MatMul/Gemm weights; grouped CNN,
normalization and activation math remain FP32. This avoids unsupported grouped
ConvInteger kernels. BASE keeps external float32 IO. ARM/mobile/web-provider
compatibility and real-device latency MUST be measured separately: desktop ORT CPU
single-thread timing is only a proxy. No claim of all-operator INT8 or GPU speedup.

## Proposed cascade (client policy,not a clinical recommendation)

1. Run NANO locally/offline first. Cache by manifest SHA256/version; validate model
   checksum before loading. Never fetch a model per stroke/window.
2. Escalate to BASE if the raw aggregated NANO probability is **0.30–0.70 inclusive**
   OR this is a **six-task session**. Download BASE once,verify hash,cache privately
   (e.g.IndexedDB/service-worker/native storage); reuse the SAME feature tensor.
3. If BASE remains uncertain or a full PRO review is requested,offer PRO. Server
   upload requires **explicit informed consent**,not merely acceptance of app terms.
   Feature tensors and device/task metadata are health-sensitive too; minimize
   retained data,encrypt transport,allow refusal/deletion. No automatic upload.
   Optional desktop PRO download can keep processing local instead.
4. Without consent/connectivity,retain local results or abstain; do not label a
   person healthy/diseased because escalation failed. Six-task aggregation and
   uncertainty thresholds need a prespecified real-device study before medical use.

## Original inputs — NEVER regenerate the partition

Stage ORIGINAL files privately in Drive `NEUROTRACE/v6_production/`:

- Existing `neurotrace_v6_tremorformer.onnx` + `scaler_config_v6.json` (read-only).
- Original locked `partition.json`,62development/15holdout subjects,not a new seed.
- Original canonical `v6_prepared_records.json` for those exact77parents.
- Actual selected teacher checkpoint (e.g.`fold_0_best.pth`),with original
  `model_config`, `model_state_dict`, `feature_contract`, `training_subjects`,
  `validation_subjects`, `ssl_subjects`. Do NOT reconstruct missing auxiliary weights
  from a pruned ONNX or fabricate the identity lists.

Reuse the teacher's original training/validation groups within development.
Teacher validation identities cannot become student training/calibration examples;
no holdout enters KD,profiling,normalization fitting,selection or quantization parity.
No synthetic inputs; no random new split. ONNX-only KD can use an actual
`teacher_provenance.json` binding teacher AND original partition SHA256 plus the
original train/validation/SSL lists,but final PRO PyTorch parity still needs actual
checkpoint weights. An unverifiable ONNX graph alone is insufficient identity audit.

Connected Drive audit2026-11-05 found only the production ONNX/scaler,not the locked
partition,records or selected checkpoint/provenance. Final training/holdout acceptance
is therefore BLOCKED here. The repository contains no placeholder holdout report.

## Execute on T4

Use `notebooks/v6_family_distill.ipynb`:40epochs EACH,NANO+BASE,batch8,T3,alpha.5,
AdamW3e-4,weight_decay.05,deviceCE weight.1,seed42,AMP,synthetic=False. Frozen eval
ONNX teacher logits are cached ONCE for training parents only. Binary KD is KL on
`[sigmoid(-z/T),sigmoid(z/T)]` timesT² + true-label BCE,not zero-valued one-logit
softmax. Student GRL remains enabled. Validation selection is subject-level AUC.

Save path `/content/drive/MyDrive/NEUROTRACE/v6_production/family/`. The notebook
profiles development-only steps BEFORE the full run,projects all80epochs,requires
the projection to fit a configured~0.9CU wall-time allowance,reserves export time,
and aborts rather than shortening epochs/data or claiming completion. CU/hour is
an OPERATOR-VERIFIED Colab rate,not a constant or measured guarantee. Default2CU/h
is a planning assumption; setup/idle/retries also consume allocation. No T4 is
available in this sandbox; under1CU completion remains unmeasured,not promised.

Standalone (set paths to the original private files):

```bash
python -m pip install -r requirements-v6-family.txt
python -m src.models.tremorformer_family
python -u -m src.training.distill --teacher-onnx "$TEACHER_ONNX" \
  --teacher-metadata "$TEACHER_CHECKPOINT" --records "$RECORDS" \
  --partition "$PARTITION" --scaler "$SCALER" --output "$FAMILY" \
  --epochs 40 --batch-size 8 --temperature 3 --alpha .5 --device cuda \
  --max-seconds 1500 --reserve-seconds 240
python -u -m src.export.export_family --nano "$FAMILY/nano_best.pth" \
  --base "$FAMILY/base_best.pth" --teacher-checkpoint "$TEACHER_CHECKPOINT" \
  --teacher-onnx "$TEACHER_ONNX" --records "$RECORDS" --partition "$PARTITION" \
  --scaler "$SCALER" --output "$FAMILY"
python -u -m src.eval.benchmark_family --family "$FAMILY" \
  --teacher-onnx "$TEACHER_ONNX" --teacher-metadata "$TEACHER_CHECKPOINT" \
  --records "$RECORDS" --partition "$PARTITION" \
  --output reports/private/v6_family_holdout.md
```

Size/hash/100tensor dynamic parity manifest gates:FP32<1e-5,FP16<.01,INT8<.10
absolute raw-logit difference vs PyTorch. Lossy FP16/INT8 are not exact1e-5 parity;
gates are fixed BEFORE seeing holdout,never relaxed to make a trained model pass.
At most50tensors come from real member-validation windows; remaining cases are
random engineering shapes. PRO is copied unchanged and checked vs actual weights.
Scoring on15holdout subjects occurs only AFTER all gates pass. Report includes
real subject ROC-AUC,PR-AUC,sensitivity/specificity@.50 and single-thread CPU512window
latency. It is written only after all three models/all holdout parents finish.

Weights,caches,teacher provenance,predictions and reports remain private/ignored.
Only this README is tracked here. V5 and threshold0.50 are untouched; PR#4 stays open.
