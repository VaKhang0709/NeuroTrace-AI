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

# TremorFormer family verification — 2026-11-05

**Implementation + engineering checks completed. Requested trained-family acceptance
is BLOCKED,not passed. No holdout performance numbers or release students fabricated.**
Branch `arena/4edc99ce-neurotrace-ai`; keep PR#4 OPEN. No teacher retraining,V5
artifact/threshold changes,main merge,force push or branch deletion.

## Connected Drive evidence (read-only)

Located `NEUROTRACE/v6_production/` via Google Drive connector. Complete directory
listing contained exactly:

| Existing original artifact | Bytes | SHA256 |
|---|---:|---|
|`neurotrace_v6_tremorformer.onnx`|63,112,269|`0593c9c17470f7868f70aa2bc5de6fa51e2dfb27d067bfd15c83648cf28424b4`|
|`scaler_config_v6.json`|3,747|`5122784b0f9b53120d50d1d1eb5bc028fed61cda1ba12e43d3d4c811e01fd392`|

Both downloaded through the connector into ignored `google_drive/`. ONNX hash
matches the scaler's recorded teacher hash. Scaler declares DWT19,per-window
population zscore epsilon1e-6,no extra global standardization. PRO graph loads in
CPU ORT with exact dynamic float32/int64 input/output signature. The scaler records
historical PyTorch parity,but we do NOT call that a newly executed PyTorch parity
check without the actual checkpoint. No original Drive artifact was modified.

No `partition.json`,selected V6 checkpoint,canonical V6 records or teacher identity
provenance was found in production or the relevant named Drive searches. Local
ignored data/model directories also contain none. Do NOT reconstruct a split from
seed42 or guess subject lists from aggregate counts. ONNX pruning removes unused
auxiliary-head parameters and identity metadata;it is not the original checkpoint.
PRO's50epoch training is the operator's supplied context,not a new retraining here.

## Implemented deliverables

- `src/models/tremorformer_family.py`:single configurable subclass of original
  TremorFormer;PRO has exactly original config/state-dict layout. NANO uses four
  depthwise-separable multi-scale CNN stems,2Conformer blocks;BASE4blocks;PRO6blocks.
  Exact parameters:NANO**718,404**,BASE**3,648,400**,PRO**15,779,536**.
- `src/training/distill.py`:frozen eval/no-grad production ONNX teacher,train-only
  target cache;binary two-outcome KL atT3 timesT² plus BCE,alpha.5;student GRL +
  deviceCE .1 retained. No one-logit softmax degeneracy. Default40epochs each,batch8,
  AdamW3e-4/weight_decay.05,AMP,seed42,synthetic=False. Reuses teacher's ORIGINAL
  training/validation groups within locked development,never creates new groups.
- `src/training/family_data.py`:immutable input/hash/identity/scaler guards. Require
  EXACT original62development/15holdout parents,both diagnoses,complete teacher
  provenance,zero validation/holdout exposure to teacher SSL/training or student KD.
  Same existing DWT19 extractor,no frontend feature changes or global refitting.
- `src/export/export_family.py`:dynamic batch/time,`trajectory`,`device_id`→`logits`.
  NANO mixed constant-linear INT8 (CNN remains FP32 for grouped-kernel portability),
  BASE FP16 WITH float32 IO. PRO copied byte-identically,not re-exported/retrained.
  Same scaler copied byte-identically. Incomplete/<40epoch students cannot release.
  Actual checkpoint required for PRO-vs-PyTorch parity;no fake recovered auxiliary
  weights. All size/hash/signature/precision gates must pass for accepted manifest.
- `src/eval/benchmark_family.py`:requires accepted manifest and matching original
  partition/records/teacher/scaler hashes. Computes real subject-mean holdout ROC-AUC,
  PR-AUC,sensitivity/specificity@.50;single-thread hostCPU latency [1,19,512],20warmups,
  100timed calls. Writes `reports/private/v6_family_holdout.md` ONLY when all tiers
  and every original holdout parent succeed. No teacher isotonic applied to students.
- `notebooks/v6_family_distill.ipynb`:fresh T4 workflow,Drive save path exactly
  `/content/drive/MyDrive/NEUROTRACE/v6_production/family/`,40epochs each,batch8,
  live logs,development-only runtime projection and wall-time kill/abort guards.
  Configured~0.9CU allowance at assumed2CU/hour,including setup since first cell,
  export reserve,30% timing margin. Actual Colab rate/idle/retries are not measured;
  under1CU completion is NOT guaranteed or verified. Abort rather than cap data or
  shorten epochs while claiming completion. No target GPU available here.
- `models/v6/family/README.md`:cascade/cache/consent policy,exact commands,staging
  inputs and honest Wacom-only limitations. Only README tracked,not untrained weights.

## What was actually executed — NOT distilled student evidence

The following student graphs use explicitly RANDOM INITIALIZATION to verify
export mechanics. They are not called trained NANO/BASE or supplied for deployment.
All generated student/checkpoint test binaries stayed under `/tmp`. No fake metrics
were written to the required private holdout report path.

| Engineering check | NANO random INT8 | BASE random FP16 | Existing production PRO |
|---|---:|---:|---:|
|Parameters|718,404|3,648,400|15,779,536 architecture|
|Actual ONNX bytes|1,218,714|7,333,240|63,112,269|
|Declared size target|≤2,000,000 PASS for engineering graph|≤12,000,000 PASS for engineering graph|original unchanged|
|ORTCPU load / exact common dynamic signature|PASS|PASS|PASS|
|100tensor compressed-vs-PyTorch maxlogit error|0.007441267371177673|0.000965118408203125|NOT fresh-verified:weights missing|
|Prespecified absolute precision gate|<0.10|<0.01|<1e-5 when actual weights provided|
|Single-thread hostCPU median ms|1.9707195|9.49678|34.4814185|

Student tests also verify FP32 exports below1e-5. Each engineering parity run above
uses91random shape tensors plus9windows from bounded public raw-format test
fixtures,NOT an original member-validation cohort or locked-holdout study. No
clinical labels are used to choose accuracy/performance. These fixture windows do
not establish original split membership. Production export uses only the actual
teacher-validation parents once provenance is supplied. Engineering SHA256s,signatures,
precision details and timing samples are in `reports/v6_family_engineering.json`.

Precision gates are fixed BEFORE holdout. FP16/INT8 are lossy;do not misrepresent
these as1e-5 parity or relax tolerances after seeing holdout. If trained quantized
weights fail,the release fails. Size may change after training,so engineering file
sizes do not certify final release artifacts. HostCPU latency excludes preprocessing,
loading and network;not ARM/mobile/web-provider measurements or a speed guarantee.

Caught/fixed real conversion issues:ORT dynamic Gemm transposition of shared zero
FiLM initializer aliases (materialize constant aliases before quantization);FP16
converter retagging explicit float Cast outputs without updating Cast destination
attributes (align FP16-typed internal casts while preserving float32 external IO).
Both regressions are covered by full-size preset export/load/dynamic parity tests.

## Regression / unchanged production

Full suite:**57passed**,including16new family test cases;7warnings (legacy GroupNorm
trace warning + expected FP16 tiny-number truncation warnings). Command:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/user/.venv/bin/python -m pytest -q
PYTHON=/home/user/.venv/bin/python node tests/test_js_preprocessor.js
```

V5 JS oracle:6synthetic cases,25windows,maxabsolute difference0. V5 artifact hashes
remain unchanged (`a672a165...b8d22` ONNX,`ce7f992a...e7f42e` scaler). Python3.11,
Torch2.5.1+cu124,CUDA=False,0devices. New notebook validates schema/Python syntax and
has no saved outputs. Operator's executed teacher notebook is deliberately preserved;
legacy notebook test now validates its structure/syntax,not deletion of training logs.

## Exact acceptance blockers / unblock files

1. Stage ORIGINAL locked `partition.json` and matching canonical
   `v6_prepared_records.json` in authorized storage. NEVER regenerate the62/15split.
2. Stage actual selected teacher checkpoint with weights/model config and original
   train/validation/SSL identities (or hash-bound original provenance for ONNX KD;
   actual weights still needed for final PRO PyTorch parity). Do not invent metadata.
3. Execute new notebook on T4;admit only if measured80epoch projection fits verified
   billing budget. Finish both40epoch students,then all release gates and original
   real15subject holdout report. No such CUDA run or real report was possible here.

No clinically established touchscreen PD sensitivity. Wacom-trained tiers cannot
resolve the healthy-phone high-PD web-domain gap by compression/distillation alone.
Cascade0.30–0.70/six-task escalation is a resource heuristic;server PRO requires
explicit consent for health-sensitive feature upload. No clinical/SOTA claim or
model promotion is authorized by these engineering results.
