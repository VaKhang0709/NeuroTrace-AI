# V6 blockers — 2026-10-24

Branch: `arena/4edc99ce-neurotrace-ai`; PR #4 must remain open for Tech Lead.
No blockers are bypassed with fabricated data, substitute weights or clinical claims.

| Phase | Implemented/tested | Blocker / uncompleted deliverable | Unblock action |
|---|---|---|---|
|1 Firebase|SDK ingestion, explicit schema, private manifest/QC, safe identifiers|No service account/bucket access; no remote counts or rates measured|Operator provides authorized local credential, reviews schema/consent and executes ingestion|
|2 External sources|Original-UCI adapter, gated unified-schema adapters, access audit|No raw UCI; network TLS/DNS blocks institutional portals; PaHaW/HandPD/NewHandPD native units/license/form unverified; PhysioNet findings unknown; mPower governed|Obtain licensed originals and dictionaries outside sandbox; retain checksums; review real conversion|
|3 Preprocessing|60/100/133/200Hz synthetic tests,db4/derivatives,source anti-alias|No device acquisition validation/short-trial exclusion audit|Collect deidentified real paired trajectories with explicit units/strokes|
|4 Architecture|Full15,779,536param forward/backward; CPU full ONNX|No CUDA; T4/A100 VRAM/throughput unmeasured|Profile target GPU and prespecify microbatch before long run|
|5 SSL|Fold-local MAE test + oneepoch/fold tiny smoke|No real100epoch pretrained backbone|Authorized train/unlabeled partitions plus GPU; never pretrain on val/test|
|6 Augmentations|Raw transforms,train-only donor guards,masked mixing|No real donor validation; injected labels scientifically unvalidated|Keep injection OFF for baseline; train-only opt-in ablation with real independent mobile PD test|
|7 Training|Fivefold tiny synthetic protocol,AMP code,top3/last,CSV/TensorBoard|No clinical150–300epoch/fold weights; no real population splits|Reviewed patient identities,diagnoses,locked holdout,real data and GPU|
|8 Ensemble|10TTA×5models on four synthetic holdout fixtures|No five clinically trained models or uncertainty validation|Finish real training; use only all-member-unseen holdout|
|9 Evaluation|Subject aggregation,bootstrap,plots,HC grouped probe,paired McNemar exercised with synthetic inputs|No real ROC/PR/sensitivity/CI,probe<.70 result or paired V5 predictions|Collect independent real device-balanced labeled holdout; preserve V5 pairing|
|10 Export/calibration|Full random15.78M ONNX100tensor parity passed;member validation provenance guards|No trained V6 ONNX,isotonic map,production scaler JSON or student;browser feasibility unknown|Export real locked checkpoints;calibrate member on own validation and ensemble on separate unseen cohort|
|11 Docs|Architecture/protocol/limitations/results/blockers and README|Publishable clinical results unavailable|Fill only from locked real evaluation,never smoke data|

## Delivery boundaries

Code is ready to **attempt** full-data training when authorized data and GPU are
available; this is not a claim the inaccessible native formats are integrated or
that the model is production-ready. Dataset-specific gated adapters need reviewed
conversion. Exact current institutional application forms/contact addresses are
UNKNOWN; operator verification steps are in `docs/v6_external_datasets.md`.
No pretrained weights or real ONNX/calibration download link can honestly be supplied.
Synthetic/random smoke artifacts remain private under `/tmp`; no false model release.
Default and tested dependencies: `requirements-v6.txt`, `requirements-v6-tested.txt`.

## 2026-10-25 acquisition / >10000 / GPU notebook extension

- The requested official UCI zip was attempted by `auto_downloader.py`;TLS/SSL EOF
  blocked acquisition. No real dynamic trajectories are available locally.
- Open DOI fetchers require reviewed DOI/direct-file/checksum/license metadata;
  none was supplied or guessed. PhysioNet has no confirmed handwriting source here.
- Generator scale was verified on10002**MOCK_SOURCE**trajectories under`/tmp`:
  5001touch+5001mouse,3formula parents. This is NOT10002patients or real mobile PD.
  Production runner rejects mock evidence;real-donor/HC-baseline synthesis is blocked.
- GPU notebook and standalone runner are implemented and structurally/CPU tested,
  but A100/T4/P100 hardware remains unavailable. No full real-data training occurred.
- Requested detail-level channel layout is explicitly versioned`v6-dwt19-v1`;
  earlier checkpoints use`v6-bands19-v1`.Both widths19do not imply compatibility.
- Dynamic-time full15.78M random ONNX parity passed100cases(max1.073e-6);this is not
  a trained V6 model. No real calibrated config or clinical operating threshold exists.
- New guide:`docs/v6_gpu_execution_guide.md`;evidence:
  `reports/v6_gpu_workflow_verification.md`.No new clinical performance claim.
