# Current correction — real UCI395 loader, 2026-10-25

**Supersedes the historical source-availability/test-count statements below.**
Real pinned-mirror UCI preparation now passes:77subjects (15HC/62PD),182eligible
trajectories,2588DWT19 windows; fresh full suite **41passed**,11new recorded-fixture
tests. Raw clock milliseconds remains an explicitly disclosed assumption because
the mirrored original README does not declare units. No GPU device is available;
the required fresh GPU sequence is NOT verified. No mobile-PD validation or web
domain-gap fix is claimed. Details and exact reproduction:
`reports/v6_uci395_loader_verification.md`, `docs/v6_gpu_execution_guide.md`.

---

## Historical engineering checks (before real-source loader correction)

# V6 acquisition / synthesis / GPU workflow verification

Date:2026-10-25. Branch:`arena/4edc99ce-neurotrace-ai`. PR#4 remains open.
Engineering verification only; **no clinical training or validation claim**.

## Phase outcomes

| Phase | Implemented | Executed here | Uncompleted/blocked |
|---|---|---|---|
|1 Acquisition|Bounded HTTPS/checksums,safe ZIP,local fallback,reviewed mirror fetchers,strict UCI parser|Official requested UCI URL attempted;mock local archive/parser tested|Real download TLS-blocked;no verified open handwriting DOI/source supplied |
|2 Tremor|4–8Hz extraction,modulated overlay,dual-parent provenance,NPZ+CSV|**10002 MOCK-source trajectories materialized**,5001touch+5001mouse|No real UCI-PD or recorded touch/mouse HC baseline available for recorded-source generation |
|3 GPU notebook|Colab/Kaggle notebook + standalone driver,fold-safe synthetic sampler,subject metrics|Notebook schema/code syntax checked;CPU tiny fivefold smoke and mock synthetic training tested|CUDA unavailable;no A100/T4/P100 run,no200epoch clinical training |
|4 Export|Dynamic batch/time,validation-only isotonic/threshold,measured train-only stats|Full15.78M RANDOM-weight dynamic ONNX,100tensor parity;mock end-to-end config writer tested|No real-trained V6 checkpoint/map/clinical threshold available |
|5 Verification/docs|Regression tests,guide,README,blockers|**30Python tests PASS**,V5 JS regression PASS|No performance/SOTA/domain-robustness result |

## Recorded-source availability

Downloader report: `reports/v6_ingested_datasets_summary.md`. Requested exact URL:
`https://archive.ics.uci.edu/static/public/395/parkinson+disease+spiral+drawings+using+digitized+graphics+tablet.zip`.
Attempt returned urllib TLS/SSL EOF,recorded BLOCKED;CLI exit2 intentionally signals
no acquired dataset. No alternate weights/data were downloaded to conceal this.
Local fallback and bad-checksum handling were validated with mock archives.
Zenodo/Figshare discovery helpers exist;PhysioNet uses reviewed public direct-file
metadata. No DOI,license or dataset count is fabricated. No native unknown-format
mirror loader is represented as already integrated.

## 10002-trajectory scale verification — MOCK ONLY

The generator was executed with explicit `--mock-fixtures` and `/tmp` outputs.

| Measurement | Observed |
|---|---:|
|Manifest rows|10002|
|Materialized NPZ files|10002|
|Touch-target variants|5001|
|Mouse-target variants|5001|
|Total generated points|5601120|
|Compressed NPZ bytes|107114971|
|Independent formula-source parent IDs|3|
|Rows tagged MOCK_SOURCE|10002|
|Real human subjects / real phone-PD recordings used|0|

These are artificial source formulas,not real PD tremor extraction. They prove
materialization/count/provenance mechanics and do NOT satisfy the outstanding
real-data acquisition requirement. Neither data files nor manifests are committed.
Production generator/runner/loader reject declared mock sources;CLI mock outputs
must remain under `/tmp`. Recorded-source10k synthesis remains BLOCKED until
licensed real donors AND baseline HC streams are available.5001per-domain is
intentional:5000+5000 would not meet the strictly “>10000” wording.

## Dynamic ONNX evidence

Default network parameter count unchanged:**15779536**. Full RANDOM initialization,
CPU,float32,opset17,dynamo=False,dynamic batch and sequence. Self-attention now
uses explicit QKV matmul with existing MHA weight names;learned positions interpolate
to token count and sinusoids are generated dynamically.512step contract remains
the reference;variable-length support is not a validated clinical protocol.

-100random tensor cases,batch cycle1/2/4,time lengths256/384/512/768.
-Max absolute raw-logit difference:**1.0728836059570312e-6 <1e-5 (PASS)**.
-ONNX size:62684809bytes.
-Random ONNX SHA-256:`9ef90cb0c80fbbc15d5f8e10127b7691e5442a9ec33d273db3504a262e5c7ec1`.
-Artifact stays in scratch storage,NOT a pretrained/fine-tuned or production release.

Separate mock end-to-end exporter checks train-statistics vectors,complete member
validation cohort,isotonic map,threshold0.50 policy,feature-contract metadata and
ORT input shape`['batch',19,'sequence_length']`.The code also supports explicit
V6-only validation-Youden selection;no real clinical threshold was fitted here.
V5 is never touched. JSON measured mean/scale are diagnostics only because model
normalization is per-window;`apply_global_standardization=false` prevents double
standardization. See the guide before integrating any web reader of this JSON.

## Regression commands and results

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest \
  tests/test_v6_gpu_workflow.py tests/test_v6.py tests/test_v5_modules.py \
  tests/test_v5_input_contract.py -q
PYTHON="$(command -v python)" node tests/test_js_preprocessor.js
```

**30passed**,oneGroupNorm trace boolean warning. Nonempty tested dynamic shapes
pass;there is no blanket guarantee for arbitrary sequence lengths. Node/Python
V5 comparison:6synthetic cases,25windows,maxpacked difference0. No raw identifiable
trajectories are involved in any test. Environment:Python3.11,torch2.5.1+cu124,
NumPy2.4.6,SciPy1.17.1,ORT1.31.0;CUDA unavailable. Package snapshot:
`requirements-v6-gpu-tested.txt`.

The tiny CPU fivefold smoke was rerun after integration (oneSSL and one supervised
epoch/fold,50predictions per artificial heldout trial). Its100tensor static export
maxlogiterror was4.76837158203125e-7. Tests also train one tiny fold with explicitly
permitted MOCK synthetic samples and verify dual-parent provenance,real-only
validation and metric CSV columns. Production runner never enables that permission.

## Preserved V5 artifacts

```text
neurotrace_v5.onnx     a672a165cfb24bcd62c54a72a3bd13c1595632d0e2a69116914200d2d09b8d22
scaler_config_v5.json  ce7f992a175a8d8a3ca2c7639946ef7efc79724dc5c8722555679694e8e7f42e
```

Threshold remains0.50. No V5 artifact deletion/replacement,main merge,force push,
new branch/PR,web app edits or patient-data publication occurred.
