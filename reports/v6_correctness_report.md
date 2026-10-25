# V6 engineering correctness — 2026-10-24

**Synthetic tests only. No clinical training or held-out patient results.**

## Executed checks

| Check | Measured result |
|---|---|
| CUDA availability | False (CPU-only sandbox) |
| Default TremorFormer parameter count | **15,779,536** |
| Default full-width forward + backward | PASS, finite stem gradients |
| Combined V5/V6 Python regression suite | **21 passed**, no skipped tests in final environment |
| V5 Node-vs-Python reference preprocessor | PASS,6 synthetic cases,25 windows,max packed difference0 |
| V6 multi-rate preprocessing | PASS on synthetic60/100/133/200Hz inputs |
| Fold-local augmentation and donor/recipient guards | PASS on synthetic fixtures |
| SSL/calibration forbidden-parent rejection | PASS |
| Tiny fivefold loop | 5folds,1SSL epoch/fold,1supervised epoch/fold |
| Tiny ensemble path | 10TTA×5members=50predictions/trial,4synthetic holdout trials |
| Deliberately device-leaky feature probe | PASS: high-AUC leakage correctly fails <0.70 gate |
| Synthetic evaluator | subject aggregation,20bootstrap replicates,ROC/calibration plot generation exercised |
| Tiny calibration plumbing | two-knot isotonic map on synthetic member-validation fixtures only |
| Tiny ONNX parity | 100tensors,max\|Δlogit\|=**3.5762786865234375e-7** |
| Full-width RANDOM-INITIALIZATION ONNX parity | 100synthetic tensors,max\|Δlogit\|=**4.470348358154297e-7** (<1e-5) |

Tiny test config: width32,oneblock,4heads,FFexpansion2. This is a correctness
harness, not the delivered default15.78M architecture or a clinical model.
Tiny parity includes6 supplied synthetic validation windows and94 random tensors;
“supplied” does not mean real human data. Full-model parity uses100random tensors.
Batch sizes1,2,4 and five device IDs are exercised. Export sets opset17,dynamo=False,
dynamic batch,raw logits and eval mode. Legacy trace disables MHA fastpath to avoid
unsupported fused `_native_multi_head_attention`, restoring it afterwards.

## Artifacts and reproducibility

Run `python -m scripts.v6_smoke_test --output /tmp/v6_smoke --full-export`.
It writes synthetic-only weights,plots,CSVs,TensorBoard records,ONNXs and summary
JSON under the specified private/scratch output path. Real V6 calibration config
requires approved member-validation inputs via `scripts/v6_export_onnx.py` and
is NOT provided as a fitted production map. No smoke weights are committed or
advertised for deployment, even if their files fit under100MB.

Recorded parity artifact hashes (random/smoke files, NOT releases):

```text
full RANDOM ONNX  9a89102307397badacc7e326493ce70c87a069e4d2bbb42acadd9bce7a5ac2d7
tiny SMOKE ONNX   7834e08374583c03e6875ea2b487c3b7f6b1c15a3111770b783be435e50e19ac
```

Environment: Python3.11,PyTorch2.5.1+cu124 (CUDA not available),NumPy2.4.6,
SciPy1.17.1,ONNX Runtime1.31.0. Full tested package snapshot is
`requirements-v6-tested.txt`; GPU installation must match the operator's platform.
No institutional datasets were downloaded. Firebase ingestion was not executed
against a live service. Metadata/access-gated adapters remain unverified on natives.

Final commands:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest \
  tests/test_v6.py tests/test_v5_modules.py tests/test_v5_input_contract.py -q
PYTHON="$(command -v python)" node tests/test_js_preprocessor.js
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m scripts.v6_smoke_test \
  --output /tmp/v6_smoke --full-export
```

The CLI export/config writer was also exercised on tiny synthetic checkpoints;
its V6 JSON retained threshold0.50 and19channels. That smoke calibration config
remains private and is not a fitted real-data production artifact.

## V5 preservation

Unchanged root production artifact SHA-256 values:

```text
neurotrace_v5.onnx      a672a165cfb24bcd62c54a72a3bd13c1595632d0e2a69116914200d2d09b8d22
scaler_config_v5.json   ce7f992a175a8d8a3ca2c7639946ef7efc79724dc5c8722555679694e8e7f42e
```

V5 threshold remains0.50. No raw subject trajectories,credentials,patient identifiers,
trained V6 clinical weights or clinical performance numbers are published.
