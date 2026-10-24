# V5 production domain-shift diagnostic report

Date: 2026-10-24. Branch: `arena/4edc99ce-neurotrace-ai`. PR #4 remains open.
Engineering diagnostics only: **not clinical validation**.

## Decision summary

**INCONCLUSIVE: phone/touch HC systematic shift toward PD relative to Wacom HC
cannot be determined.** There are no locally available real trajectory cohorts,
clinical labels, subject identities or held-out split manifests to compare.
The reported healthy-phone ~0.9 score and mouse instability are user observations,
not independently measured recordings in this run. They are not counted as data.

Production artifact prerequisites now **PASS** using the uploaded root files.
Actual production ONNX inference was performed on existing synthetic fixture paths.
Those paths score high but have **unknown domain and unknown clinical label**;
they cannot be called healthy, phone, mouse or Wacom observations.

**Single recommended next experiment: paired, consented multi-device HC data
collection**, with frozen production weights/threshold and retained subject grouping.
Do not implement calibration or retraining on the present evidence.

## 1. VERIFIED artifacts and runtime

| Loaded path | Bytes | SHA-256 first 12 |
|---|---:|---|
| `neurotrace_v5.onnx` (repository root) | 279329 | `a672a165cfb2` |
| `scaler_config_v5.json` (repository root) | 1081 | `ce7f992a175a` |

Full SHA-256 values (verified unchanged after inference):

```text
ONNX   a672a165cfb24bcd62c54a72a3bd13c1595632d0e2a69116914200d2d09b8d22
scaler ce7f992a175a8d8a3ca2c7639946ef7efc79724dc5c8722555679694e8e7f42e
```

No substitute weights were downloaded; no `models/` copies were made. Existing
root paths were used directly. This resolves the earlier path-only prerequisite
failure and supersedes the older input-contract audit's artifact-unavailable finding
for these two files only. Checkpoint and real dataset availability remain unresolved.

ONNX checker passed. ORT CPU inspection:

| Property | VERIFIED value |
|---|---|
| Input name / dtype | `input` / `tensor(float)` (float32) |
| Input shape | `['batch_size', 5, 256]` |
| Output name / dtype | `logits` / `tensor(float)` |
| Output shape | `['batch_size']` |
| ONNX opset | 17 |
| Output tail | `LayerNormalization → Gemm → Squeeze` (constant axis node omitted) |
| Sigmoid nodes | 0 |
| Probability conversion | `scipy.special.expit(raw_logit)`, once, outside ONNX |

Runtime: ONNX Runtime 1.31.0 (CPUExecutionProvider), NumPy 2.4.6, SciPy 1.17.1.
Inference ran one window per call. No PyTorch checkpoint parity was performed.

## 2. VERIFIED scaler metadata

The following fields are quoted from root `scaler_config_v5.json`, not inferred or
refitted. **Production classification_threshold remains 0.5 (0.50).**

```json
{
  "model_version": "neurotrace_v5_production",
  "promoted_from_tag": "expD_v5_production",
  "in_channels": 5,
  "seq_len": 256,
  "feature_names": [
    "x",
    "y",
    "pressure",
    "velocity",
    "dt"
  ],
  "mean": [
    -7.64641954140427e-11,
    2.766540157772983e-11,
    -3.906035821776082e-12,
    1.1061299610624562,
    0.009553401427134666
  ],
  "scale": [
    0.7218062399419907,
    0.6920951899605052,
    0.9980028036499135,
    1.6854436521895446,
    0.24163805754977769
  ],
  "classification_threshold": 0.5,
  "output_activation_in_onnx": "raw_logits_apply_sigmoid_at_inference"
}
```

JSON `decision_gate` metrics are stored historical metadata, not revalidated by
this diagnostic. No operating threshold or classifier bias was adjusted.

## 3. Evaluation-set inventory and protocol

Searched repository files plus accessible `/home/user`, `/content`, `/tmp` for
trajectory JSON, CSV/TSV/DAT, NPZ and dataset directories. No real trajectory data,
recorded browser paths, production checkpoint or held-out parent lists were found.
Only scaler/audit JSON files and synthetic test definitions were usable; runtime
package files were excluded from the data inventory. No remote datasets fetched.

| Requested source | Available real trials / subjects | Status |
|---|---:|---|
| Held-out mobile/touch HC | 0 / 0 | INCONCLUSIVE: data and split manifest absent |
| Held-out Wacom/UCI HC | 0 / 0 | INCONCLUSIVE: data and split manifest absent |
| Held-out Wacom/UCI PD | 0 / 0 | INCONCLUSIVE: data and split manifest absent |
| Recorded browser phone/mouse trajectories | 0 / 0 | INCONCLUSIVE: recordings absent |
| Existing synthetic fixture paths | 6 / not applicable | Executed; domain/label unknown |

Synthetic paths reuse `trajectory()` and six case specifications from
`tests/test_js_preprocessor.js`. They are generated formulas, not human recordings.
All diagnostic cases use **production-default** window=256, stride=64; specifically,
fixture 5's test-only window=128/stride=53 override is NOT used. Thus this run has
22 windows, not the 25 from the earlier preprocessing comparison. No subject_id
exists for synthetic paths; a fixture is a trial identifier, not a participant.

Preprocessing uses `V5Preprocessor` from `src/data/preprocessor.py:L14-148`:
approximately 100 Hz interpolation → zero-phase fourth-order 12 Hz Butterworth
filter → trial-normalized velocity with fixed 0.01 derivative interval → pressure
filter or proxy → overlapping windows / 256 interpolation → window x/y RMS,
pressure mean/std and relative-median dt normalization → production scaler →
float32 `[1,5,256]`. No bounding-box scaling, native-unit velocity substitution,
per-drawing scaler fitting, or exact-100Hz grid correction was introduced.
Synthetic timestamps are already strictly increasing seconds; no heuristic raw
unit conversion is required. Window probabilities are sigmoid(logit); trial score
is the arithmetic mean of its window probabilities.

## 4. VERIFIED score distributions

Quantiles use NumPy percentile (linear interpolation). Trial-weighted rows give
each synthetic path equal weight; pooled-window rows give longer paths more weight.
All values are probabilities, not percentages. Missing cohorts have no scores.

| Domain | Clinical label | Unit | N | Mean | Median | p10 | p90 |
|---|---|---|---:|---:|---:|---:|---:|
| mobile_touch | HC | real trial | 0 | N/A | N/A | N/A | N/A |
| wacom | HC | real trial | 0 | N/A | N/A | N/A | N/A |
| wacom | PD | real trial | 0 | N/A | N/A | N/A | N/A |
| mouse | unknown | real trial | 0 | N/A | N/A | N/A | N/A |
| unknown | unknown | synthetic trial | 6 | 0.807300 | 0.806029 | 0.765573 | 0.850298 |
| unknown | unknown | synthetic window | 22 | 0.831666 | 0.837913 | 0.735444 | 0.922930 |

All six synthetic trial means exceed 0.50. This establishes only that these
particular deterministic paths elicit high scores under the production model;
it is **not a false-positive rate**, since there is no clinical label.

### Trial-level measurements and pressure ablations

All pressure variants retain exactly the same x/y/t path and window set. Baseline
“varying” means formula-generated pressure, **not a native measured sensor**.
“Missing” uses the V5 proxy; “constant” also falls back to proxy because the Python
implementation requires at least four distinct pressure values rounded to 4 dp.
Constant raw pressure is therefore **not** a forced constant feature-channel test.
We do not bypass that behavior, since doing so would change the input contract.

| Synthetic fixture | Points | Baseline pressure | Windows | Baseline trial mean | Constant trial mean | Proxy trial mean | Constant − baseline | Proxy − baseline |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 41 | missing → proxy | 1 | 0.800640 | 0.800640 | 0.800640 | +0.000000 | +0.000000 |
| 2 | 190 | synthetic varying | 1 | 0.733382 | 0.659175 | 0.659175 | -0.074208 | -0.074208 |
| 3 | 650 | missing → proxy | 9 | 0.869717 | 0.869717 | 0.869717 | +0.000000 | +0.000000 |
| 4 | 510 | constant → proxy | 5 | 0.797763 | 0.797763 | 0.797763 | +0.000000 | +0.000000 |
| 5 | 400 | synthetic varying | 4 | 0.830879 | 0.802945 | 0.802945 | -0.027934 | -0.027934 |
| 6 | 320 | synthetic varying | 2 | 0.811417 | 0.771529 | 0.771529 | -0.039888 | -0.039888 |

| Domain | Trial-delta summary (constant or proxy − baseline) | N | Mean | Median | p10 | p90 |
|---|---|---:|---:|---:|---:|---:|
| unknown (synthetic only) | Both ablations identical | 6 | -0.023672 | -0.013967 | -0.057048 | +0.000000 |
| mobile_touch / wacom / mouse | Real-domain pressure deltas unavailable | 0 | N/A | N/A | N/A | N/A |

Only three fixtures have varying baseline pressure, so the all-six mean includes
three zero-delta controls. These are paired engineering effects, not a domain-level
clinical estimate. No real/native phone, pen or mouse pressure was available.

### Per-window baseline scores (synthetic, no identifying data)

- Fixture 1: 0.800640.
- Fixture 2: 0.733382.
- Fixture 3: 0.723618, 0.856908, 0.823568, 0.848294, 0.907262, 0.884899, 0.933463, 0.924671, 0.924774.
- Fixture 4: 0.692087, 0.817771, 0.768649, 0.824852, 0.885454.
- Fixture 5: 0.753995, 0.853975, 0.827532, 0.888015.
- Fixture 6: 0.764157, 0.858677.

## 5. Aggregation match / mismatch

**VERIFIED source protocol:**
`src/ai_model/scripts/train_pipeline_v5_diagnostic.py:L994-1001` applies sigmoid
per window. `L966-972` averages probabilities across **all kept windows of each
parent subject** and thresholds the mean. Dataset capping can select a subset of
windows (`L726-753`), so production metric aggregation is not necessarily an
uncapped single trial. Multiple trials per parent are pooled by window, not by an
equal-weight mean of trial means. Real clinical metrics are subject-level.

**Conditional match:** mean(window probabilities) matches within a single trial
or a parent represented by only that trial with the same retained windows. With
one window the mean equals its probability. **Mismatch:** retaining only the first
window of a multi-window trial, sigmoid(mean logits), equal-weight averaging of
trials with different window counts, or claiming a single new drawing reproduces
the multi-trial capped subject evaluation.

| Synthetic fixture | First-window probability | Mean window probability | Sigmoid(mean logit) |
|---|---:|---:|---:|
| 1 | 0.800640 | 0.800640 | 0.800640 |
| 2 | 0.733382 | 0.733382 | 0.733382 |
| 3 | 0.723618 | 0.869717 | 0.880708 |
| 4 | 0.692087 | 0.797763 | 0.805546 |
| 5 | 0.753995 | 0.830879 | 0.836341 |
| 6 | 0.764157 | 0.811417 | 0.816075 |

**Actual deployed web aggregation: INCONCLUSIVE.** This repository's frontend is
placeholder directories, and `src/export/v5_preprocessor.js` returns windows but
performs no inference/aggregation. The supplied statement that the web app uses a
mean is not a browser execution trace. No web code was modified. To confirm the
live app, privately capture retained window count and per-window probabilities
alongside its displayed aggregate and compare under identical weights/scaler.

## 6. Interpretation, confounding and next experiment

**Phone HC systematically shifted relative to Wacom HC? INCONCLUSIVE**, not “no.”
Both comparison cohorts have N=0 here. Synthetic high scores cannot identify a
sensor cause, quantify specificity or explain an individual phone result.
Proxy substitution reduced the three varying-pressure synthetic scores but left
all six synthetic trial means above 0.50. This supports pressure sensitivity of
these inputs only, not a verified explanation for the healthy phone observation.

Historical cohort descriptions say mobile training data were HC-only and PD came
from Wacom; real counts cannot be rechecked without data/manifests. **If mobile
examples contain only HC, mobile PD sensitivity is unmeasurable**, regardless of
HC device-probe AUC. There are no real phone-PD patient recordings available in this
workspace. Stored leakage AUC 0.67904 does not establish mobile clinical validity.

### Single recommended next experiment: paired multi-device HC collection

Collect consented, deidentified repeated spiral trials from the **same held-out HC
participants** on phone/touch, mouse and Wacom, with acquisition order balanced.
A small prespecified engineering pilot (for example 20 participants × 3 domains ×
3 repeats) would distinguish within-person device effects from between-person
variation; it would still not constitute clinical validation. Retain authorized
raw timestamps/units, canvas/DPR geometry, pointer type, pressure availability,
stroke boundaries and private parent IDs. Keep trajectories private. Freeze these
exact artifact hashes and threshold 0.50; run audited preprocessing, native versus
constant/proxy paired ablations, window counts, subject-pooled probabilities and
within-participant device deltas with uncertainty. Compare browser and offline
scores on the same retained windows to isolate transport/aggregation issues.

Calibration is a later option only after representative independently labeled
validation data establish a repeatable score shift; no calibration is fitted now.
Retraining is a later option if acquisition/coverage deficiencies persist, requiring
properly labeled multi-device PD and HC cohorts with parent-disjoint evaluation.
Neither is justified by this synthetic-only diagnostic or a single healthy user.

## 7. Limitations and preservation checks

- No real held-out clinical cohort, source counts, parent mapping or raw-browser
  replay; no measured mobile-versus-Wacom contrast or phone-PD sensitivity.
- Six correlated formula paths are not independent people or representative devices.
- Existing source tests support limited numerical compatibility, not universal
  browser equivalence or accuracy. Python preprocessing was used for this inference.
- ONNX/checker and inference succeeded, but no checkpoint/PyTorch parity was possible.
- Per-window overlap, historical cap selection and subject aggregation affect
  comparisons. No confidence interval is meaningful for absent real cohorts.
- Only this Markdown report is added. No model/scaler/web changes, calibration,
  threshold change, weight updates, new branch/PR, merge or raw-data publication.

## 8. Reproduction (synthetic engineering measurements)

Install `numpy==2.4.6 scipy==1.17.1 onnxruntime==1.31.0 onnx` in an isolated
Python environment. Run the following from repository root. It reads the actual
root artifacts, uses the existing six fixture formulas, prints per-window results
and summaries, and writes only scratch JSON under `/tmp`, not participant data.
The environment-specific path to Python is left to the runner.

```python
import sys,json,hashlib
from pathlib import Path
import numpy as np
import scipy,onnx,onnxruntime as ort
from scipy.special import expit
sys.path.insert(0,str(Path.cwd()))
from src.data.preprocessor import V5Preprocessor
cfg=json.loads(Path('scaler_config_v5.json').read_text())
assert cfg['classification_threshold']==.5
s=ort.InferenceSession('neurotrace_v5.onnx',providers=['CPUExecutionProvider'])
g=onnx.load('neurotrace_v5.onnx');onnx.checker.check_model(g)
print('ORT',ort.__version__,'NumPy',np.__version__,'SciPy',scipy.__version__)
print('INPUT',[(v.name,v.shape,v.type) for v in s.get_inputs()]);print('OUTPUT',[(v.name,v.shape,v.type) for v in s.get_outputs()])
print('OPSET',[(o.domain,o.version) for o in g.opset_import]);print('FINAL',g.graph.node[-1].op_type)
p=V5Preprocessor(cfg['mean'],cfg['scale'])
# Same six deterministic fixture paths as tests/test_js_preprocessor.js trajectory().
specs=[(41,'missing',False,{}),(190,'real',False,{}),(650,'missing',True,{}),(510,'constant',False,{}),(400,'real',True,{'window':128,'stride':53}),(320,'real',False,{})]
rows=[]
for k,(n,mode,irregular,original_options) in enumerate(specs):
 t=np.zeros(n)
 for i in range(1,n): t[i]=t[i-1]+(.007+.004*(1+np.sin(i*.7)) if irregular else .01)
 x=300+80*np.cos(t*4.1)+2*np.sin(2*np.pi*25*t);y=200+50*np.sin(t*3.7)
 pressure=None if mode=='missing' else (np.full(n,.5) if mode=='constant' else .5+.1*np.sin(t*9))
 if k==5: pressure[40]=np.nan
 # Use production-default windows for ALL diagnostic cases (not fixture custom option).
 results={}
 for name,pr in [('baseline',pressure),('constant',np.full(n,.5)),('proxy',None)]:
  tensor=p.transform(x,y,t,pr)
  # Per-window calls remove ONNX dynamic-batch ambiguity from measurement.
  logits=np.array([s.run(['logits'],{'input':w[None]})[0].item() for w in tensor])
  probs=expit(logits)
  results[name]={'n_windows':len(probs),'window_probabilities':probs.tolist(),'trial_mean':float(probs.mean()),'first_window':float(probs[0]),'sigmoid_mean_logit':float(expit(logits.mean()))}
 rows.append({'fixture':k+1,'n_points':n,'baseline_pressure':mode,'domain':'unknown','clinical_label':'unknown','results':results})
def stats(a):
 a=np.asarray(a);return dict(n=len(a),mean=float(np.mean(a)),median=float(np.median(a)),p10=float(np.percentile(a,10)),p90=float(np.percentile(a,90)))
windows=[v for r in rows for v in r['results']['baseline']['window_probabilities']]
trials=[r['results']['baseline']['trial_mean'] for r in rows]
deltas={name:stats([r['results'][name]['trial_mean']-r['results']['baseline']['trial_mean'] for r in rows]) for name in ['constant','proxy']}
result=dict(versions={'onnxruntime':ort.__version__,'numpy':np.__version__,'scipy':scipy.__version__},input=[dict(name=v.name,shape=v.shape,type=v.type) for v in s.get_inputs()],output=[dict(name=v.name,shape=v.shape,type=v.type) for v in s.get_outputs()],opset=[o.version for o in g.opset_import],last_op=g.graph.node[-1].op_type,rows=rows,window_stats=stats(windows),trial_stats=stats(trials),deltas=deltas)
Path('/tmp/v5_domain_results.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
```
