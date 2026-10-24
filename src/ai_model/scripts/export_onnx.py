%%writefile /content/drive/MyDrive/NEUROTRACE/export_onnx.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_onnx.py — NeuroTrace v3 deployment export (PyTorch -> ONNX + scaler JSON)
===============================================================================

Turns the v3 subject-grouped checkpoint into artefacts a web or mobile client
can actually run:

  models/neurotrace_v3.onnx      the network, dynamic batch, input (B, 6, 256)
  models/scaler_config.json      StandardScaler mean/scale for client-side
                                 preprocessing (ONNX Runtime Web / JavaScript)

===============================================================================
 MEDICAL DISCLAIMER (keep this in every artefact derived from this file)
===============================================================================
NeuroTrace AI is a RESEARCH PROTOTYPE, not a medical device. It is not approved
by any regulator (FDA / EMA / Vietnam MoH). Its output is a statistical risk
indicator for research and education only — NOT a diagnosis, and never a reason
to start, stop or change treatment. Anyone noticing tremor, stiffness, slowness
or handwriting change must see a qualified neurologist.

ĐÂY LÀ NGUYÊN MẪU NGHIÊN CỨU, KHÔNG PHẢI THIẾT BỊ Y TẾ — không phải chẩn đoán.
The disclaimer is copied into scaler_config.json so it travels with the model.

===============================================================================
 TWO DECISIONS THAT MATTER FOR DEPLOYMENT
===============================================================================
1. THE SCALER IS NOT BAKED INTO THE GRAPH.
   StandardScaler is fit per fold during training and its statistics are part of
   the model definition, so in principle it could be folded in. It is kept
   separate on purpose: the client must standardise before inference anyway if
   it ever reuses the raw channels, the two arrays stay human-inspectable, and
   a preprocessing bug stays visible instead of being hidden inside an opaque
   graph. The client-side step is one line (see the JSON and the summary).

       Xs[b, c, l] = (X[b, c, l] - mean[c]) / scale[c]

   If you would rather ship a single self-contained graph, re-run with
   --fold-scaler to wrap the standardisation inside the ONNX model, in which
   case scaler_config.json becomes documentation only.

2. THE GRAPH EMITS LOGITS BY DEFAULT, NOT PROBABILITIES.
   The .pth stores a model whose forward() returns raw logits (it is trained
   with BCEWithLogitsLoss), so the ONNX graph matches that exactly and the
   parity check compares logits to logits. Passing --apply-sigmoid appends the
   sigmoid to the exported graph so a JavaScript client does not have to
   implement it; the verification then compares probabilities instead.
   Either way the output type is recorded in scaler_config.json.

===============================================================================
 WHAT IS VERIFIED (not assumed)
===============================================================================
  * onnx.checker validates the protobuf.
  * the exported input signature really is (batch_size, 6, 256) with a DYNAMIC
    batch axis — proven by running batch sizes 1 and 8 through one session.
  * PyTorch vs ONNX Runtime outputs are compared on the same tensor and the max
    absolute difference is printed and checked against --tolerance (1e-4).
  * the scaler JSON round-trips: standardising with the JSON arrays must
    reproduce sklearn's transform on random data.
  * the model is put in eval() mode before export. Skipping this would export
    BatchNorm batch statistics and live Dropout, and the ONNX output would
    quietly disagree with a trained-model forward pass.

===============================================================================
 USAGE / CÁCH DÙNG (Google Colab)
===============================================================================
  Cell 1: paste this file (the %%writefile line creates it on Drive)
  Cell 2: !python3 /content/drive/MyDrive/NEUROTRACE/export_onnx.py

  Probabilities in the graph instead of logits:
      ... --apply-sigmoid
  Do not overwrite existing artefacts:
      ... --no-overwrite
  Skip the runtime comparison (export only):
      ... --skip-verify

Dependencies
  pip install onnx onnxruntime
  Since torch 2.9 ``torch.onnx.export`` defaults to the new torch.export-based
  exporter (``dynamo=True``), which additionally requires ``onnxscript``.
  This script passes ``dynamo=False`` explicitly and therefore does NOT need
  onnxscript — if you ever see "No module named 'onnxscript'", something
  forced the dynamo path back on.

Exit codes
  0 exported and verified        3 exported but NOT verified (ort missing/failed)
  2 missing input artefact       4 export failed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sys
import time
import traceback
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# torch — required to rebuild and export the network
# ---------------------------------------------------------------------------
try:  # pragma: no cover - environment dependent
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None
    nn = None
    HAS_TORCH = False

# ---------------------------------------------------------------------------
# onnx / onnxruntime — onnx is needed to export, ort to verify
# ---------------------------------------------------------------------------
try:  # pragma: no cover
    import onnx

    HAS_ONNX = True
except Exception:  # pragma: no cover
    onnx = None
    HAS_ONNX = False

try:  # pragma: no cover
    import onnxruntime as ort

    HAS_ORT = True
except Exception:  # pragma: no cover
    ort = None
    HAS_ORT = False


# ===========================================================================
# Configuration / Cấu hình
# ===========================================================================
PROJECT_ROOT = Path("/content/drive/MyDrive/NEUROTRACE")
MODEL_REL = Path("models")
CHECKPOINT_NAME = "neurotrace_v3_grouped.pth"
SCALER_NAME = "scaler_v3.pkl"
ONNX_NAME = "neurotrace_v3.onnx"
SCALER_JSON_NAME = "scaler_config.json"

FEATURE_NAMES: Tuple[str, ...] = ("x", "y", "z", "pressure", "velocity", "dt")
N_CHANNELS = len(FEATURE_NAMES)

INPUT_NAME = "input"
OUTPUT_NAME = "logits"
BATCH_AXIS_NAME = "batch_size"

DISCLAIMER = "RESEARCH PROTOTYPE — not a medical device, not a diagnosis."


@dataclass
class ExportConfig:
    """Everything the export needs. Echoed into the summary and the JSON."""

    project_root: str = str(PROJECT_ROOT)
    model_path: str = ""        # checkpoint (.pth)
    scaler_path: str = ""       # fitted StandardScaler (.pkl)
    onnx_path: str = ""         # output graph
    scaler_json_path: str = ""  # output scaler config

    opset: int = 17
    dynamic_batch: bool = True
    apply_sigmoid: bool = False
    fold_scaler: bool = False
    tolerance: float = 1e-4
    batch_sizes: str = "1,8"
    seed: int = 42
    overwrite: bool = True
    skip_verify: bool = False
    verbose: bool = True

    def resolved_paths(self) -> Tuple[Path, Path, Path, Path]:
        root = Path(self.project_root)
        mdir = root / MODEL_REL
        model = Path(self.model_path) if self.model_path else (mdir / CHECKPOINT_NAME)
        scaler = Path(self.scaler_path) if self.scaler_path else (mdir / SCALER_NAME)
        onnx_p = Path(self.onnx_path) if self.onnx_path else (mdir / ONNX_NAME)
        sjson = Path(self.scaler_json_path) if self.scaler_json_path else (mdir / SCALER_JSON_NAME)
        return model, scaler, onnx_p, sjson


# ===========================================================================
# Small helpers / Hàm phụ trợ
# ===========================================================================
def rule(char: str = "=", width: int = 88) -> str:
    return char * width


def section(title: str, width: int = 88) -> None:
    print("\n" + rule("=", width))
    print(f" {title}")
    print(rule("=", width))


def kv(label: str, value: Any, indent: int = 2, width: int = 36) -> None:
    print(f"{' ' * indent}{label:<{width}}: {value}")


def human(n: Any) -> str:
    return f"{n:,}" if isinstance(n, (int, np.integer)) else ("n/a" if n is None else str(n))


def bytes_human(n: int) -> str:
    step = 1024.0
    val = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if val < step or unit == "GB":
            return f"{val:.1f} {unit}" if unit != "B" else f"{int(val)} B"
        val /= step
    return f"{val:.1f} GB"


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def to_jsonable(obj: Any) -> Any:
    """numpy / Path / datetime are not JSON-native — make them so."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (Path, datetime)):
        return str(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


# ===========================================================================
# 1. Model reconstruction / Dựng lại kiến trúc
# ===========================================================================
class _FallbackHybridNet(nn.Module if HAS_TORCH else object):
    """Verbatim copy of NeuroTraceHybridNet from train_pipeline_v3_subject_grouped.py.

    Only used when the training script cannot be imported. It must stay
    byte-identical in structure and parameter names, otherwise
    ``load_state_dict(strict=True)`` will fail — which is the point of loading
    strictly: a silent architectural drift would produce a wrong model that
    still exports cleanly.
    """

    def __init__(self, in_channels: int = N_CHANNELS, conv1: int = 64, conv2: int = 128,
                 lstm_hidden: int = 64, lstm_layers: int = 2, dropout: float = 0.3,
                 dense1: int = 128, dense2: int = 64):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, conv1, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv1), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2), nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(conv1, conv2, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv2), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2), nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=conv2, hidden_size=lstm_hidden, num_layers=lstm_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, dense1), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(dense1, dense2), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(dense2, 1),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        h = self.block1(x)                 # (B, 64, L/2)
        h = self.block2(h)                 # (B, 128, L/4)
        h = h.transpose(1, 2)              # (B, L/4, 128) for the LSTM
        out, _ = self.lstm(h)              # (B, L/4, 2*hidden)
        last = out[:, -1, :]               # final timestep
        return self.head(last).squeeze(-1)


def import_trainer_class(search_dirs: List[Path]):
    """Return NeuroTraceHybridNet from the training script, or None.

    Importing the real class is preferred: it is the definition the checkpoint
    was produced by, so it cannot drift. Returns None (rather than raising)
    when the script is not on disk, so the caller can fall back.
    """
    import importlib.util

    seen = []
    for d in search_dirs:
        for name in ("train_pipeline_v3_subject_grouped",):
            candidate = d / f"{name}.py"
            seen.append(str(candidate))
            if not candidate.is_file():
                continue
            try:
                spec = importlib.util.spec_from_file_location(f"_nt_{name}", candidate)
                if spec is None or spec.loader is None:
                    continue
                mod = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = mod
                spec.loader.exec_module(mod)
                cls = getattr(mod, "NeuroTraceHybridNet", None)
                if cls is not None:
                    return cls, str(candidate)
            except Exception as exc:  # noqa: BLE001 - a broken import is not fatal
                print(f"   (could not import {candidate.name}: "
                      f"{type(exc).__name__}: {exc})")
    return None, None


def build_model(cfg_dict: Dict[str, Any], n_channels: int,
                search_dirs: List[Path]) -> Tuple[Any, str, Dict[str, Any]]:
    """Rebuild the network from the checkpoint config."""
    g = cfg_dict.get
    kwargs = dict(
        in_channels=int(n_channels),
        conv1=int(g("conv1", 64)),
        conv2=int(g("conv2", 128)),
        lstm_hidden=int(g("lstm_hidden", 64)),
        lstm_layers=int(g("lstm_layers", 2)),
        dropout=float(g("dropout", 0.3)),
    )
    cls, origin = import_trainer_class(search_dirs)
    if cls is not None:
        return cls(**kwargs), f"imported from {Path(origin).name}", kwargs
    if not HAS_TORCH:
        raise RuntimeError("torch is required to rebuild the model.")
    return _FallbackHybridNet(**kwargs), "built-in fallback copy", kwargs


# ===========================================================================
# 2. Artefact loading / Nạp artefact
# ===========================================================================
def load_checkpoint(path: Path) -> Dict[str, Any]:
    """torch.load with a weights_only fallback for older/newer torch."""
    try:
        bundle = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - torch < 1.13
        bundle = torch.load(path, map_location="cpu")
    if not isinstance(bundle, dict):
        raise RuntimeError(f"{path.name} does not hold a state bundle "
                           f"(got {type(bundle).__name__}).")
    return bundle


def load_scaler(path: Path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def pick_state_dict(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Accept either the v3 bundle or a bare state_dict."""
    for key in ("model_state_dict", "state_dict", "model"):
        v = bundle.get(key)
        if isinstance(v, dict):
            return v
    # A bare state_dict has tensor values at the top level.
    if bundle and all(torch.is_tensor(v) for v in bundle.values()):
        return bundle
    raise RuntimeError("No model weights found in the checkpoint "
                       "(expected key 'model_state_dict').")


# ===========================================================================
# 3. Scaler -> JSON / Xuất scaler sang JSON
# ===========================================================================
def export_scaler_json(scaler, bundle: Dict[str, Any], out_path: Path,
                       cfg: ExportConfig, n_channels: int, seq_len: int,
                       feature_names: List[str]) -> Dict[str, Any]:
    mean = np.asarray(scaler.mean_, dtype=np.float64).reshape(-1)
    scale = np.asarray(scaler.scale_, dtype=np.float64).reshape(-1)

    if mean.size != n_channels:
        raise RuntimeError(
            f"Scaler has {mean.size} channels but the model expects {n_channels}. "
            f"These artefacts were not produced by the same training run.")

    payload: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_checkpoint": str(bundle.get("_checkpoint_path", "")),
        "source_checkpoint_sha256": bundle.get("_checkpoint_sha256", None),

        # ---- what the client must do BEFORE the ONNX call ----
        "standardise": "Xs[b, c, l] = (X[b, c, l] - mean[c]) / scale[c]",
        "mean": [float(v) for v in mean],
        "scale": [float(v) for v in scale],

        # ---- tensor contract ----
        "input_layout": "(batch, channels, length)  — channels-first, like Conv1d",
        "feature_names": list(feature_names),
        "n_channels": int(n_channels),
        "seq_len": int(seq_len),
        "input_shape": [None, int(n_channels), int(seq_len)],
        "onnx_input_name": INPUT_NAME,
        "onnx_output_name": OUTPUT_NAME,

        # ---- output semantics ----
        "output": ("probability (sigmoid already applied inside the graph)"
                   if cfg.apply_sigmoid else
                   "logit — apply sigmoid: p = 1 / (1 + exp(-logit))"),
        "output_is_logit": (not cfg.apply_sigmoid),
        "threshold": float(bundle.get("threshold", 0.5)),
        "scaler_folded_into_graph": bool(cfg.fold_scaler),
        "scaler_still_needed_client_side": (not cfg.fold_scaler),

        "opset": int(cfg.opset),
        "dynamic_batch": bool(cfg.dynamic_batch),
        "disclaimer": DISCLAIMER,
    }

    # Carry the training metrics through so a client build can display them.
    for key, dest in (("cv_metrics", "cv_metrics"),
                      ("device_leakage_probe", "device_leakage_probe")):
        if key in bundle:
            payload[dest] = to_jsonable(bundle[key])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return payload


def verify_scaler_json(payload: Dict[str, Any], scaler, n_channels: int,
                       seq_len: int, rng: np.random.Generator) -> float:
    """Standardising with the JSON arrays must reproduce sklearn exactly."""
    n, l = 4, seq_len
    X = rng.normal(size=(n, n_channels, l)).astype(np.float64) * 3.0 + 1.5

    flat = np.transpose(X, (0, 2, 1)).reshape(-1, n_channels)
    expected = scaler.transform(flat).reshape(n, l, n_channels)
    expected = np.transpose(expected, (0, 2, 1))

    mean = np.asarray(payload["mean"], dtype=np.float64).reshape(1, n_channels, 1)
    scale = np.asarray(payload["scale"], dtype=np.float64).reshape(1, n_channels, 1)
    got = (X - mean) / scale

    return float(np.max(np.abs(got - expected)))


# ===========================================================================
# 4. ONNX export / Xuất ONNX
# ===========================================================================
class _ScalerInGraph(nn.Module if HAS_TORCH else object):
    """Wraps standardisation inside the graph (--fold-scaler)."""

    def __init__(self, inner: nn.Module, mean: np.ndarray, scale: np.ndarray):
        super().__init__()
        self.inner = inner
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, -1, 1))
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32).view(1, -1, 1))

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.inner((x - self.mean) / self.scale)


class _SigmoidHead(nn.Module if HAS_TORCH else object):
    """Appends the sigmoid so the graph outputs a probability."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return torch.sigmoid(self.inner(x))


def export_to_onnx(model, dummy: "torch.Tensor", out_path: Path,
                   opset: int, dynamic_batch: bool,
                   output_name: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Export, trying exporter back-ends in order of preference.

    The ONNX exporter API has been in flux: the TorchScript exporter
    (``dynamo=False``) is the most predictable for a CNN+BiLSTM and honours
    ``dynamic_axes`` directly, but newer torch builds are moving to the dynamo
    exporter. Rather than pin one and break on someone else's Colab image, try
    them in order and report which one worked.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {INPUT_NAME: {0: BATCH_AXIS_NAME},
                        output_name: {0: BATCH_AXIS_NAME}}

    common = dict(
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=[INPUT_NAME],
        output_names=[output_name],
    )

    # NOTE: `dynamo` must be passed EXPLICITLY. Since torch 2.9 the argument
    # defaults to True, i.e. the new torch.export-based exporter, which needs
    # the separate `onnxscript` package. Leaving it implicit therefore turns
    # "works everywhere" into "works only if onnxscript happens to be
    # installed". The legacy TorchScript exporter is tried first because it is
    # the one that reliably handles this Conv1d + BiLSTM stack.
    attempts: List[Tuple[str, Dict[str, Any]]] = []
    if dynamic_axes is not None:
        attempts.append(("torchscript (dynamo=False)",
                         dict(common, dynamo=False, dynamic_axes=dynamic_axes)))
    else:
        attempts.append(("torchscript (dynamo=False, static)",
                         dict(common, dynamo=False)))
    if dynamic_axes is not None:
        attempts.append(("dynamo (dynamic_axes)",
                         dict(common, dynamo=True, dynamic_axes=dynamic_axes)))
    attempts.append(("dynamo (default)", dict(common, dynamo=True)))

    errors: List[Tuple[str, str]] = []
    for label, kwargs in attempts:
        try:
            with warnings.catch_warnings():
                # The legacy exporter emits a deprecation notice on every call;
                # it drowns the numbers the user actually came for.
                warnings.simplefilter("ignore")
                torch.onnx.export(model, dummy, str(out_path), **kwargs)
            return label, errors
        except Exception as exc:  # noqa: BLE001
            errors.append((label, f"{type(exc).__name__}: {exc}"))

    detail = "\n  ".join(f"{lbl}: {err}" for lbl, err in errors)
    if any("onnxscript" in err for _, err in errors):
        detail += ("\n\n  The torch.export-based exporter needs the `onnxscript` package:\n"
                   "      pip install onnxscript\n"
                   "  (the legacy exporter above avoids that dependency)")
    raise RuntimeError("Every ONNX export back-end failed:\n  " + detail)


# ===========================================================================
# 5. Verification with onnxruntime / Kiểm chứng bằng onnxruntime
# ===========================================================================
def check_with_onnx(onnx_path: Path) -> None:
    if not HAS_ONNX:
        print("   [warn] onnx not installed — skipping protobuf validation "
              "(pip install onnx).")
        return
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    print(f"   OK  onnx.checker validated the graph "
          f"(opset {getattr(model, 'opset_import', [None])[0].version if model.opset_import else 'n/a'})")


def verify_with_ort(onnx_path: Path, model, cfg: ExportConfig,
                    n_channels: int, seq_len: int,
                    batch_sizes: List[int]) -> Tuple[bool, List[Dict[str, Any]]]:
    """Compare PyTorch outputs against ONNX Runtime on identical inputs."""
    if not HAS_ORT:
        print("\n   [warn] onnxruntime is not installed — the graph was exported but "
              "NOT verified.\n          pip install onnxruntime")
        return False, []

    # The reference must be deterministic or the comparison is meaningless.
    # This is not paranoia: torch.onnx.export restores the module's training
    # flag when it finishes, so a wrapper built in train() mode leaves the real
    # network in train() mode afterwards, and Dropout then makes every forward
    # pass differ. Without this guard that shows up as a mysterious large
    # "precision" gap instead of as the mode bug it really is.
    if hasattr(model, "eval"):
        model.eval()
    if getattr(model, "training", False):
        print("\n   [FAIL] the reference model is still in training mode — "
              "Dropout/BatchNorm\n          would make the comparison random.")
        return False, []

    try:
        sess = ort.InferenceSession(str(onnx_path),
                                    providers=["CPUExecutionProvider"])
    except Exception as exc:  # noqa: BLE001
        print(f"\n   [warn] could not create an InferenceSession: {exc}")
        return False, []

    # ---- inspect the exported signature ----
    inputs = sess.get_inputs()
    outputs = sess.get_outputs()
    print(f"\n   signature / chữ ký:")
    for i in inputs:
        print(f"     input  {i.name:<12} shape {i.shape}   dtype {i.type}")
    for o in outputs:
        print(f"     output {o.name:<12} shape {o.shape}   dtype {o.type}")

    in_name = inputs[0].name
    out_name = outputs[0].name
    declared = list(inputs[0].shape)
    if cfg.dynamic_batch:
        if len(declared) != 3 or declared[0] != BATCH_AXIS_NAME:
            print(f"\n   [FAIL] expected a dynamic batch axis named "
                  f"'{BATCH_AXIS_NAME}', got {declared}")
            return False, []
        if declared[1] not in (n_channels, "n_channels", None) or \
           declared[2] not in (seq_len, "seq_len", None):
            print(f"\n   [FAIL] unexpected channel/length axes: {declared}")
            return False, []
    print(f"     providers: {sess.get_providers()}")

    # ---- parity across batch sizes ----
    rng = np.random.default_rng(cfg.seed)
    rows: List[Dict[str, Any]] = []
    all_ok = True

    print(f"\n   PyTorch vs ONNX Runtime "
          f"({'probability' if cfg.apply_sigmoid else 'logit'} parity):")
    print(f"     {'batch':>6}  {'max |diff|':>12}  {'tolerance':>10}  {'result':>8}  {'ort ms':>8}")
    print("     " + "-" * 54)

    for b in batch_sizes:
        dummy = torch.from_numpy(
            rng.normal(size=(b, n_channels, seq_len)).astype(np.float32))
        with torch.no_grad():
            ref = model(dummy)
            ref_repeat = model(dummy)
        # If two identical forward passes disagree, the reference is stochastic
        # (Dropout live / model left in train mode) and no tolerance is going to
        # make this comparison meaningful. Say so instead of printing a number.
        if not torch.allclose(ref, ref_repeat, rtol=0, atol=0):
            drift = float((ref - ref_repeat).abs().max())
            print(f"     {b:>6}  {'n/a':>12}  {'—':>10}  {'FAIL':>8}  "
                  f"reference not deterministic (spread {drift:.2e}) — "
                  f"Dropout/BatchNorm still live?")
            all_ok = False
            rows.append({"batch": b, "max_abs_diff": None, "ok": False,
                         "ort_ms": float("nan"),
                         "error": f"non-deterministic reference (spread {drift:.2e})"})
            continue
        # No sigmoid here: the reference module is the very object that was
        # exported, so --apply-sigmoid is already inside it. Applying it twice
        # would compare p against sigmoid(p) and fail with a very plausible
        # looking number.
        ref_np = ref.detach().cpu().numpy()

        t0 = time.perf_counter()
        got = sess.run([out_name], {in_name: dummy.numpy()})[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if got.shape != ref_np.shape:
            print(f"     {b:>6}  shape mismatch: torch {ref_np.shape} vs ort {got.shape}")
            all_ok = False
            rows.append({"batch": b, "max_abs_diff": None, "ok": False,
                         "ort_ms": elapsed_ms,
                         "error": f"shape {ref_np.shape} vs {got.shape}"})
            continue

        diff = float(np.max(np.abs(got - ref_np)))
        ok = diff < cfg.tolerance and np.isfinite(diff)
        all_ok = all_ok and ok
        rows.append({"batch": b, "max_abs_diff": diff, "ok": bool(ok),
                     "ort_ms": float(elapsed_ms)})
        print(f"     {b:>6}  {diff:>12.3e}  {cfg.tolerance:>10.0e}  "
              f"{'PASS' if ok else 'FAIL':>8}  {elapsed_ms:>8.2f}")

    # ---- sanity: the probability variant must land in [0, 1] ----
    if cfg.apply_sigmoid and rows:
        probe = torch.from_numpy(
            rng.normal(size=(1, n_channels, seq_len)).astype(np.float32))
        p = float(sess.run([out_name], {in_name: probe.numpy()})[0].ravel()[0])
        inside = 0.0 <= p <= 1.0
        print(f"\n     range check: sample probability {p:.6f} "
              f"({'inside [0,1]' if inside else 'OUT OF RANGE'})")
        all_ok = all_ok and inside

    return all_ok, rows


# ===========================================================================
# 6. Main / Hàm chính
# ===========================================================================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="NeuroTrace v3 — export the trained model to ONNX plus a "
                    "scaler JSON for web/mobile inference (research prototype).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    d = ExportConfig()
    p.add_argument("--project-root", default=d.project_root)
    p.add_argument("--model-path", default=None, help="checkpoint .pth")
    p.add_argument("--scaler-path", default=None, help="fitted scaler .pkl")
    p.add_argument("--onnx-path", default=None, help="output .onnx")
    p.add_argument("--scaler-json-path", default=None, help="output scaler json")

    p.add_argument("--opset", type=int, default=d.opset)
    p.add_argument("--no-dynamic-batch", action="store_true")
    p.add_argument("--apply-sigmoid", action="store_true",
                   help="export probabilities instead of logits")
    p.add_argument("--fold-scaler", action="store_true",
                   help="bake the scaler into the graph instead of shipping it separately")
    p.add_argument("--tolerance", type=float, default=d.tolerance)
    p.add_argument("--batch-sizes", default=d.batch_sizes,
                   help="comma-separated batch sizes for the parity check")

    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--no-overwrite", action="store_true")
    p.add_argument("--skip-verify", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p


def parse_batch_sizes(raw: str) -> List[int]:
    out: List[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            v = int(part)
        except ValueError:
            continue
        if v > 0:
            out.append(v)
    if not out:
        out = [1, 8]
    return sorted(set(out))


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    cfg = ExportConfig(
        project_root=args.project_root,
        model_path=args.model_path or "",
        scaler_path=args.scaler_path or "",
        onnx_path=args.onnx_path or "",
        scaler_json_path=args.scaler_json_path or "",
        opset=args.opset,
        dynamic_batch=not args.no_dynamic_batch,
        apply_sigmoid=args.apply_sigmoid,
        fold_scaler=args.fold_scaler,
        tolerance=args.tolerance,
        batch_sizes=args.batch_sizes,
        seed=args.seed,
        overwrite=not args.no_overwrite,
        skip_verify=args.skip_verify,
        verbose=not args.quiet,
    )

    model_path, scaler_path, onnx_path, json_path = cfg.resolved_paths()
    batch_sizes = parse_batch_sizes(cfg.batch_sizes)

    print(rule("="))
    print(" NEUROTRACE v3 — ONNX EXPORT FOR WEB / MOBILE")
    print(" Xuất ONNX v3 cho web / di động")
    print(rule("="))
    print(f" run at       : {datetime.now().isoformat(timespec='seconds')}")
    print(f" torch        : {'yes ' + torch.__version__ if HAS_TORCH else 'NO (required)'}")
    print(f" onnx         : {'yes ' + onnx.__version__ if HAS_ONNX else 'NO (optional)'}")
    print(f" onnxruntime  : {'yes ' + ort.__version__ if HAS_ORT else 'NO (needed to verify)'}")
    print(f" checkpoint   : {model_path}")
    print(f" scaler       : {scaler_path}")
    print(f" onnx out     : {onnx_path}")
    print(f" scaler json  : {json_path}")
    print("\n RESEARCH PROTOTYPE — a statistical risk indicator, NOT a diagnosis.")

    if not HAS_TORCH:
        print("\n[FATAL] torch is required to export the model. "
              "Install it with: pip install torch")
        return 2

    # ------------------------------------------------------------ inputs
    section("[1] LOADING ARTEFACTS / NẠP ARTEFACT")
    for path, label in ((model_path, "checkpoint"), (scaler_path, "scaler")):
        if not path.exists():
            print(f"\n[FATAL] {label} not found: {path}")
            print("        Train the model first "
                  "(train_pipeline_v3_subject_grouped.py) or pass --model-path / "
                  "--scaler-path.")
            return 2
        print(f"   {label:<12} {bytes_human(path.stat().st_size):>10}  {path}")

    bundle = load_checkpoint(model_path)
    bundle["_checkpoint_path"] = str(model_path)
    bundle["_checkpoint_sha256"] = sha256_of(model_path)
    scaler = load_scaler(scaler_path)
    print(f"   sha256       {bundle['_checkpoint_sha256'][:32]}…")

    state = pick_state_dict(bundle)
    n_channels = int(bundle.get("n_channels", N_CHANNELS))
    seq_len = int(bundle.get("seq_len", 256))
    feature_names = list(bundle.get("feature_names", FEATURE_NAMES))
    if len(feature_names) != n_channels:
        feature_names = list(FEATURE_NAMES)[:n_channels]
    print(f"   bundle keys  {human(len(bundle))}")
    print(f"   channels     {n_channels} ({', '.join(feature_names)})")
    print(f"   seq_len      {seq_len}")
    print(f"   stored output: {bundle.get('output', 'logit (assumed)')}")

    # ------------------------------------------------------- rebuild model
    search_dirs = [model_path.parent, Path(cfg.project_root), Path(__file__).resolve().parent]
    model, origin, kwargs = build_model(bundle.get("config", {}) or {},
                                        n_channels, search_dirs)
    print(f"   architecture NeuroTraceHybridNet  ({origin})")
    print(f"   arch kwargs  {kwargs}")

    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        print(f"\n[FATAL] state_dict mismatch — missing={missing}, unexpected={unexpected}")
        return 2
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   weights      {human(n_params)} parameters loaded (strict)")

    # CRITICAL: eval() freezes BatchNorm statistics and disables Dropout.
    # Exporting in train() mode bakes the running estimates of whatever batch
    # happens to be passed and leaves Dropout live, so the ONNX graph would
    # disagree with a trained-model forward pass — and the parity check below
    # would be the only thing standing between that bug and production.
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print("   mode         eval() — BatchNorm frozen, Dropout disabled")

    # --------------------------------------------------------- scaler json
    section("[2] SCALER -> JSON / XUẤT SCALER SANG JSON")
    payload = export_scaler_json(scaler, bundle, json_path, cfg,
                                 n_channels, seq_len, feature_names)
    print(f"   written      {json_path}  ({bytes_human(json_path.stat().st_size)})")
    print(f"   mean         {[round(v, 4) for v in payload['mean']]}")
    print(f"   scale        {[round(v, 4) for v in payload['scale']]}")
    print(f"   client step  {payload['standardise']}")

    rng = np.random.default_rng(cfg.seed)
    scaler_diff = verify_scaler_json(payload, scaler, n_channels, seq_len, rng)
    scaler_ok = scaler_diff < 1e-10
    print(f"   round-trip   max |json - sklearn| = {scaler_diff:.3e}  "
          f"{'OK' if scaler_ok else 'MISMATCH'}")
    if not scaler_ok:
        print("   [warn] the JSON does not reproduce sklearn — do not ship it.")

    # ------------------------------------------------------------ onnx out
    section("[3] ONNX EXPORT / XUẤT ONNX")
    output_name = "probability" if cfg.apply_sigmoid else OUTPUT_NAME

    # Build the module exactly as it will appear in the graph. The verification
    # below then compares against THIS object, so --fold-scaler and
    # --apply-sigmoid are included on both sides automatically.
    inner = model
    if cfg.fold_scaler:
        inner = _ScalerInGraph(
            model,
            np.asarray(payload["mean"], dtype=np.float32),
            np.asarray(payload["scale"], dtype=np.float32))
        print("   scaler       folded into the graph (--fold-scaler)")
    if cfg.apply_sigmoid:
        inner = _SigmoidHead(inner)
        print("   output       probability (sigmoid inside the graph)")

    if onnx_path.exists() and not cfg.overwrite:
        print(f"   [skip] {onnx_path} already exists (--no-overwrite).")
        print("          the existing graph is checked against the CURRENT flags;")
        print("          if it was exported differently, parity will fail.")
    else:
        # A freshly constructed nn.Module defaults to training mode. The
        # exporter records that flag, switches to eval for tracing, and then
        # RESTORES it afterwards — which silently flips the real network back
        # into train() with live Dropout. The exported graph stays correct, but
        # the PyTorch reference used for verification becomes stochastic and
        # the parity check reports a bogus mismatch. Force eval before export.
        inner.eval()
        dummy = torch.zeros(1, n_channels, seq_len, dtype=torch.float32)
        backend, errors = export_to_onnx(inner, dummy, onnx_path, cfg.opset,
                                         cfg.dynamic_batch, output_name)
        for lbl, err in errors:
            print(f"   (back-end '{lbl}' failed: {err})")
        print(f"   back-end     {backend}")
        print(f"   opset        {cfg.opset}")
        print(f"   written      {onnx_path}  ({bytes_human(onnx_path.stat().st_size)})")

    # -------------------------------------------------------- verification
    verified = False
    rows: List[Dict[str, Any]] = []
    if cfg.skip_verify:
        section("[4] VERIFICATION / KIỂM CHỨNG")
        print("   skipped (--skip-verify).")
    else:
        section("[4] VERIFICATION / KIỂM CHỨNG")
        check_with_onnx(onnx_path)
        verified, rows = verify_with_ort(onnx_path, inner, cfg,
                                         n_channels, seq_len, batch_sizes)

    # ------------------------------------------------------------- summary
    section("SUMMARY / TỔNG KẾT")
    print(f"   onnx graph        : {onnx_path.name}  "
          f"({bytes_human(onnx_path.stat().st_size) if onnx_path.exists() else 'missing'})")
    print(f"   scaler json       : {json_path.name}")
    print(f"   input contract    : ({BATCH_AXIS_NAME if cfg.dynamic_batch else 1}, "
          f"{n_channels}, {seq_len})   channels-first")
    print(f"   output            : {'probability' if cfg.apply_sigmoid else 'logit'}"
          f"   (threshold {payload['threshold']})")
    print(f"   verification      : "
          f"{'PASSED' if verified else ('SKIPPED' if cfg.skip_verify else 'FAILED')}")
    for r in rows:
        if r.get("max_abs_diff") is not None:
            print(f"     batch {r['batch']:>3}: max |diff| {r['max_abs_diff']:.3e}  "
                  f"{'PASS' if r['ok'] else 'FAIL'}   ort {r['ort_ms']:.1f} ms")

    print("\n   CLIENT-SIDE INFERENCE (ONNX Runtime Web) / SUY LUẬN PHÍA CLIENT")
    print("     1. build the 6 channels, spline-resample to "
          f"{seq_len} points  -> Float32Array(1 * {n_channels} * {seq_len})")
    print("     2. standardise: Xs[b,c,l] = (X[b,c,l] - mean[c]) / scale[c]")
    if not cfg.fold_scaler:
        print("        (mean/scale are in scaler_config.json — apply BEFORE the run)")
    print(f"     3. session.run({{ {INPUT_NAME}: Xs }})")
    if not cfg.apply_sigmoid:
        print(f"     4. p = 1 / (1 + Math.exp(-logit))   // graph emits a LOGIT")
    print(f"     5. flag if p >= {payload['threshold']}")
    print("\n   REMINDERS / NHẮC NHỚ:")
    print("     * Research prototype — not a diagnosis, not a medical device.")
    print("     * The scaler is part of the model: skipping step 2 silently")
    print("       produces meaningless probabilities.")
    if not cfg.apply_sigmoid and not cfg.fold_scaler:
        print("     * Two client-side steps (standardise, sigmoid) are required.")
    print(f"     * Serving suggestion: ship {json_path.name} alongside the .onnx")

    print("\n" + rule("="))
    if cfg.skip_verify:
        print(" EXPORT COMPLETE — NOT VERIFIED / ĐÃ XUẤT, CHƯA KIỂM CHỨNG")
    elif verified:
        print(" EXPORT COMPLETE AND VERIFIED / XUẤT XONG, ĐÃ KIỂM CHỨNG")
    else:
        print(" EXPORT COMPLETE — VERIFICATION FAILED / XUẤT XONG, KIỂM CHỨNG THẤT BẠI")
    print(rule("=") + "\n")

    if cfg.skip_verify:
        return 0
    if not verified:
        return 3
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:  # pragma: no cover
        print("\n[INTERRUPTED] export cancelled / đã huỷ.")
        sys.exit(130)
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover
        # Flush first: stdout is block-buffered when piped, so without this the
        # traceback (stderr, unbuffered) prints ABOVE the message and the one
        # line the user actually needs scrolls away.
        sys.stdout.flush()
        print(f"\n[FATAL] unexpected error / lỗi không mong đợi: "
              f"{type(exc).__name__}: {exc}")
        sys.stdout.flush()
        traceback.print_exc()
        sys.exit(4)
