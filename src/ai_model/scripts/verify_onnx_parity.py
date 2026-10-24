%%writefile /content/drive/MyDrive/NEUROTRACE/verify_onnx_parity.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_onnx_parity.py — NeuroTrace v5 PyTorch vs ONNX Runtime Parity Auditor
=============================================================================
Validates exact numerical parity between `neurotrace_v5_production.pth` (PyTorch)
and `neurotrace_v5.onnx` (`onnxruntime.InferenceSession`) across 100 test tensors
(50 synthetic multi-batch stress tensors + 50 real human kinematic windows from
UCI #395 Wacom & Mobile Touchscreen subjects).

Strict Pass Criterion:
  Max |Logit_PyTorch - Logit_ORT| < 1.0e-5 across all 100 test tensors.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import onnxruntime as ort
import torch


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Verify PyTorch vs ONNX Runtime Numerical Parity")
    parser.add_argument("--project-root", type=str, default="/content/drive/MyDrive/NEUROTRACE")
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args(argv)

    root = Path(args.project_root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import train_pipeline_v5_diagnostic as v5

    models_dir = root / "models"
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = models_dir / "neurotrace_v5_production.pth"
    onnx_path = models_dir / "neurotrace_v5.onnx"
    scaler_cfg_path = models_dir / "scaler_config_v5.json"

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    scaler_cfg = json.loads(scaler_cfg_path.read_text(encoding="utf-8"))

    model = v5.NeuroTraceDiagnosticNet(**ckpt["model_kwargs"])
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_channels = int(ckpt["in_channels"])
    seq_len = int(ckpt["seq_len"])
    mean_arr = np.asarray(scaler_cfg["mean"], dtype=np.float32).reshape(1, in_channels, 1)
    scale_arr = np.asarray(scaler_cfg["scale"], dtype=np.float32).reshape(1, in_channels, 1)

    rng = np.random.default_rng(20261023)
    test_cases: List[tuple[str, np.ndarray]] = []

    # 50 Synthetic multi-batch & edge-amplitude tensors (covering batch_size = 1, 2, 4, 8, 16, 32)
    batch_cycle = [1, 2, 4, 8, 16, 32]
    for i in range(50):
        bs = batch_cycle[i % len(batch_cycle)]
        amp = float(0.25 + 0.5 * (i % 6))
        arr = (rng.standard_normal((bs, in_channels, seq_len)) * amp).astype(np.float32)
        test_cases.append((f"synthetic_{i+1:03d}_bs{bs}", arr))

    # 50 Real human participant tensors (25 UCI Wacom + 25 Mobile App Touchscreen)
    drop_idx = tuple(
        idx for idx, name in enumerate(v5.CANONICAL_FEATURES) if name not in ckpt["feature_names"]
    )
    cfg_real = v5.V5Config(
        project_root=str(root),
        source_filter="all",
        max_windows_per_subject=20,
        drop_channels=drop_idx,
        tag="parity_audit",
    )
    bundle, _ = v5.build_v5_dataset(cfg_real)
    X_real_std = ((bundle["X"] - mean_arr) / scale_arr).astype(np.float32)
    wac_indices = np.where(bundle["sources"] == 0)[0]
    app_indices = np.where(bundle["sources"] == 1)[0]
    sel_wac = rng.choice(wac_indices, size=25, replace=False)
    sel_app = rng.choice(app_indices, size=25, replace=False)

    for k, idx in enumerate(sel_wac, start=1):
        pid = str(bundle["parents"][idx])
        test_cases.append((f"real_uci_{k:02d}_{pid}", X_real_std[idx:idx + 1]))
    for k, idx in enumerate(sel_app, start=1):
        pid = str(bundle["parents"][idx])
        test_cases.append((f"real_app_{k:02d}_{pid}", X_real_std[idx:idx + 1]))

    assert len(test_cases) == 100, f"Expected 100 parity test cases, got {len(test_cases)}"

    max_logit_diffs: List[float] = []
    mean_logit_diffs: List[float] = []
    max_prob_diffs: List[float] = []
    detail_lines: List[str] = []

    with torch.no_grad():
        for name, arr in test_cases:
            pt_logits = model(torch.from_numpy(arr)).detach().cpu().numpy().reshape(-1)
            ort_logits = np.asarray(sess.run(["logits"], {"input": arr})[0]).reshape(-1)
            diff_l = np.abs(pt_logits.astype(np.float64) - ort_logits.astype(np.float64))
            pt_prob = 1.0 / (1.0 + np.exp(-pt_logits.astype(np.float64)))
            ort_prob = 1.0 / (1.0 + np.exp(-ort_logits.astype(np.float64)))
            diff_p = np.abs(pt_prob - ort_prob)

            mx_l = float(np.max(diff_l))
            mn_l = float(np.mean(diff_l))
            mx_p = float(np.max(diff_p))
            max_logit_diffs.append(mx_l)
            mean_logit_diffs.append(mn_l)
            max_prob_diffs.append(mx_p)
            detail_lines.append(
                f"  [{len(max_logit_diffs):03d}/100] {name:<38s} | shape={str(arr.shape):<14s} | "
                f"max|dLogit|={mx_l:.3e} | max|dProb|={mx_p:.3e}"
            )

    global_max_logit = max(max_logit_diffs)
    global_mean_logit = float(np.mean(mean_logit_diffs))
    global_max_prob = max(max_prob_diffs)
    passed = global_max_logit < float(args.tolerance)

    header = [
        "==========================================================================================",
        " NEUROTRACE AI v5 — PYTORCH vs ONNX RUNTIME PARITY VERIFICATION REPORT",
        "==========================================================================================",
        f"  Timestamp (UTC)          : {datetime.now(timezone.utc).isoformat()}",
        f"  PyTorch Checkpoint       : {ckpt_path}",
        f"  ONNX Runtime Model       : {onnx_path}",
        f"  ONNX Runtime Version     : {ort.__version__}",
        f"  PyTorch Version          : {torch.__version__}",
        f"  Total Verified Tensors   : {len(test_cases)} (50 synthetic multi-batch + 50 real human subjects)",
        f"  Global Max |dLogit|      : {global_max_logit:.6e} (Threshold < {args.tolerance:.1e})",
        f"  Global Mean |dLogit|     : {global_mean_logit:.6e}",
        f"  Global Max |dProb|       : {global_max_prob:.6e}",
        f"  Parity Gate Status       : {'PASS' if passed else 'FAIL'}",
        "------------------------------------------------------------------------------------------",
        "  Per-Tensor Verification Trace (All 100 Tensors):",
        *detail_lines,
        "==========================================================================================",
    ]
    report_text = "\n".join(header) + "\n"
    (models_dir / "parity_report_v5.txt").write_text(report_text, encoding="utf-8")
    (reports_dir / "parity_report_v5.txt").write_text(report_text, encoding="utf-8")

    print("\n".join(header[:14]))
    print("  Sample Verified Tensors (First 5 & Last 5 of 100):")
    for ln in detail_lines[:5] + ["  ..."] + detail_lines[-5:]:
        print(ln)
    print("=" * 90)

    if not passed:
        raise AssertionError(
            f"ONNX parity check failed: max |dLogit|={global_max_logit:.6e} >= {args.tolerance:.1e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
