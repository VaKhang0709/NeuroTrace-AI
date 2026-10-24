%%writefile /content/drive/MyDrive/NEUROTRACE/export_onnx_v5.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_onnx_v5.py — NeuroTrace v5 Phase 5 Production ONNX Exporter
==================================================================
Exports ONLY the clean clinical inference path (`NeuroTraceDiagnosticNet.forward(x)`
-> RAW LOGITS `(batch_size,)`) to a single self-contained ONNX graph (`< 15 MB`)
strictly adhering to v5 ONNX Export Standards:
  - `dynamo=False`
  - `opset_version=17`
  - `do_constant_folding=True`
  - Dynamic batch axis `(batch_size, input_channels, 256)`
  - `model.eval()` invoked explicitly BOTH before AND after tracing
  - Zero GradientReversalLayer or Device Classifier ops present in exported graph
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import onnx
import torch


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Export NeuroTrace v5 Production Model to ONNX")
    parser.add_argument("--project-root", type=str, default="/content/drive/MyDrive/NEUROTRACE")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args(argv)

    root = Path(args.project_root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from train_pipeline_v5_diagnostic import NeuroTraceDiagnosticNet

    models_dir = root / "models"
    ckpt_path = Path(args.checkpoint) if args.checkpoint else models_dir / "neurotrace_v5_production.pth"
    onnx_path = Path(args.output) if args.output else models_dir / "neurotrace_v5.onnx"
    scaler_cfg_path = models_dir / "scaler_config_v5.json"

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Production checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = NeuroTraceDiagnosticNet(**ckpt["model_kwargs"])
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    # Mandatory pre-trace eval mode
    model.eval()

    in_channels = int(ckpt["in_channels"])
    seq_len = int(ckpt["seq_len"])
    dummy_input = torch.randn(1, in_channels, seq_len, dtype=torch.float32)

    # Remove any external data sidecar before clean single-file export
    for p in (onnx_path, Path(f"{onnx_path}.data")):
        if p.exists():
            p.unlink()

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            dynamo=False,
        )

    # Mandatory post-trace eval assertion
    model.eval()
    assert not model.training, "Model left eval() state after ONNX trace!"

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    op_types = sorted({node.op_type for node in onnx_model.graph.node})
    node_names = [node.name for node in onnx_model.graph.node]
    assert not any("device_classifier" in n or "grl" in n.lower() for n in node_names), (
        "Adversarial training branch leaked into production ONNX graph!"
    )
    assert "Sigmoid" not in [onnx_model.graph.node[-1].op_type], (
        "Final ONNX node must emit RAW LOGITS, not probabilities!"
    )

    size_bytes = onnx_path.stat().st_size
    size_mb = size_bytes / (1024.0 * 1024.0)
    assert size_mb < 15.0, f"ONNX size {size_mb:.2f} MB exceeds 15 MB ceiling!"

    # Ensure companion scaler_config_v5.json carries full ONNX input/output contract
    existing_cfg = (
        json.loads(scaler_cfg_path.read_text(encoding="utf-8"))
        if scaler_cfg_path.is_file()
        else {}
    )
    existing_cfg.update({
        "onnx_model_file": onnx_path.name,
        "onnx_sha256": sha256_file(onnx_path),
        "onnx_size_bytes": int(size_bytes),
        "onnx_opset_version": 17,
        "input_tensor_name": "input",
        "input_tensor_shape": ["batch_size", in_channels, seq_len],
        "output_tensor_name": "logits",
        "output_semantics": "raw_logits_use_sigmoid_for_probability",
        "in_channels": in_channels,
        "seq_len": seq_len,
        "feature_names": list(ckpt["feature_names"]),
        "mean": [float(v) for v in ckpt["scaler_mean"]],
        "scale": [float(v) for v in ckpt["scaler_scale"]],
        "classification_threshold": float(ckpt["classification_threshold"]),
        "exported_at": datetime.now(timezone.utc).isoformat(),
    })
    scaler_cfg_path.write_text(json.dumps(existing_cfg, indent=2), encoding="utf-8")

    print("=" * 88)
    print(" NEUROTRACE v5 ONNX EXPORT SUMMARY (PHASE 5)")
    print("=" * 88)
    print(f"  Source Checkpoint    : {ckpt_path}")
    print(f"  Exported ONNX File   : {onnx_path}")
    print(f"  File Size            : {size_bytes:,} bytes ({size_mb:.4f} MB < 15.0 MB PASS)")
    print(f"  SHA-256              : {existing_cfg['onnx_sha256']}")
    print(f"  Input Tensor Shape   : ('batch_size', {in_channels}, {seq_len}) | Channels: {ckpt['feature_names']}")
    print(f"  Output Tensor        : 'logits' ('batch_size',) [RAW LOGITS — Sigmoid Free]")
    print(f"  ONNX Graph Ops ({len(onnx_model.graph.node):2d})  : {op_types}")
    print(f"  Companion Scaler Cfg : {scaler_cfg_path}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
