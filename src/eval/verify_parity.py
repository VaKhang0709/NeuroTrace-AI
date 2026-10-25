"""Compare raw logits for exactly 100 tensors; exit nonzero on parity failure."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from src.export.export_onnx import load_checkpoint


def verify(checkpoint, onnx_path, report, real_windows=None, tolerance=1e-5):
    import onnxruntime as ort
    if not 0 < tolerance <= 1e-5:
        raise ValueError('Tolerance must be in (0, 1e-5]')
    model, _ = load_checkpoint(checkpoint)
    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    rng = np.random.default_rng(20261023)
    real = None
    if real_windows:
        real = np.load(real_windows, allow_pickle=False)
        if real.ndim != 3 or real.shape[0] < 50 or real.shape[1:] != (5, 256) or not np.isfinite(real).all():
            raise ValueError('Real data must contain >=50 standardized (5,256) windows')
    details = []
    with torch.no_grad():
        for i in range(100):
            if real is not None and i >= 50:
                arr = real[i-50:i-49].astype(np.float32)
                source = 'real_window'
            else:
                batch = [1, 2, 4, 8][i % 4]
                arr = (rng.standard_normal((batch, 5, 256)) * (0.25 + i % 6 * 0.5)).astype(np.float32)
                source = 'synthetic'
            pt = model(torch.from_numpy(arr)).numpy()
            runtime = session.run(['logits'], {'input': arr})[0]
            if runtime.shape != pt.shape:
                raise ValueError('Output shape mismatch')
            error = float(np.max(np.abs(pt.astype(float)-runtime.astype(float))))
            details.append(dict(index=i, source=source, shape=list(arr.shape), max_logit_error=error if np.isfinite(error) else None))
    finite = all(d['max_logit_error'] is not None for d in details)
    maximum = max(d['max_logit_error'] for d in details) if finite else None
    result = dict(passed=finite and maximum < tolerance, max_logit_error=maximum,
                  tolerance=tolerance, real_windows=50 if real is not None else 0, tensors=details)
    path = Path(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(f"Parity {'PASS' if result['passed'] else 'FAIL'}: max logit error={maximum}; report={path}")
    return result['passed']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--onnx', required=True)
    parser.add_argument('--real-windows', help='Optional .npy of pre-standardized real windows; otherwise synthetic-only')
    parser.add_argument('--report', default='artifacts/v5/parity_report_v5.json')
    args = parser.parse_args()
    raise SystemExit(0 if verify(args.checkpoint, args.onnx, args.report, args.real_windows) else 1)


if __name__ == '__main__':
    main()
