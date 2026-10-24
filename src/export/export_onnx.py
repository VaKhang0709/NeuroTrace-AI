"""Export a trusted V5 checkpoint: python -m src.export.export_onnx --help."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from src.models.network_v5 import NeuroTraceDualHeadV5


def load_checkpoint(path):
    # Restricted loader: never unpickle arbitrary Python objects from downloads.
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    model = NeuroTraceDualHeadV5(**ckpt['model_kwargs'])
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()
    if ckpt['in_channels'] != 5 or ckpt['seq_len'] != 256:
        raise ValueError('Expected production shape (batch, 5, 256)')
    if ckpt['feature_names'] != ['x', 'y', 'pressure', 'velocity', 'dt']:
        raise ValueError('Unexpected channel order')
    return model, ckpt


def export(checkpoint, output_dir):
    import onnx
    model, ckpt = load_checkpoint(checkpoint)
    mean = np.asarray(ckpt['scaler_mean'], dtype=float)
    scale = np.asarray(ckpt['scaler_scale'], dtype=float)
    if mean.shape != (5,) or scale.shape != (5,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('Invalid checkpoint scaler statistics')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / 'neurotrace_v5.onnx'
    try:
        with torch.no_grad():
            torch.onnx.export(model, torch.zeros(1, 5, 256), str(path),
                              opset_version=17, dynamo=False, do_constant_folding=True,
                              input_names=['input'], output_names=['logits'],
                              dynamic_axes={'input': {0: 'batch_size'}, 'logits': {0: 'batch_size'}})
    finally:
        model.eval()
    onnx.checker.check_model(onnx.load(str(path)))
    if path.stat().st_size >= 15 * 1024**2:
        raise ValueError('Model exceeds 15 MiB limit')
    config = dict(feature_names=ckpt['feature_names'], mean=mean.tolist(), scale=scale.tolist(),
                  in_channels=5, seq_len=256, input_tensor_name='input',
                  input_tensor_shape=['batch_size', 5, 256], output_tensor_name='logits',
                  output_semantics='raw_logits_use_sigmoid_for_probability',
                  classification_threshold=float(ckpt['classification_threshold']),
                  onnx_opset_version=17, onnx_model_file=path.name,
                  onnx_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (output_dir / 'scaler_config_v5.json').write_text(json.dumps(config, indent=2)+'\n')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', default='artifacts/v5')
    args = parser.parse_args()
    print(export(args.checkpoint, args.output_dir))


if __name__ == '__main__':
    main()
