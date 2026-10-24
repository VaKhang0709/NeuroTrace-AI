import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from src.data.preprocessor import V5Preprocessor, extract_trial_windows, SOS_LP_12HZ
from scipy import signal


def test_preprocessing_matches_legacy():
    source = Path('src/ai_model/scripts/train_pipeline_v5_diagnostic.py').read_text().split('\n', 1)[1]
    tree = ast.parse(source)
    names = {'virtual_pressure_proxy', 'slice_windows', 'resample_window', 'extract_trial_windows'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = dict(np=np, signal=signal, SOS_LP_12HZ=SOS_LP_12HZ, List=list,
              RawTrial=object, V5Config=object)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<legacy>', 'exec'), ns)
    t = np.arange(600) / 100
    trial = SimpleNamespace(x=np.sin(t*4), y=np.cos(t*4), t=t,
                            z=np.full_like(t, np.nan), pressure=np.full_like(t, np.nan), has_real_pressure=False)
    cfg = SimpleNamespace(window=256, stride=64, seq_len=256)
    np.testing.assert_array_equal(extract_trial_windows(trial, cfg), ns['extract_trial_windows'](trial, cfg))
    output = V5Preprocessor([0]*5, [1]*5).transform(trial.x, trial.y, t)
    assert output.shape[1:] == (5, 256)
    assert output.dtype == np.float32
    assert np.isfinite(output).all()


def test_invalid_inputs():
    with pytest.raises(ValueError):
        V5Preprocessor([0]*5, [0]*5)
    pre = V5Preprocessor([0]*5, [1]*5)
    with pytest.raises(ValueError):
        pre.transform(np.ones(30), np.ones(30), np.zeros(30))


def test_architecture_compatibility():
    torch = pytest.importorskip('torch')
    from src.models.network_v5 import NeuroTraceDualHeadV5, GradientReversalLayer
    model = NeuroTraceDualHeadV5().eval()
    x = torch.randn(2, 5, 256)
    with torch.no_grad():
        clinical, device, features = model.forward_dual(x)
        torch.testing.assert_close(model(x), clinical)
        assert clinical.shape == device.shape == (2,)
        assert features.shape == (2, 4)
    a = torch.ones(2, requires_grad=True)
    GradientReversalLayer(0.3)(a).sum().backward()
    torch.testing.assert_close(a.grad, torch.full_like(a, -0.3))
