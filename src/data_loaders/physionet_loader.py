"""Access-gated physionet adapter, not a fabricated native-format parser.

See docs/v6_external_datasets.md for source URLs and verification blockers.
Requires a locally authorized conversion to neurotrace-trajectory-1. Static
images and tapping summaries cannot be treated as handwriting time series.
"""
from .schema import load_unified

def load(root=None):
    if root is None:
        raise PermissionError('physionet: approved data and reviewed conversion required')
    yield from load_unified(root)
