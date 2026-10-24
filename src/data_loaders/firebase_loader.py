"""Read authorized unified ingestion records, not manifest/QC JSON files."""
import json
from pathlib import Path
from .schema import Trajectory

def load(root):
    for path in sorted(Path(root).glob('*/*/*.json')):
        yield Trajectory.from_dict(json.loads(path.read_text()))
