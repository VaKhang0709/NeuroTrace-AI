"""PaHaW access portal: https://bdalab.utko.feec.vutbr.cz/

Native .svc units/participant map require licensed documentation. Only load a
reviewed, explicitly harmonized local copy; never guess the .svc column units.
"""
from .schema import load_unified

def load(root=None):
    if root is None:
        raise PermissionError('PaHaW license and reviewed schema conversion required; see docs/v6_external_datasets.md')
    yield from load_unified(root)
