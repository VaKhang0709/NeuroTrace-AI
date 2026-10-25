"""Compatibility entry point for canonical UCI395 ingestion (V5 is untouched)."""
from .uci395_loader import load as _load

def load(root,**kwargs):
    yield from _load(root,**kwargs)
