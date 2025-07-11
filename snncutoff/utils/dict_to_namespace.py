
from types import SimpleNamespace

def dict_to_namespace(d):
    """Recursively convert a dictionary to a SimpleNamespace."""
    if isinstance(d, dict):
        # Recursively convert each dictionary entry to a SimpleNamespace
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        # Recursively process each item in a list (if there are nested dicts in lists)
        return [dict_to_namespace(item) for item in d]
    else:
        # Return any non-dict/list items as-is
        return d
