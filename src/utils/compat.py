"""
compat.py
---------
Compatibility shims for optional dependencies (tqdm, xgboost, lightgbm, mlxtend).
Import from here so the rest of the codebase never crashes on a missing dep.
"""
from __future__ import annotations

# ── tqdm ──────────────────────────────────────────────────────────────────────
try:
    from tqdm import tqdm, trange  # noqa: F401
except ImportError:
    # Minimal no-op replacement
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, iterable=None, *args, **kwargs):
            self._it = iterable
        def __iter__(self):
            return iter(self._it) if self._it is not None else iter([])
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def update(self, n=1): pass
        def set_description(self, s): pass
        def close(self): pass
        @staticmethod
        def write(msg): print(msg)

    def trange(*args, **kwargs):  # type: ignore[misc]
        return range(*args[:1])

# ── xgboost ───────────────────────────────────────────────────────────────────
try:
    import xgboost as xgb  # noqa: F401
    HAS_XGBOOST = True
except ImportError:
    xgb = None  # type: ignore[assignment]
    HAS_XGBOOST = False

# ── lightgbm ──────────────────────────────────────────────────────────────────
try:
    import lightgbm as lgb  # noqa: F401
    HAS_LIGHTGBM = True
except ImportError:
    lgb = None  # type: ignore[assignment]
    HAS_LIGHTGBM = False

# ── mlxtend ───────────────────────────────────────────────────────────────────
try:
    from mlxtend.classifier import EnsembleVoteClassifier  # noqa: F401
    HAS_MLXTEND = True
except ImportError:
    HAS_MLXTEND = False
