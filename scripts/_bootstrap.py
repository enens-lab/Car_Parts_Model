"""Make `carparts` importable when the package is not installed (plain `python scripts/x.py`), and apply
the BLAS/UTF-8 environment fixes *before* numpy/torch are imported. Import this first in every script."""
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import carparts  # noqa: E402,F401  (sets OPENBLAS_NUM_THREADS etc.)

from carparts.config import load_env  # noqa: E402

load_env()
