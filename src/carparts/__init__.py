"""carparts — commercial-grade car-parts detection & instance segmentation.

Two model families are trained from one canonical COCO pipeline:

* ``exterior_seg``  — 23 exterior body-part classes with instance masks (carparts-seg, CC-BY-4.0)
* ``engine_bay_det`` — 26 engine-bay component classes with boxes (Kaggle engine-bay, MIT; + optional
  CC-BY-4.0 Roboflow extras)

Backbone/detector: RF-DETR (Apache-2.0, ICLR 2026) — chosen because the whole training + inference
stack and every pretrained weight we use is permissively licensed, unlike Ultralytics YOLO (AGPL-3.0).
"""

import os as _os

# numpy's bundled OpenBLAS aborts on this Windows box when it tries to allocate one buffer per logical
# core ("OpenBLAS error: Memory allocation still failed after 10 retries"). Capping BLAS threads *before*
# numpy is imported fixes it and costs nothing for GPU training. Import `carparts` first in every script.
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_k, "4")
_os.environ.setdefault("PYTHONUTF8", "1")

__version__ = "0.1.0"
