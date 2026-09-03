"""GPU policy for the shared training box: GPU 0 is forbidden, GPU 3 is the default.

Applied once per process by ``import carparts`` (before torch is imported, so ``CUDA_VISIBLE_DEVICES`` takes
effect). Only kicks in on machines with >= ``MULTI_GPU_THRESHOLD`` GPUs, so laptops and single-GPU boxes are
untouched. Override knobs (do not use them on the box):

* ``CARPARTS_GPU_POLICY=off``  — skip entirely
* ``CARPARTS_ALLOW_GPU0=1``    — allow GPU 0 in the visible set
"""
from __future__ import annotations

import os
import shutil
import subprocess

FORBIDDEN_GPU = "0"
DEFAULT_GPU = "3"
MULTI_GPU_THRESHOLD = 4


def _gpu_count() -> int:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return 0
    try:
        out = subprocess.run([exe, "--query-gpu=index", "--format=csv,noheader"], capture_output=True, text=True,
                             timeout=10)
        return len([ln for ln in out.stdout.splitlines() if ln.strip()]) if out.returncode == 0 else 0
    except Exception:  # noqa: BLE001 — never let the policy itself crash a job
        return 0


def apply_gpu_policy() -> str | None:
    """Return a one-line note describing what was applied (or ``None`` when nothing applied)."""
    if os.environ.get("CARPARTS_GPU_POLICY", "").lower() == "off" or os.environ.get("_CARPARTS_GPU_POLICY_DONE"):
        return None
    os.environ["_CARPARTS_GPU_POLICY_DONE"] = "1"  # children (DataLoader workers, torchrun ranks) inherit it
    n = _gpu_count()
    if n < MULTI_GPU_THRESHOLD:
        return None
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or visible.strip() == "":
        os.environ["CUDA_VISIBLE_DEVICES"] = DEFAULT_GPU
        return f"[carparts] {n} GPUs detected and CUDA_VISIBLE_DEVICES unset -> defaulting to GPU {DEFAULT_GPU}"
    ids = [v.strip() for v in visible.split(",") if v.strip()]
    if FORBIDDEN_GPU in ids and os.environ.get("CARPARTS_ALLOW_GPU0") != "1":
        raise SystemExit(f"[carparts] GPU {FORBIDDEN_GPU} is forbidden on this box (CUDA_VISIBLE_DEVICES={visible}). "
                         f"Use GPU {DEFAULT_GPU} (default) or borrow 1,2 when idle. See HANDOFF.md.")
    return f"[carparts] using GPU(s) {visible}"
