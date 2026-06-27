"""Device selection for neural style transfer.

`pick_device` honours an explicit request and resolves ``"auto"`` to the best
available accelerator (CUDA -> MPS -> CPU). Kept dependency-free so any entry
point (or another project) can reuse it.
"""

from __future__ import annotations

import torch

########################################
#           Device selection           #
########################################


def pick_device(
    requested: str,
) -> torch.device:
    # ── Honour an explicit request verbatim ──
    if requested != "auto":
        return torch.device(requested)

    # ── "auto": prefer CUDA, then MPS, else CPU ──
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
