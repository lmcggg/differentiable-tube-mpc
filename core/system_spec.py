from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor


@dataclass(frozen=True)
class SystemSpec:
    name: str
    nx: int
    nu: int
    dt: float
    horizon_N: int
    task_horizon_H: int

    # True and nominal discrete dynamics (same nominal f in the user's formulation)
    f: Callable[[Tensor, Tensor], Tensor]                # x_{k+1}=f(x_k,u_k)
    u_project: Callable[[Tensor], Tensor]                # enforce u∈U
    sample_w: Callable[[Tensor], Tensor]                 # w_t ∈ W

    # Safety set h(x)>0 for DBaS
    h: Callable[[Tensor], Tensor]

    # Measurement / task feature y(x) used in cost (can be identity or end-effector etc.)
    y: Callable[[Tensor], Tensor]

    # Nominal target y* (constant vector)
    y_target: Tensor

