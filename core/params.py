from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def _pos(x: Tensor) -> Tensor:
    """Positive transform for weights (keeps optimization unconstrained)."""
    return torch.nn.functional.softplus(x)


@dataclass
class NominalTheta:
    # raw parameters (unconstrained)
    Q_raw: Tensor   # [nx]
    R_raw: Tensor   # [nu]
    Qf_raw: Tensor  # [nx]
    qb_raw: Tensor  # []

    # DBaS parameters (raw -> constrained where needed)
    alpha_raw: Tensor  # []
    gamma_raw: Tensor  # [] -> mapped to [-1,1]

    # Constraint tightening for nominal MPC (implements x_k ∈ X̄(θ̄) via h̄(x)=h(x)-s)
    tight_raw: Tensor  # [] -> s >= 0

    def Q(self) -> Tensor: return _pos(self.Q_raw)
    def R(self) -> Tensor: return _pos(self.R_raw)
    def Qf(self) -> Tensor: return _pos(self.Qf_raw)
    def qb(self) -> Tensor: return _pos(self.qb_raw)
    def alpha(self) -> Tensor: return _pos(self.alpha_raw) + 1e-6
    def gamma(self) -> Tensor: return torch.tanh(self.gamma_raw)  # in (-1,1)
    def tight(self) -> Tensor: return _pos(self.tight_raw)

    def tensors(self) -> list[Tensor]:
        return [self.Q_raw, self.R_raw, self.Qf_raw, self.qb_raw, self.alpha_raw, self.gamma_raw, self.tight_raw]


@dataclass
class AuxiliaryTheta:
    Q_raw: Tensor
    R_raw: Tensor
    Qf_raw: Tensor
    qb_raw: Tensor

    alpha_raw: Tensor
    gamma_raw: Tensor

    def Q(self) -> Tensor: return _pos(self.Q_raw)
    def R(self) -> Tensor: return _pos(self.R_raw)
    def Qf(self) -> Tensor: return _pos(self.Qf_raw)
    def qb(self) -> Tensor: return _pos(self.qb_raw)
    def alpha(self) -> Tensor: return _pos(self.alpha_raw) + 1e-6
    def gamma(self) -> Tensor: return torch.tanh(self.gamma_raw)

    def tensors(self) -> list[Tensor]:
        return [self.Q_raw, self.R_raw, self.Qf_raw, self.qb_raw, self.alpha_raw, self.gamma_raw]

