"""Small numeric helpers shared across the model."""

import torch
from torch import Tensor


def inverse_sigmoid(x: Tensor, eps: float = 1e-5) -> Tensor:
    """Numerically stable inverse sigmoid (logit).

    Clamps *x* to (eps, 1 - eps) before computing log(x / (1 - x)).
    """
    x = x.clamp(min=0.0, max=1.0)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)
