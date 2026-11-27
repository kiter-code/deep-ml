import torch
from typing import Callable

def compute_hessian(
    f: Callable,
    point: torch.Tensor,
    h: float = 1e-5
) -> torch.Tensor:

    x = point.clone().to(torch.float64)
    n = len(x)

    H = torch.zeros((n, n), dtype=x.dtype, device=x.device)

    f0 = f(x)

    for i in range(n):
        for j in range(n):

            # Диагональные элементы
            if i == j:
                x_plus = x.clone()
                x_minus = x.clone()

                x_plus[i] += h
                x_minus[i] -= h

                H[i, j] = (
                    f(x_plus)
                    - 2 * f0
                    + f(x_minus)
                ) / h**2

            # Смешанные производные
            else:
                x_pp = x.clone()
                x_pm = x.clone()
                x_mp = x.clone()
                x_mm = x.clone()

                x_pp[i] += h
                x_pp[j] += h

                x_pm[i] += h
                x_pm[j] -= h

                x_mp[i] -= h
                x_mp[j] += h

                x_mm[i] -= h
                x_mm[j] -= h

                H[i, j] = (
                    f(x_pp)
                    - f(x_pm)
                    - f(x_mp)
                    + f(x_mm)
                ) / (4 * h**2)

    return H