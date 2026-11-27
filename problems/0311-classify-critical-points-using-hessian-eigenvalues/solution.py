import torch

def classify_critical_point(
    hessian: torch.Tensor,
    tol: float = 1e-10
):
    # 1. Вычисляем собственные значения
    eigenvalues = torch.linalg.eigvalsh(
        hessian.to(torch.float64)
    )

    # 2. Проверяем значения, близкие к нулю
    if torch.any(torch.abs(eigenvalues) <= tol):
        return None

    # 3. Локальный минимум
    if torch.all(eigenvalues > tol):
        return -1

    # 4. Локальный максимум
    if torch.all(eigenvalues < -tol):
        return 1

    # 5. Седловая точка
    return 0