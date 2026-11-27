import torch

def lagrange_optimize(
    Q: torch.Tensor,
    c: torch.Tensor,
    a: torch.Tensor,
    b: float
) -> dict:

    # 1. Преобразуем входные данные
    Q = torch.as_tensor(Q, dtype=torch.float64)
    c = torch.as_tensor(c, dtype=torch.float64, device=Q.device)
    a = torch.as_tensor(a, dtype=torch.float64, device=Q.device)

    # 2. Создаём верхнюю часть матрицы KKT
    top = torch.cat([Q, a.reshape(-1, 1)], dim=1)

    # 3. Создаём нижнюю часть матрицы KKT
    bottom = torch.cat([
        a.reshape(1, -1),
        Q.new_zeros((1, 1))
    ], dim=1)

    # 4. Объединяем части матрицы
    K = torch.cat([top, bottom], dim=0)

    # 5. Формируем правую часть уравнения
    rhs = torch.cat([-c, Q.new_tensor([b])])

    # 6. Решаем систему линейных уравнений
    solution = torch.linalg.solve(K, rhs)

    # 7. Извлекаем оптимальные параметры
    x = solution[:-1]
    lam = -solution[-1]

    # 8. Вычисляем значение целевой функции
    objective = 0.5 * x @ Q @ x + c @ x

    # 9. Возвращаем результат
    return {
        "x": [round(v, 4) for v in x.tolist()],
        "lambda": round(lam.item(), 4),
        "objective": round(objective.item(), 4)
    }