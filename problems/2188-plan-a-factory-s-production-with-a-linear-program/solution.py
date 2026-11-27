import numpy as np
from itertools import combinations

def plan_production(profit, usage, capacity, demand_cap=None):

    # 1. Преобразуем входные данные в массивы
    profit = np.array(profit, dtype=float)
    usage = np.array(usage, dtype=float)
    capacity = np.array(capacity, dtype=float)

    # Количество видов продукции
    n = len(profit)

    # 2. Формируем ограничения Ax <= b
    A = usage.copy()
    b = capacity.copy()

    # 3. Добавляем условие неотрицательности x >= 0
    # Эквивалентная запись: -x <= 0
    A = np.vstack([A, -np.eye(n)])
    b = np.concatenate([b, np.zeros(n)])

    # 4. Добавляем ограничения спроса
    if demand_cap is not None:
        for i, cap in enumerate(demand_cap):

            if cap is not None:
                row = np.zeros(n)
                row[i] = 1

                A = np.vstack([A, row])
                b = np.append(b, cap)

    # 5. Инициализируем лучшее решение
    best_profit = -np.inf
    best_plan = None

    # 6. Перебираем комбинации из n ограничений
    for indices in combinations(range(len(b)), n):

        indices = list(indices)

        # Выбираем активные ограничения
        A_sub = A[indices]
        b_sub = b[indices]

        # 7. Находим точку пересечения ограничений
        try:
            x = np.linalg.solve(A_sub, b_sub)

        except np.linalg.LinAlgError:
            # Пропускаем вырожденные системы
            continue

        # 8. Проверяем выполнение всех ограничений
        if np.all(A @ x <= b + 1e-8):

            # Вычисляем прибыль
            current_profit = profit @ x

            # 9. Обновляем лучшее решение
            if current_profit > best_profit:
                best_profit = current_profit
                best_plan = x.copy()

    # 10. Находим полностью использованные ресурсы
    remaining = capacity - usage @ best_plan

    binding = [
        i for i, value in enumerate(remaining)
        if value <= 1e-6
    ]

    # 11. Возвращаем результат
    return {
        "profit": round(float(best_profit), 4),
        "plan": [
            round(float(v), 4) for v in best_plan
        ],
        "binding": binding
    }