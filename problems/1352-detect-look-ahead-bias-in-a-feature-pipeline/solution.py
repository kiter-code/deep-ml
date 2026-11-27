import numpy as np

def causal_zscore(x: np.ndarray, window: int) -> np.ndarray:

    x = np.asarray(x, dtype=float)

    # Создаём массив, заполненный NaN
    result = np.full(len(x), np.nan)

    # Начинаем с первого полного окна
    for t in range(window - 1, len(x)):

        # Выбираем только текущие и прошлые значения
        values = x[t - window + 1:t + 1]

        # Вычисляем среднее и стандартное отклонение
        mean = np.mean(values)
        std = np.std(values, ddof=0)

        # Защита от деления на ноль
        if std == 0:
            result[t] = 0.0
        else:
            # Вычисляем Z-score
            result[t] = (x[t] - mean) / std

    return result


def has_lookahead(feature_fn, x: np.ndarray) -> bool:

    # Получаем первоначальный результат
    base = feature_fn(x)

    # Создаём копию исходного массива
    x_copy = np.array(x, dtype=float, copy=True)

    # Изменяем только последнее значение
    x_copy[-1] += 1000.0

    # Повторно вычисляем признаки
    new = feature_fn(x_copy)

    # Проверяем, изменились ли предыдущие значения
    return not np.allclose(
        base[:-1],
        new[:-1],
        equal_nan=True
    )