import numpy as np

def find_leaky_features(X: np.ndarray, y: np.ndarray,
                        threshold: float) -> list:

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    result = []

    # Стандартное отклонение целевой переменной
    y_std = np.std(y)

    # Перебираем все столбцы матрицы X
    for i in range(X.shape[1]):

        col = X[:, i]

        # Проверяем наличие нулевой дисперсии
        if np.std(col) == 0 or y_std == 0:
            continue

        # Вычисляем корреляцию Пирсона
        corr = np.corrcoef(col, y)[0, 1]

        # Проверяем абсолютное значение корреляции
        if abs(corr) >= threshold:
            result.append(i)

    return result


def is_affine_function_of(col: np.ndarray,
                          y: np.ndarray) -> bool:

    col = np.asarray(col, dtype=float)
    y = np.asarray(y, dtype=float)

    # Создаём матрицу из столбца единиц и y
    A = np.column_stack([np.ones(len(y)), y])

    # Находим коэффициенты a и b
    # методом наименьших квадратов
    weights = np.linalg.lstsq(A, col, rcond=None)[0]

    # Вычисляем предсказанные значения
    predictions = A @ weights

    # Вычисляем сумму квадратов остатков
    rss = np.sum((col - predictions) ** 2)

    # Проверяем точность зависимости
    return bool(rss < 1e-9)