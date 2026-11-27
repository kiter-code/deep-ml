import numpy as np

def smooth_l1(pred, target, beta=1.0, reduction='mean'):

    # 1. Проверяем корректность beta
    if beta <= 0:
        raise ValueError("Beta must be positive")

    # 2. Преобразуем данные в массивы NumPy
    pred = np.array(pred, dtype=float)
    target = np.array(target, dtype=float)

    # 3. Вычисляем абсолютную ошибку
    error = np.abs(pred - target)

    # 4. Вычисляем квадратичную ошибку
    quadratic = 0.5 * error ** 2 / beta

    # 5. Вычисляем линейную ошибку
    linear = error - 0.5 * beta

    # 6. Выбираем формулу в зависимости от beta
    loss = np.where(error < beta, quadratic, linear)

    # 7. Возвращаем результат в зависимости от reduction
    if reduction == 'mean':
        return float(np.mean(loss))

    elif reduction == 'sum':
        return float(np.sum(loss))

    elif reduction == 'none':
        return loss

    else:
        raise ValueError("Invalid reduction")