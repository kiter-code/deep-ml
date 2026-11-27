import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):

    # 1. Преобразуем входные данные в массивы NumPy
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    # 2. Вычисляем абсолютную ошибку
    error = np.abs(y_true - y_pred)

    # 3. Вычисляем квадратичную ошибку
    quadratic = 0.5 * error ** 2

    # 4. Вычисляем линейную ошибку
    linear = delta * (error - 0.5 * delta)

    # 5. Выбираем формулу в зависимости от delta
    loss = np.where(error <= delta, quadratic, linear)

    # 6. Возвращаем среднее значение потерь
    return float(np.mean(loss))