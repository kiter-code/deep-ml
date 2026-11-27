import numpy as np

def rmse(y_true, y_pred):

    # Проверка размерностей
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes must match")

    # Проверка на пустые массивы
    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("Arrays must not be empty")

    # Вычисляем MSE
    mse = np.mean((y_true - y_pred) ** 2)

    # Вычисляем RMSE
    rmse_res = np.sqrt(mse)

    return round(rmse_res, 3)