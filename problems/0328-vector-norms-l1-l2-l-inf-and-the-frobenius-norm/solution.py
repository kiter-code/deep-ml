import numpy as np

def compute_norm(arr: np.ndarray, norm_type: str) -> float:

    # 1. Преобразуем входные данные в массив NumPy
    arr = np.array(arr, dtype=float)

    # 2. Вычисляем норму L1
    if norm_type == 'l1':
        result = np.sum(np.abs(arr))

    # 3. Вычисляем норму L2
    elif norm_type == 'l2':
        result = np.sqrt(np.sum(arr ** 2))

    # 4. Вычисляем норму L-infinity
    elif norm_type == 'linf':
        result = np.max(np.abs(arr))

    # 5. Вычисляем норму Фробениуса
    elif norm_type == 'frobenius':

        # Проверяем, является ли массив двумерным
        if arr.ndim != 2:
            raise ValueError("Array must be 2D")

        result = np.sqrt(np.sum(arr ** 2))

    # 6. Обрабатываем неизвестный тип нормы
    else:
        raise ValueError("Invalid norm type")

    # 7. Возвращаем результат
    return float(result)