import numpy as np

def dummy_regressor(y_train, n_test, strategy='mean',
                    constant=None, quantile=None):

    # Преобразуем данные в NumPy-массив
    y_train = np.array(y_train, dtype=float)

    # 1. Среднее арифметическое
    if strategy == "mean":
        p = np.mean(y_train)

    # 2. Медиана
    elif strategy == "median":
        p = np.median(y_train)

    # 3. Квантиль
    elif strategy == "quantile":

        # Проверяем корректность квантиля
        if quantile is None or not 0 <= quantile <= 1:
            raise ValueError("Invalid quantile")

        p = np.quantile(y_train, quantile)

    # 4. Постоянное значение
    elif strategy == "constant":

        # Проверяем, задана ли константа
        if constant is None:
            raise ValueError("Constant is required")

        p = constant

    # 5. Обрабатываем неизвестную стратегию
    else:
        raise ValueError("Unknown strategy")

    # Повторяем предсказание n_test раз
    return [float(p)] * n_test