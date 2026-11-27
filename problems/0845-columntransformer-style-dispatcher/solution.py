import numpy as np

def column_transformer(X, transformers):

    # 1. Преобразуем входные данные в NumPy-массив
    X = np.array(X, dtype=float)

    # Список для хранения преобразованных групп столбцов
    result = []

    # 2. Перебираем все преобразования
    for name, kind, columns in transformers:

        # Выбираем необходимые столбцы исходной матрицы
        block = X[:, columns]

        # 3. Стандартизация
        if kind == "standard":

            mean = np.mean(block, axis=0)
            std = np.std(block, axis=0)

            # Защита от деления на ноль
            std = np.where(std == 0, 1.0, std)

            block = (block - mean) / std

        # 4. Min-Max нормализация
        elif kind == "minmax":

            min_val = np.min(block, axis=0)
            max_val = np.max(block, axis=0)

            denominator = max_val - min_val

            # Защита от деления на ноль
            denominator = np.where(
                denominator == 0, 1.0, denominator
            )

            block = (block - min_val) / denominator

        # 5. Оставляем столбцы без изменений
        elif kind == "passthrough":
            pass

        # 6. Пропускаем ненужные столбцы
        elif kind == "drop":
            continue

        # 7. Сохраняем обработанную группу столбцов
        result.append(block)

    # 8. Если все столбцы удалены, возвращаем пустой список
    if not result:
        return []

    # 9. Объединяем столбцы и округляем результат
    return np.round(np.hstack(result), 4).tolist()