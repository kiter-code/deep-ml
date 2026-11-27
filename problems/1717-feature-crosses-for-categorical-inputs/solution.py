import numpy as np

def cross_features(X, levels, pairs):

    # 1. Создаём столбец единиц (bias)
    names = ["1"]
    columns = [np.ones(len(X))]

    # 2. Выполняем One-Hot Encoding
    for f, values in enumerate(levels):
        for v in values:

            # Формируем название признака
            names.append(f"{f}={v}")

            # Создаём бинарный столбец
            column = [int(row[f] == v) for row in X]

            columns.append(column)

    # 3. Создаём пересечения признаков
    for i, j in pairs:
        for v in levels[j]:
            for u in levels[i]:

                # Формируем название комбинации
                names.append(f"{i}={u}&{j}={v}")

                # Проверяем совпадение двух категорий
                column = [
                    int(row[i] == u and row[j] == v)
                    for row in X
                ]

                columns.append(column)

    # 4. Объединяем столбцы в матрицу
    Phi = np.column_stack(columns).astype(float)

    # 5. Возвращаем матрицу и названия столбцов
    return Phi, names