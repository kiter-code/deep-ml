import numpy as np

def cross_val_predict(X, y, k):

    # 1. Преобразуем данные в массивы NumPy
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    n = len(y)

    # 2. Добавляем столбец единиц для bias
    X = np.column_stack((np.ones(n), X))

    # 3. Разделяем индексы на k блоков без перемешивания
    indices = np.arange(n)
    folds = np.array_split(indices, k)

    # 4. Создаем массив для предсказаний
    predictions = np.zeros(n)

    # 5. Проходим по каждому блоку
    for i in range(k):

        # Индексы валидационной выборки
        val_idx = folds[i]

        # Индексы обучающей выборки
        train_idx = np.concatenate([
            folds[j] for j in range(k) if j != i
        ])

        # 6. Формируем обучающую выборку
        X_train = X[train_idx]
        y_train = y[train_idx]

        # 7. Формируем валидационную выборку
        X_val = X[val_idx]

        # 8. Обучаем линейную регрессию методом OLS
        theta = np.linalg.lstsq(
            X_train, y_train, rcond=None
        )[0]

        # 9. Получаем и сохраняем предсказания
        predictions[val_idx] = X_val @ theta

    # 10. Возвращаем обычный список Python
    return predictions.tolist()