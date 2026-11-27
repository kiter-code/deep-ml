# --- cell 1 ---
import numpy as np


def split_and_baseline(X, y, train_frac, val_frac, test_frac, seed):

    # 1. Преобразуем данные в NumPy-массивы
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    # 2. Определяем количество объектов
    n = len(y)

    # 3. Создаём генератор случайных чисел
    rng = np.random.default_rng(seed)

    # 4. Перемешиваем индексы
    indices = rng.permutation(n)

    # 5. Вычисляем размеры выборок
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    # 6. Разделяем индексы
    train_idx = indices[:n_train]

    val_idx = indices[n_train:n_train + n_val]

    test_idx = indices[n_train + n_val:]

    # 7. Формируем обучающую выборку
    X_train = X[train_idx]
    y_train = y[train_idx]

    # 8. Формируем тестовую выборку
    X_test = X[test_idx]

    # 9. Добавляем столбец единиц (bias)
    X_train = np.column_stack([
        X_train,
        np.ones(len(X_train))
    ])

    X_test = np.column_stack([
        X_test,
        np.ones(len(X_test))
    ])

    # 10. Обучаем линейную регрессию методом OLS
    w = np.linalg.lstsq(
        X_train,
        y_train,
        rcond=None
    )[0]

    # 11. Получаем предсказания
    test_predictions = X_test @ w

    # 12. Возвращаем результаты
    return test_predictions, train_idx, val_idx, test_idx