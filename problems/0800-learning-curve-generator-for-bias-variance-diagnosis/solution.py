import numpy as np

def learning_curve(X_train, y_train, X_val, y_val,
                   train_sizes, degree,
                   bias_threshold=0.5,
                   variance_threshold=0.5):

    # 1. Преобразуем данные в массивы NumPy
    X_train = np.array(X_train, dtype=float).ravel()
    X_val = np.array(X_val, dtype=float).ravel()
    y_train = np.array(y_train, dtype=float).ravel()
    y_val = np.array(y_val, dtype=float).ravel()

    train_errors = []
    val_errors = []

    # 2. Создаём полиномиальные признаки
    # Например: [1, x, x^2, x^3]
    Phi_train = np.column_stack([
        X_train ** i for i in range(degree + 1)
    ])

    Phi_val = np.column_stack([
        X_val ** i for i in range(degree + 1)
    ])

    # 3. Перебираем размеры обучающих выборок
    for n in train_sizes:

        # Берём первые n объектов
        X_subset = Phi_train[:n]
        y_subset = y_train[:n]

        # 4. Вычисляем веса через псевдообратную матрицу
        w = np.linalg.pinv(X_subset) @ y_subset

        # 5. Получаем предсказания
        train_pred = X_subset @ w
        val_pred = Phi_val @ w

        # 6. Вычисляем MSE
        train_mse = np.mean((y_subset - train_pred) ** 2)
        val_mse = np.mean((y_val - val_pred) ** 2)

        # 7. Сохраняем ошибки
        train_errors.append(float(train_mse))
        val_errors.append(float(val_mse))

    # 8. Берём последние значения ошибок
    train_error = train_errors[-1]
    val_error = val_errors[-1]

    # 9. Определяем состояние модели
    if train_error > bias_threshold:
        diagnosis = "high_bias"

    elif val_error - train_error > variance_threshold:
        diagnosis = "high_variance"

    else:
        diagnosis = "good_fit"

    # 10. Возвращаем результат
    return {
        "train_errors": train_errors,
        "val_errors": val_errors,
        "diagnosis": diagnosis
    }