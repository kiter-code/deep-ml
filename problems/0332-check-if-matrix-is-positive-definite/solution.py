import torch

def check_positive_definite(matrix: list) -> dict:

    # 1. Преобразуем список в тензор
    A = torch.as_tensor(matrix, dtype=torch.float64)

    # 2. Проверяем симметричность матрицы
    is_symmetric = torch.allclose(A, A.T)

    # 3. Вычисляем собственные значения
    if is_symmetric:
        # Для симметричных матриц
        eigenvalues = torch.linalg.eigvalsh(A)
    else:
        # Для несимметричных матриц
        # Берём только действительные части
        eigenvalues = torch.linalg.eigvals(A).real

    # 4. Сортируем значения по возрастанию
    eigenvalues = torch.sort(eigenvalues).values

    # 5. Проверяем, что все значения положительные
    is_positive = torch.all(eigenvalues > 1e-10).item()

    # 6. Проверяем оба условия одновременно
    is_positive_definite = is_symmetric and is_positive

    # 7. Округляем собственные значения
    eigenvalues = [
        round(v, 4) for v in eigenvalues.tolist()
    ]

    # 8. Возвращаем результат
    return {
        "is_positive_definite": is_positive_definite,
        "eigenvalues": eigenvalues
    }