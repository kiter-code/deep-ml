import numpy as np
from collections import Counter

def dummy_classifier(y_train, n_test, strategy, constant=None):

    # 1. Подсчитываем количество объектов каждого класса
    counts = Counter(y_train)

    # Получаем отсортированные уникальные классы
    classes = sorted(counts)

    # 2. Стратегия наиболее частого класса
    if strategy == "most_frequent":

        # При равной частоте выбираем наименьший класс
        most_frequent = max(classes, key=lambda c: counts[c])

        return [most_frequent] * n_test

    # 3. Стратегия постоянного значения
    elif strategy == "constant":

        return [constant] * n_test

    # 4. Стратегия равномерного распределения
    elif strategy == "uniform":

        k = len(classes)

        return [classes[i % k] for i in range(n_test)]

    # 5. Стратегия стратифицированного распределения
    elif strategy == "stratified":

        total = len(y_train)

        # Количество предсказаний для каждого класса
        allocation = {}

        # Остатки для распределения оставшихся предсказаний
        remainders = {}

        for c in classes:

            # Целая часть и остаток от деления
            allocation[c], remainders[c] = divmod(
                n_test * counts[c], total
            )

        # Сколько предсказаний осталось распределить
        remaining = n_test - sum(allocation.values())

        # Сортируем классы по убыванию остатка
        # При равенстве выбираем меньший класс
        ranked = sorted(
            classes,
            key=lambda c: (-remainders[c], c)
        )

        # Распределяем оставшиеся предсказания
        for c in ranked[:remaining]:
            allocation[c] += 1

        # Формируем итоговый список предсказаний
        result = []

        for c in classes:
            result.extend([c] * allocation[c])

        return result