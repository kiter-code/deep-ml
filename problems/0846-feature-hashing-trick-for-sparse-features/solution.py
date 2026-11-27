import hashlib

def feature_hash(features, n_features):
    # 1. Создаём выходной вектор, заполненный нулями
    result = [0.0] * n_features

    # 2. Перебираем названия признаков и их значения
    for name, value in features.items():

        # 3. Вычисляем MD5-хеш названия признака
        hash_value = hashlib.md5(name.encode('utf-8')).hexdigest()

        # 4. Берём первые 8 символов хеша
        # Переводим их в целое число и вычисляем индекс
        index = int(hash_value[:8], 16) % n_features

        # 5. Берём следующие 8 символов хеша
        sign_value = int(hash_value[8:16], 16)

        # 6. Определяем знак по чётности числа
        if sign_value % 2 == 0:
            sign = 1
        else:
            sign = -1

        # 7. Добавляем значение с учётом знака
        result[index] += sign * value

    # 8. Возвращаем итоговый вектор
    return result