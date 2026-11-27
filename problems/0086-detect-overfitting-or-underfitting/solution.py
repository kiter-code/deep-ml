def model_fit_quality(training_accuracy, test_accuracy):

    difference = training_accuracy - test_accuracy

    # 1. Проверяем переобучение
    if difference > 0.2:
        return 1

    # 2. Проверяем недообучение
    elif training_accuracy < 0.7 and test_accuracy < 0.7:
        return -1

    # 3. Хорошее обучение
    else:
        return 0