import numpy as np

def r_squared(y_true, y_pred):

    # Среднее реальных значений
    mean = np.mean(y_true)

    # Сумма квадратов ошибок модели
    ss_res = np.nansum((y_true - y_pred) ** 2)

    # Сумма квадратов отклонений от среднего
    ss_tot = np.nansum((y_true - np.nanmean(y_true)) ** 2)


    # Коэффициент детерминации
    return 1 - ss_res/ss_tot