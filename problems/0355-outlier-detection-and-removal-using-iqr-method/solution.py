import numpy as np

def detect_outliers_iqr(data: list[float], k: float = 1.5) -> dict:
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR

    cleaned_data = []
    outlier_indices = []

    for index, value in enumerate(data):
        if value > upper_bound or value < lower_bound:
            # Добавь индекс выброса в нужный список
            outlier_indices.append(index)
        else:
            # Добавь нормальное значение в нужный список
            cleaned_data.append(round(value, 4))

    return {
    'cleaned_data': cleaned_data,
    'outlier_indices': outlier_indices,
    'lower_bound': lower_bound,
    'upper_bound': upper_bound
} 