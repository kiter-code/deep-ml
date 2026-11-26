import numpy as np

def standard_scaler(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    std = np.where(std == 0, 1.0, std)

    return (X_test - mean) / std