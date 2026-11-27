# --- cell 1 ---
import numpy as np

def train(X, y, W, b):
    """
    Train linear regression weights on standardized data.
    
    Args:
        X: numpy array of shape (n_samples, n_features) -- standardized features
        y: numpy array of shape (n_samples,) -- standardized targets
        W: numpy array of shape (n_features,) -- initial random weights
        b: float -- initial bias (0.0)
    
    Returns:
        W: numpy array of shape (n_features,) -- trained weights
        b: float -- trained bias
    """
    lr = 0.01
    epochs = 500
    n = X.shape[0]
    for _ in range(epochs):
        preds = X @ W + b
        errors = preds - y
        grad_W = (2 / n) * (X.T @ errors)
        grad_b = (2 / n) * np.sum(errors)
        W = W - lr * grad_W
        b = b - lr * grad_b

    return W, b