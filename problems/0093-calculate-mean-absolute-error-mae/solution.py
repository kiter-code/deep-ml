import numpy as np

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))