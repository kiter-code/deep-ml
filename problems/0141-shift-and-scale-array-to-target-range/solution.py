import numpy as np

def convert_range(values: np.ndarray, c: float, d: float) -> np.ndarray:
    return c + ((d - c) / (np.max(values) - np.min(values))) * (values - np.min(values))