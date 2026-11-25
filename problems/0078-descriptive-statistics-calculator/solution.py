import numpy as np

def descriptive_statistics(data: list | np.ndarray) -> dict:

    values, counts = np.unique(data, return_counts=True)
    mode = values[np.argmax(counts)]

    return {
        'mean': float(np.mean(data)),
        'median': float(np.median(data)),
        'mode': int(mode),
        'variance': float(np.var(data)),
        'standard_deviation': float(np.std(data)),
        '25th_percentile': float(np.percentile(data, 25)),
        '50th_percentile': float(np.percentile(data, 50)),
        '75th_percentile': float(np.percentile(data, 75)),
        'interquartile_range': float(
            np.percentile(data, 75) - np.percentile(data, 25)
        )
    }