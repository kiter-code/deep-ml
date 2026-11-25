import numpy as np

def grouped_cv(values, groups):
    values = np.array(values)
    groups = np.array(groups)

    result = {}

    for group in np.unique(groups):
        data = values[groups == group]

        n = len(data)
        x = np.mean(data)

        if n < 2 or x == 0:
            result[group] = None
        else:
            s = np.std(data, ddof=1)
            CV = (1 + 1/(4*n)) * (s/abs(x))

            result[group] = float(CV)

    return result