import numpy as np

def ordinal_encode(X, categories):
    ordinal_list = []

    for column in categories:
        d = {}

        for index, value in enumerate(column):
            d[value] = index

        ordinal_list.append(d)

    result = []

    for i in range(len(X)):
        row = []

        for j in range(len(X[i])):
            value = ordinal_list[j].get(X[i][j], -1)
            row.append(value)

        result.append(row)

    return np.array(result, dtype=int).reshape(len(X), len(categories))