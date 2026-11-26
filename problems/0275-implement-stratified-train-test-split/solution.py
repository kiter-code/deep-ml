import numpy as np

def stratified_train_test_split(X, y, test_size, random_seed=None):
    np.random.seed(random_seed)

    classes = np.unique(y)

    train_indices = []
    test_indices = []

    for c in classes:
        indices = np.where(y == c)[0]

        np.random.shuffle(indices)

        n_test = int(len(indices) * test_size)

        test_indices.extend(indices[:n_test])
        train_indices.extend(indices[n_test:])

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]