import numpy as np

def k_fold_cross_validation(n_samples, k=5, shuffle=True):
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    folds = np.array_split(indices, k)
    result = []

    for i in range(k):
        test_indices = folds[i]

        train_indices = np.concatenate(folds[:i] + folds[i+1:])

        result.append((train_indices.tolist(), test_indices.tolist()))

    return result