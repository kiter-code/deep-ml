import numpy as np

def stratified_kfold_indices(y, n_splits):
    y = np.asarray(y)
    classes = np.unique(y)
    indices = np.arange(len(y))

    class_folds = []
    result = []

    for c in classes:
        class_indices = np.where(y == c)[0]
        folds = np.array_split(class_indices, n_splits)
        class_folds.append(folds)

    for i in range(n_splits):
        test_indices = np.concatenate(
            [folds[i] for folds in class_folds]
        )

        test_indices = np.sort(test_indices)
        train_indices = np.setdiff1d(indices, test_indices)

        result.append([
            train_indices.tolist(),
            test_indices.tolist()
        ])

    return result