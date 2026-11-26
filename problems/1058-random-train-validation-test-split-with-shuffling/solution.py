import numpy as np

def random_split(data: np.ndarray, train_frac: float, validation_frac: float, seed: int = 123) -> list:
    """
    Randomly split a dataset into train, validation, and test subsets.
    """
    n = data.shape[0] # определили размерность массива

    index_rand = np.random.default_rng(seed).permutation(n) # рандомно перемешали индексы

    train_end = int(n * train_frac)
    validation_end = train_end + int(n * validation_frac)

    shuffled_data = data[index_rand] # перемешали данные

    train = shuffled_data[:train_end] # разделили срезами данные на train/val/test
    validation = shuffled_data[train_end:validation_end]
    test = shuffled_data[validation_end:]

    return [train, validation, test]