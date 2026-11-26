import numpy as np

def shuffle_data(X, y, seed=None):
    if seed != None:
        seed = np.random.seed(seed)
    
    n = X.shape[0] # kolvo strok
    index_rand = np.random.permutation(n) # рандомное перемешкиванье индексов

    return (X[index_rand], y[index_rand])
