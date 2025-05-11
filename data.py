import numpy as np

# TODO __iter__ with shuffling and batching
# shuffle can be a separate function

class Dataset:
    def __init__(self, x: np.ndarray, y: np.ndarray, batch_size: int = 32, shuffle: bool = True, drop_last: bool = False):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    ...
