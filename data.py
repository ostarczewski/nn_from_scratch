import numpy as np
from collections.abc import Callable

class Dataset:
    def __init__(self, x: np.ndarray, y: np.ndarray, batch_size: int = 32, shuffle: bool = True, drop_last: bool = False,
                 transform: Callable = None):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.data_indexes = np.arange(len(self.x))
        self.transform = transform


    def __iter__(self):
        # always keep the dataset size updated
        data_size = len(self.x)
        if data_size != len(self.data_indexes):
            self.data_indexes = np.arange(data_size)

        # shuffles the indexes instead of dataset shuffling, more efficient and doesn't change the dataset
        if self.shuffle:
            np.random.shuffle(self.data_indexes)

        for batch_start in range(0, data_size, self.batch_size):
            batch_end = batch_start + self.batch_size

            if self.drop_last and batch_end > data_size:
                break

            batch_indexes = self.data_indexes[batch_start:batch_end]  # bierze wycinki z array z indeksami

            X_batch = self.x[batch_indexes]
            y_batch = self.y[batch_indexes]

            if self.transform is not None:
                X_batch = self.transform(X_batch)

            yield X_batch, y_batch

    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def get_batch_num(self):
        if self.drop_last:
            return len(self.x) // self.batch_size
        else:
            return -(-len(self.x) // self.batch_size)
