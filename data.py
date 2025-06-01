import numpy as np

class Dataset:
    def __init__(self, x: np.ndarray, y: np.ndarray, batch_size: int = 32, shuffle: bool = True, drop_last: bool = False):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.data_indexes = np.arange(len(self.x))


    def __iter__(self):
        data_size = len(self.x)
        if data_size != len(self.data_indexes):
            self.data_indexes = np.arange(data_size)

        if self.shuffle:
            np.random.shuffle(self.data_indexes)

        for batch_start in range(0, data_size, self.batch_size):
            batch_end = batch_start + self.batch_size

            if self.drop_last and batch_end > data_size:
                break

            batch_indexes = self.data_indexes[batch_start:batch_end]  # bierze wycinki z array z indeksami

            yield self.x[batch_indexes], self.y[batch_indexes]

    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
