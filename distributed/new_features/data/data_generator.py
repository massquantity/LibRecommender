import numpy as np


class DataGenPure(object):
    def __init__(self, data, batch_size):
        self.data_size = len(data)
        self.user_indices = data.user_indices
        self.item_indices = data.item_indices
        self.labels = data.labels
        self.batch_size = batch_size

    def __iter__(self):
        for i in range(0, self.data_size, self.batch_size):
            batch_slice = slice(i, i + self.batch_size)
            yield (
                self.user_indices[batch_slice],
                self.item_indices[batch_slice],
                self.labels[batch_slice]
            )

    def __call__(self, shuffle=True):
        if shuffle:
            mask = np.random.permutation(range(self.data_size))
            self.user_indices = self.user_indices[mask]
            self.item_indices = self.item_indices[mask]
            self.labels = self.labels[mask]
        return self


