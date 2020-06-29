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


class DataGenFeat(object):
    def __init__(self, data, batch_size, dense, class_name=None):
        self.data_size = len(data)
    #    self.user_indices = data.user_indices
    #    self.item_indices = data.item_indices
        self.labels = data.labels
        self.sparse_indices = data.sparse_indices
        self.dense_indices = data.dense_indices
        self.dense_values = data.dense_values
        self.batch_size = batch_size
        self.class_name = class_name
        self.dense = dense

    def __iter__(self):
        for i in range(0, self.data_size, self.batch_size):
            batch_slice = slice(i, i + self.batch_size)
            if self.dense:
                yield(
                    self.sparse_indices[batch_slice],
                    self.dense_indices[batch_slice],
                    self.dense_values[batch_slice],
                    self.labels[batch_slice]
                )
            else:
                yield(
                    self.sparse_indices[batch_slice],
                    None,
                    None,
                    self.labels[batch_slice]
                )

    def __call__(self, shuffle=True):
        if shuffle:
            mask = np.random.permutation(range(self.data_size))
            if self.dense:
                self.sparse_indices = self.sparse_indices[mask]
                self.dense_indices = self.dense_indices[mask]
                self.dense_values = self.dense_values[mask]
                self.labels = self.labels[mask]
            else:
                self.sparse_indices = self.sparse_indices[mask]
                self.labels = self.labels[mask]

        return self


class DataGenYoutube(object):
    def __init__(self, data, batch_size, dense, mode, num_neg=1):
        if mode == "match":
            if data.has_sampled:
                self.user_indices = data.user_indices_orig
                self.item_indices = data.item_indices_orig
            else:
                self.user_indices = data.user_indices
                self.item_indices = data.item_indices
        elif mode == "ranking":
            self.user_indices = data.user_indices
            # no need for item_indices, just for consistency's sake
            self.item_indices = np.zeros_like(self.user_indices)
        else:
            raise ValueError("mode must either be 'match' or 'ranking'")
        self.data_size = len(self.user_indices)
        self.sparse_indices = data.sparse_indices
        self.dense_indices = data.dense_indices
        self.dense_values = data.dense_values
        self.labels = data.labels
        self.batch_size = batch_size
        self.dense = dense

    def __iter__(self):
        for i in range(0, self.data_size, self.batch_size):
            batch_slice = slice(i, i + self.batch_size)
            if self.dense:
                yield(
                    self.user_indices[batch_slice],
                    self.item_indices[batch_slice],
                    self.sparse_indices[batch_slice],
                    self.dense_indices[batch_slice],
                    self.dense_values[batch_slice],
                    self.labels[batch_slice],
                )
            else:
                yield(
                    self.user_indices[batch_slice],
                    self.item_indices[batch_slice],
                    self.sparse_indices[batch_slice],
                    None,
                    None,
                    self.labels[batch_slice],
                )

    def __call__(self, shuffle=True):
        if shuffle:
            mask = np.random.permutation(range(self.data_size))
            if self.dense:
                self.user_indices = self.user_indices[mask]
                self.item_indices = self.item_indices[mask]
                self.sparse_indices = self.sparse_indices[mask]
                self.dense_indices = self.dense_indices[mask]
                self.dense_values = self.dense_values[mask]
                self.labels = self.labels[mask]
            else:
                self.user_indices = self.user_indices[mask]
                self.item_indices = self.item_indices[mask]
                self.sparse_indices = self.sparse_indices[mask]
                self.labels = self.labels[mask]

        return self


