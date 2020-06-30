import numpy as np
from .sparse import sparse_indices_and_values


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
    def __init__(self, data, sparse, dense, class_name=None):
        self.user_indices = data.user_indices
        self.item_indices = data.item_indices
        self.labels = data.labels
        self.sparse_indices = data.sparse_indices
        self.dense_indices = data.dense_indices
        self.dense_values = data.dense_values
        self.sparse = sparse
        self.dense = dense
        self.data_size = len(data)
        self.class_name = class_name

    def __iter__(self, batch_size):
        for i in range(0, self.data_size, batch_size):
            batch_slice = slice(i, i + batch_size)
            res = (
                self.user_indices[batch_slice],
                self.item_indices[batch_slice],
                self.labels[batch_slice]
            )
            if self.sparse and self.dense:
                res_other = (
                    self.sparse_indices[batch_slice],
                    self.dense_indices[batch_slice],
                    self.dense_values[batch_slice]
                )
            elif self.sparse:
                res_other =  (
                    self.sparse_indices[batch_slice],
                    None,
                    None
                )
            elif self.dense:
                res_other = (
                    None,
                    self.dense_indices[batch_slice],
                    self.dense_values[batch_slice]
                )
            else:
                res_other = (
                    None,
                    None,
                    None
                )
            yield res + res_other

    def __call__(self, shuffle=True, batch_size=None):
        if shuffle:
            mask = np.random.permutation(range(self.data_size))
            if self.sparse:
                self.sparse_indices = self.sparse_indices[mask]
            if self.dense:
                self.dense_indices = self.dense_indices[mask]
                self.dense_values = self.dense_values[mask]
            self.user_indices = self.user_indices[mask]
            self.item_indices = self.item_indices[mask]
            self.labels = self.labels[mask]

        return self.__iter__(batch_size)


class DataGenSequence(object):
    def __init__(self, data, sparse, dense, recent_num=10,
                 random_num=None, model=None):
        if model == "YoutubeMatch" and data.has_sampled:
            self.user_indices = data.user_indices_orig
            self.item_indices = data.item_indices_orig
            self.labels = data.labels_orig
            self.sparse_indices = data.sparse_indices_orig
            self.dense_indices = data.dense_indices_orig
            self.dense_values = data.dense_values_orig
        else:
            self.user_indices = data.user_indices
            self.item_indices = data.item_indices
            self.labels = data.labels
            self.sparse_indices = data.sparse_indices
            self.dense_indices = data.dense_indices
            self.dense_values = data.dense_values
        self.sparse = sparse
        self.dense = dense
        self.data_size = len(self.user_indices)
        self.recent_num = recent_num
        self.random_num = random_num
        self.user_consumed = data.user_consumed
    #    self.zero_item = data.n_items

    def __iter__(self, batch_size):
        for i in range(0, self.data_size, batch_size):
            batch_slice = slice(i, i + batch_size)
            (interacted_indices,
             interacted_values,
             modified_batch_size) = sparse_indices_and_values(
                self.user_indices[batch_slice],
                self.item_indices[batch_slice],
                self.user_consumed,
                self.recent_num,
                self.random_num
            )
            res = (
                modified_batch_size,
                interacted_indices,
                interacted_values,
                self.user_indices[batch_slice],
                self.item_indices[batch_slice],
                self.labels[batch_slice]
            )
            if self.sparse and self.dense:
                res_other = (
                    self.sparse_indices[batch_slice],
                    self.dense_indices[batch_slice],
                    self.dense_values[batch_slice]
                )
            elif self.sparse:
                res_other = (
                    self.sparse_indices[batch_slice],
                    None,
                    None
                )
            elif self.dense:
                res_other = (
                    None,
                    self.dense_indices[batch_slice],
                    self.dense_values[batch_slice]
                )
            else:
                res_other = (
                    None,
                    None,
                    None
                )
            yield res + res_other

    def __call__(self, shuffle=True, batch_size=None):
        if shuffle:
            mask = np.random.permutation(range(self.data_size))
            if self.sparse:
                self.sparse_indices = self.sparse_indices[mask]
            if self.dense:
                self.dense_indices = self.dense_indices[mask]
                self.dense_values = self.dense_values[mask]
            self.user_indices = self.user_indices[mask]
            self.item_indices = self.item_indices[mask]
            self.labels = self.labels[mask]

        return self.__iter__(batch_size)


