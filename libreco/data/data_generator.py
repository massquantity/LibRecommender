import numpy as np
import tqdm
from .sequence import sparse_user_interacted, user_interacted_seq


class DataGenPure(object):
    def __init__(self, data):
        self.data_size = len(data)
        self.user_indices = data.user_indices
        self.item_indices = data.item_indices
        self.labels = data.labels

    def __iter__(self, batch_size):
        for i in tqdm.trange(0, self.data_size, batch_size, desc="train"):
            batch_slice = slice(i, i + batch_size)
            yield (
                self.user_indices[batch_slice],
                self.item_indices[batch_slice],
                self.labels[batch_slice],
                None,
                None
            )

    def __call__(self, shuffle=True, batch_size=None):
        if shuffle:
            mask = np.random.permutation(range(self.data_size))
            self.user_indices = self.user_indices[mask]
            self.item_indices = self.item_indices[mask]
            self.labels = self.labels[mask]
        return self.__iter__(batch_size)


class DataGenFeat(object):
    def __init__(self, data, sparse, dense, class_name=None):
        self.user_indices = data.user_indices
        self.item_indices = data.item_indices
        self.labels = data.labels
        self.sparse_indices = data.sparse_indices
        self.dense_values = data.dense_values
        self.sparse = sparse
        self.dense = dense
        self.data_size = len(data)
        self.class_name = class_name

    def __iter__(self, batch_size):
        for i in tqdm.trange(0, self.data_size, batch_size, desc="train"):
            batch_slice = slice(i, i + batch_size)
            pure_part = (
                self.user_indices[batch_slice],
                self.item_indices[batch_slice],
                self.labels[batch_slice]
            )
            sparse_part = (
                (self.sparse_indices[batch_slice],)
                if self.sparse
                else (None,)
            )
            dense_part = (
                (self.dense_values[batch_slice],)
                if self.dense
                else (None,)
            )
            yield pure_part + sparse_part + dense_part

    def __call__(self, shuffle=True, batch_size=None):
        if shuffle:
            mask = np.random.permutation(range(self.data_size))
            if self.sparse:
                self.sparse_indices = self.sparse_indices[mask]
            if self.dense:
                self.dense_values = self.dense_values[mask]
            self.user_indices = self.user_indices[mask]
            self.item_indices = self.item_indices[mask]
            self.labels = self.labels[mask]

        return self.__iter__(batch_size)


class DataGenSequence(object):
    def __init__(self, data, data_info, sparse, dense, mode=None, num=None,
                 class_name=None, padding_idx=None):
        self.user_consumed = data_info.user_consumed
        self.padding_idx = padding_idx
        self.class_name = class_name
        if class_name == "YoutubeMatch" and data.has_sampled:
            self.user_indices = data.user_indices_orig
            self.item_indices = data.item_indices_orig
            self.labels = data.labels_orig
            self.sparse_indices = data.sparse_indices_orig
            self.dense_values = data.dense_values_orig
        else:
            self.user_indices = data.user_indices
            self.item_indices = data.item_indices
            self.labels = data.labels
            self.sparse_indices = data.sparse_indices
            self.dense_values = data.dense_values
            self.user_consumed_set = {
                u: set(items) for u, items in self.user_consumed.items()
            }
        self.data_size = len(self.user_indices)
        self.sparse = sparse
        self.dense = dense
        self.mode = mode
        self.num = num

    def __iter__(self, batch_size):
        for i in tqdm.trange(0, self.data_size, batch_size, desc="train"):
            batch_slice = slice(i, i + batch_size)
            if self.class_name == "YoutubeMatch":
                (
                    interacted_indices,
                    interacted_values,
                    modified_batch_size
                ) = sparse_user_interacted(
                    self.user_indices[batch_slice],
                    self.item_indices[batch_slice],
                    self.user_consumed,
                    self.mode,
                    self.num
                )
                pure_part = (
                    modified_batch_size,
                    interacted_indices,
                    interacted_values,
                    self.user_indices[batch_slice],
                    self.item_indices[batch_slice],
                    self.labels[batch_slice]
                )
            else:
                (
                    batch_interacted,
                    batch_interacted_len
                ) = user_interacted_seq(
                    self.user_indices[batch_slice],
                    self.item_indices[batch_slice],
                    self.user_consumed,
                    self.padding_idx,
                    self.mode,
                    self.num,
                    self.user_consumed_set
                )
                pure_part = (
                    batch_interacted,
                    batch_interacted_len,
                    self.user_indices[batch_slice],
                    self.item_indices[batch_slice],
                    self.labels[batch_slice]
                )

            sparse_part = (
                (self.sparse_indices[batch_slice],)
                if self.sparse
                else (None,)
            )
            dense_part = (
                (self.dense_values[batch_slice],)
                if self.dense
                else (None,)
            )
            yield pure_part + sparse_part + dense_part

    def __call__(self, shuffle=True, batch_size=None):
        if shuffle:
            mask = np.random.permutation(range(self.data_size))
            if self.sparse:
                self.sparse_indices = self.sparse_indices[mask]
            if self.dense:
                self.dense_values = self.dense_values[mask]
            self.user_indices = self.user_indices[mask]
            self.item_indices = self.item_indices[mask]
            self.labels = self.labels[mask]

        return self.__iter__(batch_size)

