import numpy as np
from array import array
from collections import defaultdict
from scipy.sparse import csr_matrix
from ..utils.sampling import NegativeSampling


class TransformedSet(object):
    def __init__(self, user_indices=None, item_indices=None, labels=None,
                 sparse_indices=None, dense_values=None, train=True):
        self._user_indices = user_indices
        self._item_indices = item_indices
        self._labels = labels
        self._sparse_indices = sparse_indices
        self._dense_values = dense_values
        self.has_sampled = False
        if train:
            self._sparse_interaction = csr_matrix(
                (labels, (user_indices, item_indices)),
                dtype=np.float32
            )

        (self._user_consumed,
         self._item_consumed) = self.__interaction_consumed()

        self.user_indices_orig = None
        self.item_indices_orig = None
        self.labels_orig = None
        self.sparse_indices_orig = None
        self.dense_values_orig = None

    def __interaction_consumed(self):
        user_consumed = defaultdict(lambda: array("I"))
        item_consumed = defaultdict(lambda: array("I"))
        for u, i in zip(self.user_indices, self.item_indices):
            user_consumed[u].append(i)
            item_consumed[i].append(u)
        return user_consumed, item_consumed

    def build_negative_samples(self, data_info, num_neg=1,
                               item_gen_mode="random", seed=42):
        self.has_sampled = True
        self.user_indices_orig = self._user_indices
        self.item_indices_orig = self._item_indices
        self.labels_orig = self._labels
        self.sparse_indices_orig = self._sparse_indices
        self.dense_values_orig = self._dense_values

        self._build_negative_samples(data_info, num_neg, item_gen_mode, seed)

    def _build_negative_samples(self, data_info, num_neg=1,
                                item_gen_mode="random", seed=42):

        if self.sparse_indices is None and self.dense_values is None:
            neg = NegativeSampling(self, data_info, num_neg,
                                   sparse=False, dense=False)
        elif self.sparse_indices is None:
            neg = NegativeSampling(self, data_info, num_neg,
                                   sparse=False, dense=True)
        elif self.dense_values is None:
            neg = NegativeSampling(self, data_info, num_neg,
                                   sparse=True, dense=False)
        else:
            neg = NegativeSampling(self, data_info, num_neg,
                                   sparse=True, dense=True)

        (self._user_indices, self._item_indices, self._labels,
         self._sparse_indices, self._dense_values
         ) = neg.generate_all(seed=seed, item_gen_mode=item_gen_mode)

    def __len__(self):
        return len(self.labels)

    @property
    def user_indices(self):
        return self._user_indices

    @property
    def item_indices(self):
        return self._item_indices

    @property
    def sparse_indices(self):
        return self._sparse_indices

#    @property
#    def dense_indices(self):
#        return self._dense_indices

    @property
    def dense_values(self):
        return self._dense_values

    @property
    def labels(self):
        return self._labels

    @property
    def sparse_interaction(self):
        return self._sparse_interaction

    @property
    def user_consumed(self):
        return self._user_consumed

    @property
    def item_consumed(self):
        return self._item_consumed

