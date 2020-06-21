import numpy as np
from array import array
from collections import defaultdict
from functools import partial
from scipy.sparse import csr_matrix
from distributed.new_features.utils.samplingNEW import NegativeSamplingPure, NegativeSamplingFeat


class TransformedSet(object):
    def __init__(self, user_indices=None, item_indices=None, labels=None,
                 sparse_indices=None, dense_indices=None, dense_values=None,
                 train=True, feat=True):
        self._user_indices = user_indices
        self._item_indices = item_indices
        self._labels = labels
        self._sparse_indices = sparse_indices
        self._dense_indices = dense_indices
        self._dense_values = dense_values
        self.feat = feat
        self.has_sampled = False
        if train:
            self._sparse_interaction = csr_matrix(
                (labels, (user_indices, item_indices)), dtype=np.float32
            )

        (self._user_consumed,
         self._item_consumed) = self.__interaction_consumed()
   #     self.sparse_indices_sampled = None
   #     self.dense_indices_sampled = None
   #     self.dense_values_sampled = None
   #     self.label_samples = None

        self.user_indices_orig = None
        self.item_indices_orig = None
    #    self.labels_orig = None
        self.sparse_indices_orig = None
        self.dense_indices_orig = None
        self.dense_values_orig = None

    def __interaction_consumed(self):
        user_consumed = defaultdict(lambda: array("I"))
        item_consumed = defaultdict(lambda: array("I"))
        for u, i in zip(self.user_indices, self.item_indices):
            user_consumed[u].append(i)
            item_consumed[i].append(u)
        return user_consumed, item_consumed

    def build_negative_samples(self, data_info, num_neg=1,
                               mode="random", seed=42):
        self.has_sampled = True
        if not self.feat:
            self.user_indices_orig = self._user_indices
            self.item_indices_orig = self._item_indices
            self.build_negative_samples_pure(data_info, num_neg, mode, seed)
        else:
            self.sparse_indices_orig = self._sparse_indices
            self.dense_indices_orig = self._dense_indices
            self.dense_values_orig = self._dense_values
            self.build_negative_samples_feat(data_info, num_neg, mode, seed)

    def build_negative_samples_pure(self, data_info, num_neg=1,
                                    mode="random", seed=42):

        neg = NegativeSamplingPure(
            self, data_info, num_neg, batch_sampling=False)

        (self._user_indices, self._item_indices,
            self._labels) = neg.generate_all(seed, shuffle=False, mode=mode)

    def build_negative_samples_feat(self, data_info, num_neg=1,
                                    mode="random", seed=42):

        neg = NegativeSamplingFeat(self, data_info, num_neg)
        if self.dense_values is None:
            neg_generator = partial(neg.generate_all, dense=False)
        else:
            neg_generator = partial(neg.generate_all, dense=True)

        (self._sparse_indices, self._dense_indices,
            self._dense_values, self._labels) = neg_generator(seed, mode=mode)

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

    @property
    def dense_indices(self):
        return self._dense_indices

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

