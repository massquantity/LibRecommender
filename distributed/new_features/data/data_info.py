from collections import namedtuple
import numpy as np


Feature = namedtuple("Feature", ["name", "index"])


class DataInfo(namedtuple("DataInfo",
                          ["col_name_mapping", "interaction_data", "item_sparse_unique", "item_dense_unique"])):
    __slots__ = ()

    def __new__(cls, col_name_mapping=None, interaction_data=None, item_sparse_unique=None, item_dense_unique=None):
        return super(DataInfo, cls).__new__(cls, col_name_mapping, interaction_data, item_sparse_unique,
                                            item_dense_unique)

    @property
    def global_mean(self):
        return self.interaction_data.label.mean()

    @property
    def sparse_col(self):
        return Feature(name=list(self.col_name_mapping["sparse_col"].keys()),
                       index=list(self.col_name_mapping["sparse_col"].values()))

    @property
    def dense_col(self):
        return Feature(name=list(self.col_name_mapping["dense_col"].keys()),
                       index=list(self.col_name_mapping["dense_col"].values()))

    @property
    def user_sparse_col(self):
        return Feature(name=list(self.col_name_mapping["user_sparse_col"].keys()),
                       index=list(self.col_name_mapping["user_sparse_col"].values()))

    @property
    def user_dense_col(self):
        return Feature(name=list(self.col_name_mapping["user_dense_col"].keys()),
                       index=list(self.col_name_mapping["user_dense_col"].values()))

    @property
    def item_sparse_col(self):
        return Feature(name=list(self.col_name_mapping["item_sparse_col"].keys()),
                       index=list(self.col_name_mapping["item_sparse_col"].values()))

    @property
    def item_dense_col(self):
        return Feature(name=list(self.col_name_mapping["item_dense_col"].keys()),
                       index=list(self.col_name_mapping["item_dense_col"].values()))

    @property
    def user_col(self):
        # will be sorted by key
        return self.col_name_mapping["user_sparse_col"].keys().__or__(self.col_name_mapping["user_dense_col"].keys())

    @property
    def item_col(self):
        # will be sorted by key
        return self.col_name_mapping["item_sparse_col"].keys().__or__(self.col_name_mapping["item_dense_col"].keys())

    @property
    def n_users(self):
        return self.interaction_data.user.nunique()

    @property
    def n_items(self):
        return self.interaction_data.item.nunique()

    @property
    def user2id(self):
        unique = np.unique(self.interaction_data["user"])
        u2id = dict(zip(unique, range(self.n_users)))
        u2id[-1] = len(unique)   # -1 represent new user
        return u2id

    @property
    def item2id(self):
        unique = np.unique(self.interaction_data["item"])
        i2id = dict(zip(unique, range(self.n_items)))
        i2id[-1] = len(unique)  # -1 represent new item
        return i2id

    @property
    def id2user(self):
        return {j: user for user, j in self.user2id.items()}

    @property
    def id2item(self):
        return {j: item for item, j in self.item2id.items()}

    def __repr__(self):
        n_users = self.n_users
        n_items = self.n_items
        n_labels = len(self.interaction_data)
        return "n_users: %d, n_items: %d, data sparsity: %.2f %%" % (
            n_users, n_items, n_labels / (n_users*n_items) * 100)

    def get_indexed_interaction(self):
        data = self.interaction_data.copy()
        data.user = data.user.map(self.user2id)
        data.item = data.item.map(self.item2id)
        return data


"""
class DataInfo(namedtuple("DataInfo", ["a", "b"])):
    __slots__ = ()
    
    def __new__(cls, a, b):
        return super(DataInfo, cls).__new__(cls, a, b)
    
    def change1(self, aaa):
        return self._replace(a=aaa)
    
    def change2(self, bbb):
        return self._replace(b=bbb)
"""

"""
class DataInfo(namedtuple("DataInfo",
                          ["sparse_col", "dense_col", "user_col", "item_col", "interaction_data", "neg_item_unique",
                           "train_user_consumed", "train_item_consumed", "sparse_interaction"])):
    __slots__ = ()

    def __new__(cls, sparse_col, dense_col=None, user_col=None, item_col=None, interaction_data=None,
                neg_item_unique=None, train_user_consumed=None, train_item_consumed=None, sparse_interaction=None):
        return super(DataInfo, cls).__new__(cls, sparse_col, dense_col, user_col, item_col, interaction_data,
                                            neg_item_unique, train_user_consumed, train_item_consumed,
                                            sparse_interaction)

"""

"""

class DataInfo(namedtuple("DataInfo",
                          ["col_name_mapping", "interaction_data", "neg_item_unique",
                           "train_user_consumed", "train_item_consumed", "sparse_interaction"])):
    __slots__ = ()

    def __new__(cls, interaction_data=None, col_name_mapping=None, neg_item_unique=None, train_user_consumed=None,
                train_item_consumed=None, sparse_interaction=None):
        return super(DataInfo, cls).__new__(cls, interaction_data, col_name_mapping, neg_item_unique,
                                            train_user_consumed, train_item_consumed, sparse_interaction)

    def construct_neg_item_feat_mapping(self, train_data):
        neg_item_sparse_mapping = self._item_sparse_mapping(train_data)
        assert len(neg_item_sparse_mapping) == self.n_items, "number of unique negative items should match n_items"
        if train_data.dense_values is not None:
            neg_item_dense_mapping = self._item_dense_mapping(train_data)
            return self._replace(neg_item_unique=(neg_item_sparse_mapping, neg_item_dense_mapping))
        return self._replace(neg_item_unique=(neg_item_sparse_mapping, ))  ###### do not use _replace

    def _item_sparse_mapping(self, data):
        item_sparse_mapping = dict()
        # np.unique(axis=0) will sort the data based on first column, so we can do direct mapping
        all_items_sparse_unique = np.unique(data.sparse_indices[:, self.item_sparse_col], axis=0)
    #    all_items = all_items_sparse_unique[:, 0] - self.n_items - 1
        for item, feat in enumerate(all_items_sparse_unique):
            item_sparse_mapping[item] = feat.tolist()
        return item_sparse_mapping

    def _item_dense_mapping(self, data):
        assert len(data.sparse_indices) == len(data.dense_values), "length of sparse and dense columns must equal"
        item_dense_mapping = dict()
        item_indices = data.sparse_indices[:, 0].reshape(-1, 1)
        dense_values = data.dense_values.reshape(-1, 1) if data.dense_values.ndim == 1 else data.dense_values
        indices_plus_dense_values = np.concatenate([item_indices, dense_values], axis=-1)
        all_items_dense_unique = np.unique(indices_plus_dense_values, axis=0)
    #    all_items = all_items_dense_unique[:, 0]
        for item, feat in enumerate(all_items_dense_unique[:, 1:]):
            item_dense_mapping[item] = feat.tolist()
        return item_dense_mapping

    def interaction_consumed(self):
        pass

    def interaction_sparse(self):
        pass

"""
