from collections import namedtuple
import numpy as np


Feature = namedtuple("Feature", ["name", "index"])
Empty_Feature = Feature(name=[], index=[])


class DataInfo(namedtuple("DataInfo",
                          ["col_name_mapping", "interaction_data",
                           "user_sparse_unique", "user_dense_unique",
                           "item_sparse_unique", "item_dense_unique"])):
    __slots__ = ()

    def __new__(cls, col_name_mapping=None, interaction_data=None,
                user_sparse_unique=None, user_dense_unique=None,
                item_sparse_unique=None, item_dense_unique=None):
        return super(DataInfo, cls).__new__(
            cls, col_name_mapping, interaction_data, user_sparse_unique,
            user_dense_unique, item_sparse_unique, item_dense_unique
        )

    @property
    def global_mean(self):
        return self.interaction_data.label.mean()

    @property
    def min_max_rating(self):
        return (self.interaction_data.label.min(),
                self.interaction_data.label.max())

    @property
    def sparse_col(self):
        if not self.col_name_mapping["sparse_col"]:
            return Empty_Feature
        return Feature(
            name=list(self.col_name_mapping["sparse_col"].keys()),
            index=list(self.col_name_mapping["sparse_col"].values())
        )

    @property
    def dense_col(self):
        if not self.col_name_mapping["dense_col"]:
            return Empty_Feature
        return Feature(
            name=list(self.col_name_mapping["dense_col"].keys()),
            index=list(self.col_name_mapping["dense_col"].values())
        )

    @property
    def user_sparse_col(self):
        if not self.col_name_mapping["user_sparse_col"]:
            return Empty_Feature
        return Feature(
            name=list(self.col_name_mapping["user_sparse_col"].keys()),
            index=list(self.col_name_mapping["user_sparse_col"].values())
        )

    @property
    def user_dense_col(self):
        if not self.col_name_mapping["user_dense_col"]:
            return Empty_Feature
        return Feature(
            name=list(self.col_name_mapping["user_dense_col"].keys()),
            index=list(self.col_name_mapping["user_dense_col"].values())
        )

    @property
    def item_sparse_col(self):
        if not self.col_name_mapping["item_sparse_col"]:
            return Empty_Feature
        return Feature(
            name=list(self.col_name_mapping["item_sparse_col"].keys()),
            index=list(self.col_name_mapping["item_sparse_col"].values())
        )

    @property
    def item_dense_col(self):
        if not self.col_name_mapping["item_dense_col"]:
            return Empty_Feature
        return Feature(
            name=list(self.col_name_mapping["item_dense_col"].keys()),
            index=list(self.col_name_mapping["item_dense_col"].values())
        )

    @property
    def user_col(self):
        # will be sorted by key
        return self.col_name_mapping["user_sparse_col"].keys().__or__(
            self.col_name_mapping["user_dense_col"].keys())

    @property
    def item_col(self):
        # will be sorted by key
        return self.col_name_mapping["item_sparse_col"].keys().__or__(
            self.col_name_mapping["item_dense_col"].keys())

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
        return "n_users: %d, n_items: %d, data sparsity: %.4f %%" % (
            n_users, n_items, 100 * n_labels / (n_users*n_items))

    def get_indexed_interaction(self):
        data = self.interaction_data.copy()
        data.user = data.user.map(self.user2id)
        data.item = data.item.map(self.item2id)
        return data

