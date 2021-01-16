from array import array
from collections import defaultdict, namedtuple
import inspect
import json
import os
import numpy as np
import pandas as pd


Feature = namedtuple("Feature", ["name", "index"])
Empty_Feature = Feature(name=[], index=[])


class DataInfo(object):
    def __init__(
            self,
            col_name_mapping=None,
            interaction_data=None,
            user_sparse_unique=None,
            user_dense_unique=None,
            item_sparse_unique=None,
            item_dense_unique=None,
            user_indices=None,
            item_indices=None
    ):
        self.col_name_mapping = col_name_mapping
        self.interaction_data = interaction_data
        self.user_sparse_unique = user_sparse_unique
        self.user_dense_unique = user_dense_unique
        self.item_sparse_unique = item_sparse_unique
        self.item_dense_unique = item_dense_unique
        self.user_consumed, self.item_consumed = DataInfo.interaction_consumed(
            user_indices, item_indices
        )
        self._user2id = None
        self._item2id = None
        self._id2user = None
        self._id2item = None
        self.all_args = locals()

    @staticmethod
    def interaction_consumed(user_indices, item_indices):
        user_consumed = defaultdict(lambda: array("I"))
        item_consumed = defaultdict(lambda: array("I"))
        for u, i in zip(user_indices, item_indices):
            user_consumed[u].append(i)
            item_consumed[i].append(u)
        return user_consumed, item_consumed

    @property
    def global_mean(self):
        return self.interaction_data.label.mean()

    @property
    def min_max_rating(self):
        return (
            self.interaction_data.label.min(),
            self.interaction_data.label.max()
        )

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
        return (
            self.col_name_mapping["user_sparse_col"].keys().__or__(
                self.col_name_mapping["user_dense_col"].keys())
        )

    @property
    def item_col(self):
        # will be sorted by key
        return (
            self.col_name_mapping["item_sparse_col"].keys().__or__(
                self.col_name_mapping["item_dense_col"].keys())
        )

    @property
    def n_users(self):
        return self.interaction_data.user.nunique()

    @property
    def n_items(self):
        return self.interaction_data.item.nunique()

    @property
    def user2id(self):
        if self._user2id is None:
            unique = np.unique(self.interaction_data["user"])
            self._user2id = dict(zip(unique, range(self.n_users)))
            self._user2id[-1] = len(unique)   # -1 represent new user
        return self._user2id

    @property
    def item2id(self):
        if self._item2id is None:
            unique = np.unique(self.interaction_data["item"])
            self._item2id = dict(zip(unique, range(self.n_items)))
            self._item2id[-1] = len(unique)  # -1 represent new item
        return self._item2id

    @property
    def id2user(self):
        if self._id2user is None:
            self._id2user = {j: user for user, j in self.user2id.items()}
        return self._id2user

    @property
    def id2item(self):
        if self._id2item is None:
            self._id2item = {j: item for item, j in self.item2id.items()}
        return self._id2item

    def __repr__(self):
        n_users = self.n_users
        n_items = self.n_items
        n_labels = len(self.interaction_data)
        return "n_users: %d, n_items: %d, data sparsity: %.4f %%" % (
            n_users, n_items, 100 * n_labels / (n_users*n_items)
        )

    def get_indexed_interaction(self):
        data = self.interaction_data.copy()
        data.user = data.user.map(self.user2id)
        data.item = data.item.map(self.item2id)
        if data.user.isnull().any():
            data["user"].fillna(self.n_users, inplace=True)
            data["user"] = data["user"].astype("int")
        if data.item.isnull().any():
            data["item"].fillna(self.n_items, inplace=True)
            data["item"] = data["item"].astype("int")
        return data

    def save(self, path):
        if not os.path.isdir(path):
            print(f"file folder {path} doesn't exists, creating a new one...")
            os.makedirs(path)
        name_mapping_path = os.path.join(path, "data_info_name_mapping.json")
        with open(name_mapping_path, 'w') as f:
            json.dump(self.all_args["col_name_mapping"],
                      f, separators=(',', ':'))

        other_path = os.path.join(path, "data_info")
        hparams = dict()
        arg_names = inspect.signature(self.__init__).parameters.keys()
        for arg in arg_names:
            if arg == "col_name_mapping" or self.all_args[arg] is None:
                continue
            if arg == "interaction_data":
                hparams[arg] = self.all_args[arg].to_numpy()
            else:
                hparams[arg] = self.all_args[arg]

        np.savez_compressed(other_path, **hparams)

    # noinspection PyTypeChecker
    @classmethod
    def load(cls, path):
        if not os.path.exists(path):
            raise OSError(f"file folder {path} doesn't exists...")

        hparams = dict()
        name_mapping_path = os.path.join(path, "data_info_name_mapping.json")
        with open(name_mapping_path, 'r') as f:
            hparams["col_name_mapping"] = json.load(f)

        other_path = os.path.join(path, "data_info.npz")
        info = np.load(other_path)
        for arg in info:
            if arg == "interaction_data":
                hparams[arg] = pd.DataFrame(
                    info[arg], columns=["user", "item", "label"])
            else:
                hparams[arg] = info[arg]

        return cls(**hparams)
