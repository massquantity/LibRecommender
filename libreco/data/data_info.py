from collections import namedtuple
import inspect
import json
import os
import numpy as np
import pandas as pd
from ..feature import (
    interaction_consumed,
    compute_sparse_feat_indices,
    _check_oov
)


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
            item_indices=None,
            user_unique_vals=None,
            item_unique_vals=None,
            sparse_unique_vals=None,
            sparse_offset=None,
            sparse_oov=None,
    ):
        self.col_name_mapping = col_name_mapping
        self.interaction_data = interaction_data
        self.user_sparse_unique = user_sparse_unique
        self.user_dense_unique = user_dense_unique
        self.item_sparse_unique = item_sparse_unique
        self.item_dense_unique = item_dense_unique
        self.user_consumed, self.item_consumed = interaction_consumed(
            user_indices, item_indices
        )
        self.user_unique_vals = user_unique_vals
        self.item_unique_vals = item_unique_vals
        self.sparse_unique_vals = sparse_unique_vals
        self.sparse_unique_idxs = self.map_unique_vals(sparse_unique_vals)
        self.sparse_offset = sparse_offset
        self.sparse_oov = sparse_oov
        self._n_users = None
        self._n_items = None
        self._user2id = None
        self._item2id = None
        self._id2user = None
        self._id2item = None
        self._data_size = None
        self.popular_items = None
        # store old sparse len and oov
        self.old_sparse_len = None
        self.old_sparse_oov = None
        self.all_args = locals()
        self.add_oov()
        if self.popular_items is None:
            self.set_popular_items(100)

    def map_unique_vals(self, sparse_unique_vals):
        if sparse_unique_vals is None:
            return None
        res = dict()
        for col in sparse_unique_vals:
            vals = sparse_unique_vals[col]
            size = len(vals)
            res[col] = dict(zip(vals, range(size)))
        return res

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
        if self._n_users is None:
            self._n_users = len(self.user_unique_vals)
        return self._n_users

    @property
    def n_items(self):
        if self._n_items is None:
            self._n_items = len(self.item_unique_vals)
        return self._n_items

    @property
    def user2id(self):
        if self._user2id is None:
            self._user2id = dict(
                zip(self.user_unique_vals, range(self.n_users))
            )
        return self._user2id

    @property
    def item2id(self):
        if self._item2id is None:
            self._item2id = dict(
                zip(self.item_unique_vals, range(self.n_items))
            )
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

    @property
    def data_size(self):
        if self._data_size is None:
            self._data_size = len(self.interaction_data)
        return self._data_size

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

    def update_consumed(self, user_indices, item_indices, merge):
        if merge:
            old_data = self.get_indexed_interaction()
            user_indices = np.append(old_data.user.to_numpy(), user_indices)
            item_indices = np.append(old_data.item.to_numpy(), item_indices)
        self.user_consumed, self.item_consumed = interaction_consumed(
            user_indices, item_indices)
        return user_indices, item_indices

    def reset_property(self):
        self._n_users = None
        self._n_items = None
        self._user2id = None
        self._item2id = None
        self._id2user = None
        self._id2item = None
        self._data_size = None

    def store_old_info(self):
        if self.sparse_unique_vals is not None:
            self.old_sparse_len, self.old_sparse_oov = [], []
            for i, col in enumerate(self.sparse_col.name):
                self.old_sparse_len.append(len(self.sparse_unique_vals[col]))
                self.old_sparse_oov.append(self.sparse_oov[i])

    def expand_sparse_unique_vals_and_matrix(self, data):
        self.reset_property()
        self.store_old_info()
        user_diff = np.setdiff1d(data.user.to_numpy(), self.user_unique_vals)
        if len(user_diff) > 0:
            self.user_unique_vals = np.append(self.user_unique_vals, user_diff)
            self.extend_unique_matrix("user", len(user_diff))

        item_diff = np.setdiff1d(data.item.to_numpy(), self.item_unique_vals)
        if len(item_diff) > 0:
            self.item_unique_vals = np.append(self.item_unique_vals, item_diff)
            self.extend_unique_matrix("item", len(item_diff))

        if self.sparse_unique_vals is not None:
            for sparse_col in self.sparse_unique_vals:
                unique_vals = list(self.sparse_unique_vals[sparse_col])
                sparse_diff = np.setdiff1d(data[sparse_col].to_numpy(),
                                           unique_vals)
                if len(sparse_diff) > 0:
                    unique_vals = np.append(unique_vals, sparse_diff)
                    self.sparse_unique_vals[sparse_col] = unique_vals
                    size = len(unique_vals)
                    self.sparse_unique_idxs[sparse_col] = dict(
                        zip(unique_vals, range(size))
                    )

    def extend_unique_matrix(self, mode, diff_num):
        if mode == "user":
            if self.user_sparse_unique is not None:
                new_users = np.zeros(
                    [diff_num, self.user_sparse_unique.shape[1]],
                    dtype=self.user_sparse_unique.dtype
                )
                # do not include last oov unique values
                self.user_sparse_unique = np.vstack(
                    [self.user_sparse_unique[:-1], new_users]
                )
            if self.user_dense_unique is not None:
                new_users = np.zeros(
                    [diff_num, self.user_dense_unique.shape[1]],
                    dtype=self.user_dense_unique.dtype
                )
                self.user_dense_unique = np.vstack(
                    [self.user_dense_unique[:-1], new_users]
                )
        elif mode == "item":
            if self.item_sparse_unique is not None:
                new_items = np.zeros(
                    [diff_num, self.item_sparse_unique.shape[1]],
                    dtype=self.item_sparse_unique.dtype
                )
                self.item_sparse_unique = np.vstack(
                    [self.item_sparse_unique[:-1], new_items]
                )
            if self.item_dense_unique is not None:
                new_items = np.zeros(
                    [diff_num, self.item_dense_unique.shape[1]],
                    dtype=self.item_dense_unique.dtype
                )
                self.item_dense_unique = np.vstack(
                    [self.item_dense_unique[:-1], new_items]
                )

    # sparse_indices and offset will increase if sparse feature encounter new categories
    def modify_sparse_indices(self):
        old_offset = [i + 1 for i in self.old_sparse_oov[:-1]]
        old_offset = np.insert(old_offset, 0, 0)
        if self.user_sparse_unique is not None:
            user_idx = self.user_sparse_col.index
            diff = self.sparse_offset[user_idx] - old_offset[user_idx]
            self.user_sparse_unique += diff
        if self.item_sparse_unique is not None:
            item_idx = self.item_sparse_col.index
            diff = self.sparse_offset[item_idx] - old_offset[item_idx]
            self.item_sparse_unique += diff

    # todo: ignore feature oov value
    def assign_sparse_features(self, data, mode):
        data = _check_oov(self, data, mode)
        if mode == "user":
            row_idx = data["user"].to_numpy()
            col_info = self.user_sparse_col
            if self.user_sparse_unique is not None and col_info.name:
                for feat_idx, col in enumerate(col_info.name):
                    if col not in data.columns:
                        continue
                    self.user_sparse_unique[row_idx, feat_idx] = (
                        compute_sparse_feat_indices(
                            self, data, col_info.index[feat_idx], col)
                    )
        elif mode == "item":
            row_idx = data["item"].to_numpy()
            col_info = self.item_sparse_col
            if self.item_sparse_unique is not None and col_info.name:
                for feat_idx, col in enumerate(col_info.name):
                    if col not in data.columns:
                        continue
                    self.item_sparse_unique[row_idx, feat_idx] = (
                        compute_sparse_feat_indices(
                            self, data, col_info.index[feat_idx], col)
                    )
        else:
            raise ValueError("mode must be user or item.")

    def assign_dense_features(self, data, mode):
        data = _check_oov(self, data, mode)
        if mode == "user":
            row_idx = data["user"].to_numpy()
            col_info = self.user_dense_col
            if self.user_dense_unique is not None and col_info.name:
                for feat_idx, col in enumerate(col_info.name):
                    if col not in data.columns:
                        continue
                    self.user_dense_unique[row_idx, feat_idx] = (
                        data[col].to_numpy())
        elif mode == "item":
            row_idx = data["item"].to_numpy()
            col_info = self.item_dense_col
            if self.item_dense_unique is not None and col_info.name:
                for feat_idx, col in enumerate(col_info.name):
                    if col not in data.columns:
                        continue
                    self.item_dense_unique[row_idx, feat_idx] = (
                        data[col].to_numpy())

    def assign_user_features(self, user_data):
        self.assign_sparse_features(user_data, "user")
        self.assign_dense_features(user_data, "user")

    def assign_item_features(self, item_data):
        self.assign_sparse_features(item_data, "item")
        self.assign_dense_features(item_data, "item")

    def add_oov(self):
        if (self.user_sparse_unique is not None and
                len(self.user_sparse_unique) == self.n_users):
            user_sparse_oov = self.sparse_oov[self.user_sparse_col.index]
            self.user_sparse_unique = np.vstack(
                [self.user_sparse_unique, user_sparse_oov]
            )
        if (self.item_sparse_unique is not None and
                len(self.item_sparse_unique) == self.n_items):
            item_sparse_oov = self.sparse_oov[self.item_sparse_col.index]
            self.item_sparse_unique = np.vstack(
                [self.item_sparse_unique, item_sparse_oov]
            )
        if (self.user_dense_unique is not None and
                len(self.user_dense_unique) == self.n_users):
            user_dense_oov = np.mean(self.user_dense_unique, axis=0)
            self.user_dense_unique = np.vstack(
                [self.user_dense_unique, user_dense_oov]
            )
        if (self.item_dense_unique is not None and
                len(self.item_dense_unique) == self.n_items):
            item_dense_oov = np.mean(self.item_dense_unique, axis=0)
            self.item_dense_unique = np.vstack(
                [self.item_dense_unique, item_dense_oov]
            )

    def set_popular_items(self, num):
        count_items = self.interaction_data.drop_duplicates(
            subset=["user", "item"]
        ).groupby("item")["user"].count()
        selected_items = count_items.sort_values(
            ascending=False).index.tolist()[:num]
        # if not enough items, add old populars
        if len(selected_items) < num and self.popular_items is not None:
            diff = num - len(selected_items)
            selected_items.extend(self.popular_items[:diff])
        self.popular_items = selected_items

    def store_args(self, user_indices, item_indices):
        self.all_args = dict()
        inside_args = [
            "col_name_mapping",
            "interaction_data",
            "user_sparse_unique",
            "user_dense_unique",
            "item_sparse_unique",
            "item_dense_unique",
            "user_unique_vals",
            "item_unique_vals",
            "sparse_unique_vals",
            "sparse_offset",
            "sparse_oov",
        ]
        all_variables = vars(self)
        for arg in inside_args:
            if arg in all_variables and all_variables[arg] is not None:
                self.all_args[arg] = all_variables[arg]
        self.all_args["user_indices"] = user_indices
        self.all_args["item_indices"] = item_indices

    def save(self, path):
        if not os.path.isdir(path):
            print(f"file folder {path} doesn't exists, creating a new one...")
            os.makedirs(path)
        if self.col_name_mapping is not None:
            name_mapping_path = os.path.join(
                path, "data_info_name_mapping.json"
            )
            with open(name_mapping_path, 'w') as f:
                json.dump(self.all_args["col_name_mapping"],
                          f, separators=(',', ':'), indent=4)

        other_path = os.path.join(path, "data_info")
        hparams = dict()
        arg_names = inspect.signature(self.__init__).parameters.keys()
        for arg in arg_names:
            if (arg == "col_name_mapping" or
                    arg not in self.all_args or
                    self.all_args[arg] is None):
                continue
            if arg == "interaction_data":
                hparams[arg] = self.all_args[arg].to_numpy()
            elif arg == "sparse_unique_vals":
                sparse_unique_vals = self.all_args[arg]
                for col, val in sparse_unique_vals.items():
                    hparams["unique_"+str(col)] = np.asarray(val)
            else:
                hparams[arg] = self.all_args[arg]

        np.savez_compressed(other_path, **hparams)

    @classmethod
    def load(cls, path):
        if not os.path.exists(path):
            raise OSError(f"file folder {path} doesn't exists...")

        hparams = dict()
        name_mapping_path = os.path.join(path, "data_info_name_mapping.json")
        if os.path.exists(name_mapping_path):
            with open(name_mapping_path, 'r') as f:
                hparams["col_name_mapping"] = json.load(f)

        other_path = os.path.join(path, "data_info.npz")
        info = np.load(other_path, allow_pickle=True)
        info = dict(info.items())
        for i in info:
            if i.startswith("unique_"):
                hparams["sparse_unique_vals"] = dict()
                break
        for arg in info:
            if arg == "interaction_data":
                # noinspection PyTypeChecker
                hparams[arg] = pd.DataFrame(
                    info[arg], columns=["user", "item", "label"])
            elif arg.startswith("unique_"):
                hparams["sparse_unique_vals"][arg[7:]] = info[arg]
            else:
                hparams[arg] = info[arg]
        return cls(**hparams)
