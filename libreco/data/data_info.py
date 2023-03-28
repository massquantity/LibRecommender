"""Classes for Storing Various Data Information."""
import inspect
import json
import os
from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

from .consumed import interaction_consumed
from ..feature.update import (
    get_row_id_masks,
    update_new_dense_feats,
    update_new_sparse_feats,
)

Feature = namedtuple("Feature", ["name", "index"])

EmptyFeature = Feature(name=[], index=[])


# noinspection PyUnresolvedReferences
@dataclass
class MultiSparseInfo:
    """`dataclasses <https://docs.python.org/3/library/dataclasses.html>`_
    for storing multi-sparse features information.

    A group of multi-sparse features are considered as a "field",
    e.g., ("genre1", "genre2", "genre3") form a "genre" field,
    and features belong to the same field share the same oov.

    Attributes
    ----------
    field_offset : list of int
        All multi-sparse fields' offset in all expanded sparse features.
    field_len : list of int
        All multi-sparse fields' sizes.
    feat_oov : numpy.ndarray
        All multi-sparse fields' oov.
    pad_val : dict of {str : Any}
        Padding value in multi-sparse columns.
    """

    __slots__ = ("field_offset", "field_len", "feat_oov", "pad_val")

    field_offset: Iterable[int]
    field_len: Iterable[int]
    feat_oov: np.ndarray
    pad_val: Dict[str, Any]


class DataInfo:
    """Object for storing and updating information of indices and features.

    Parameters
    ----------
    col_name_mapping : dict of {dict : int} or None, default: None
        Column name to index mapping, which has the format: ``{column_family_name: {column_name: index}}``.
        If no such family, the default format would be: {column_family_name: {[]: []}}
    interaction_data : pandas.DataFrame or None, default: None
        Data contains ``user``, ``item`` and ``label`` columns
    user_sparse_unique : numpy.ndarray or None, default: None
        Unique sparse features for all users in train data.
    user_dense_unique : numpy.ndarray or None, default: None
        Unique dense features for all users in train data.
    item_sparse_unique : numpy.ndarray or None, default: None
        Unique sparse features for all items in train data.
    item_dense_unique : numpy.ndarray or None, default: None
        Unique dense features for all items in train data.
    user_indices : numpy.ndarray or None, default: None
        Mapped inner user indices from train data.
    item_indices : numpy.ndarray or None, default: None
        Mapped inner item indices from train data.
    user_unique_vals : numpy.ndarray or None, default: None
        All the unique users in train data.
    item_unique_vals : numpy.ndarray or None, default: None
        All the unique items in train data.
    sparse_unique_vals : dict of {str : numpy.ndarray} or None, default: None
        All sparse features' unique values.
    sparse_offset : numpy.ndarray or None, default: None
        Offset for each sparse feature in all sparse values. Often used in the ``embedding`` layer.
    sparse_oov : numpy.ndarray or None, default: None
        Out-of-vocabulary place for each sparse feature. Often used in cold-start.
    multi_sparse_unique_vals : dict of {str : numpy.ndarray} or None, default: None
        All multi-sparse features' unique values.
    multi_sparse_combine_info : MultiSparseInfo or None, default: None
        Multi-sparse field information.

    Attributes
    ----------
    col_name_mapping : dict of {dict : int} or None
        See Parameters
    user_consumed : dict of {int, list}
        Every users' consumed items in train data.
    item_consumed : dict of {int, list}
        Every items' consumed users in train data.

    See Also
    --------
    MultiSparseInfo
    """

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
        multi_sparse_unique_vals=None,
        multi_sparse_combine_info=None,
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
        self.sparse_offset = sparse_offset
        self.sparse_oov = sparse_oov
        self.multi_sparse_unique_vals = multi_sparse_unique_vals
        self.multi_sparse_combine_info = multi_sparse_combine_info
        self.sparse_idx_mapping = DataInfo.map_sparse_vals(
            sparse_unique_vals, multi_sparse_unique_vals
        )
        # Numpy doc states that it is recommended to use new random API
        # https://numpy.org/doc/stable/reference/random/index.html
        self.np_rng = np.random.default_rng()
        self._n_users = None
        self._n_items = None
        self._user2id = None
        self._item2id = None
        self._id2user = None
        self._id2item = None
        self._data_size = None
        self._popular_items = None
        # store old info for rebuild models
        self.old_info = None
        self.all_args = locals()
        self.add_oovs()

    @staticmethod
    def map_sparse_vals(sparse_unique_vals, multi_sparse_unique_vals):
        if sparse_unique_vals is None and multi_sparse_unique_vals is None:
            return

        def _map_vals(unique_vals):
            mapping = dict()
            if unique_vals is not None:
                for col, vals in unique_vals.items():
                    size = len(vals)
                    mapping[col] = dict(zip(vals, range(size)))
            return mapping

        res = dict()
        res.update(_map_vals(sparse_unique_vals))
        res.update(_map_vals(multi_sparse_unique_vals))
        assert len(res) > 0
        return res

    @property
    def global_mean(self):
        """Mean value of all labels in `rating` task."""
        return self.interaction_data.label.mean()

    @property
    def min_max_rating(self):
        """Min and max value of all labels in `rating` task."""
        return self.interaction_data.label.min(), self.interaction_data.label.max()

    @property
    def sparse_col(self):
        """Sparse column name to index mapping."""
        if not self.col_name_mapping or "sparse_col" not in self.col_name_mapping:
            return EmptyFeature
        return Feature(
            name=list(self.col_name_mapping["sparse_col"].keys()),
            index=list(self.col_name_mapping["sparse_col"].values()),
        )

    @property
    def dense_col(self):
        """Dense column name to index mapping."""
        if not self.col_name_mapping or "dense_col" not in self.col_name_mapping:
            return EmptyFeature
        return Feature(
            name=list(self.col_name_mapping["dense_col"].keys()),
            index=list(self.col_name_mapping["dense_col"].values()),
        )

    @property
    def user_sparse_col(self):
        """User sparse column name to index mapping."""
        if not self.col_name_mapping or "user_sparse_col" not in self.col_name_mapping:
            return EmptyFeature
        return Feature(
            name=list(self.col_name_mapping["user_sparse_col"].keys()),
            index=list(self.col_name_mapping["user_sparse_col"].values()),
        )

    @property
    def user_dense_col(self):
        """User dense column name to index mapping."""
        if not self.col_name_mapping or "user_dense_col" not in self.col_name_mapping:
            return EmptyFeature
        return Feature(
            name=list(self.col_name_mapping["user_dense_col"].keys()),
            index=list(self.col_name_mapping["user_dense_col"].values()),
        )

    @property
    def item_sparse_col(self):
        """Item sparse column name to index mapping."""
        if not self.col_name_mapping or "item_sparse_col" not in self.col_name_mapping:
            return EmptyFeature
        return Feature(
            name=list(self.col_name_mapping["item_sparse_col"].keys()),
            index=list(self.col_name_mapping["item_sparse_col"].values()),
        )

    @property
    def item_dense_col(self):
        """Item dense column name to index mapping."""
        if not self.col_name_mapping or "item_dense_col" not in self.col_name_mapping:
            return EmptyFeature
        return Feature(
            name=list(self.col_name_mapping["item_dense_col"].keys()),
            index=list(self.col_name_mapping["item_dense_col"].values()),
        )

    @property
    def user_col(self):
        """All the user column names, including sparse and dense."""
        if not self.col_name_mapping:
            return []
        user_sparse, user_dense = [], []
        if "user_sparse_col" in self.col_name_mapping:
            user_sparse = list(self.col_name_mapping["user_sparse_col"].keys())
        if "user_dense_col" in self.col_name_mapping:
            user_dense = list(self.col_name_mapping["user_dense_col"].keys())
        # The result columns will be sorted by key
        return user_sparse + user_dense

    @property
    def item_col(self):
        """All the item column names, including sparse and dense."""
        if not self.col_name_mapping:
            return []
        item_sparse, item_dense = [], []
        if "item_sparse_col" in self.col_name_mapping:
            item_sparse = list(self.col_name_mapping["item_sparse_col"].keys())
        if "item_dense_col" in self.col_name_mapping:
            item_dense = list(self.col_name_mapping["item_dense_col"].keys())
        # The result columns will be sorted by key
        return item_sparse + item_dense

    @property
    def n_users(self):
        """Number of users in train data."""
        if self._n_users is None:
            self._n_users = len(self.user_unique_vals)
        return self._n_users

    @property
    def n_items(self):
        """Number of items in train data."""
        if self._n_items is None:
            self._n_items = len(self.item_unique_vals)
        return self._n_items

    @property
    def user2id(self):
        """User original id to inner id mapping."""
        if self._user2id is None:
            self._user2id = dict(zip(self.user_unique_vals, range(self.n_users)))
        return self._user2id

    @property
    def item2id(self):
        """Item original id to inner id mapping."""
        if self._item2id is None:
            self._item2id = dict(zip(self.item_unique_vals, range(self.n_items)))
        return self._item2id

    @property
    def id2user(self):
        """User inner id to original id mapping."""
        if self._id2user is None:
            self._id2user = {j: user for user, j in self.user2id.items()}
        return self._id2user

    @property
    def id2item(self):
        """User inner id to original id mapping."""
        if self._id2item is None:
            self._id2item = {j: item for item, j in self.item2id.items()}
        return self._id2item

    @property
    def data_size(self):
        """Train data size."""
        if self._data_size is None:
            self._data_size = len(self.interaction_data)
        return self._data_size

    def __repr__(self):
        r"""Output train data information: \"n_users, n_items, data density\"."""
        n_users = self.n_users
        n_items = self.n_items
        n_labels = len(self.interaction_data)
        return "n_users: %d, n_items: %d, data density: %.4f %%" % (
            n_users,
            n_items,
            100 * n_labels / (n_users * n_items),
        )

    def assign_user_features(self, user_data):
        """Assign user features to this ``data_info`` object from ``user_data``.

        Parameters
        ----------
        user_data : pandas.DataFrame
            Data contains new user features.
        """
        assert "user" in user_data.columns, "Data must contain `user` column."
        user_data = user_data.drop_duplicates(subset=["user"], keep="last")
        user_row_idx, user_id_mask = get_row_id_masks(
            user_data["user"], self.user_unique_vals
        )
        self.user_sparse_unique = update_new_sparse_feats(
            user_data,
            user_row_idx,
            user_id_mask,
            self.user_sparse_unique,
            self.sparse_unique_vals,
            self.multi_sparse_unique_vals,
            self.user_sparse_col,
            self.col_name_mapping,
            self.sparse_offset,
        )
        self.user_dense_unique = update_new_dense_feats(
            user_data,
            user_row_idx,
            user_id_mask,
            self.user_dense_unique,
            self.user_dense_col,
        )

    def assign_item_features(self, item_data):
        """Assign item features to this ``data_info`` object from ``item_data``.

        Parameters
        ----------
        item_data : pandas.DataFrame
            Data contains new item features.
        """
        assert "item" in item_data.columns, "Data must contain `item` column."
        item_data = item_data.drop_duplicates(subset=["item"], keep="last")
        item_row_idx, item_id_mask = get_row_id_masks(
            item_data["item"], self.item_unique_vals
        )
        self.item_sparse_unique = update_new_sparse_feats(
            item_data,
            item_row_idx,
            item_id_mask,
            self.item_sparse_unique,
            self.sparse_unique_vals,
            self.multi_sparse_unique_vals,
            self.item_sparse_col,
            self.col_name_mapping,
            self.sparse_offset,
        )
        self.item_dense_unique = update_new_dense_feats(
            item_data,
            item_row_idx,
            item_id_mask,
            self.item_dense_unique,
            self.item_dense_col,
        )

    def add_oovs(self):
        def _concat_oov(uniques, cols=None):
            if uniques is None:
                return
            oov = self.sparse_oov[cols] if cols else np.mean(uniques, axis=0)
            return np.vstack([uniques, oov])

        self.user_sparse_unique = _concat_oov(
            self.user_sparse_unique, self.user_sparse_col.index
        )
        self.item_sparse_unique = _concat_oov(
            self.item_sparse_unique, self.item_sparse_col.index
        )
        self.user_dense_unique = _concat_oov(self.user_dense_unique)
        self.item_dense_unique = _concat_oov(self.item_dense_unique)

    @property
    def popular_items(self):
        """A number of popular items in train data which often used in cold-start."""
        if self._popular_items is None:
            self._popular_items = self._get_popular_items(100)
        return self._popular_items

    def _get_popular_items(self, num):
        count_items = (
            self.interaction_data.drop_duplicates(subset=["user", "item"])
            .groupby("item")["user"]
            .count()
        )
        selected_items = count_items.sort_values(ascending=False).index.tolist()[:num]
        # if not enough items, add old populars
        if len(selected_items) < num and self.old_info is not None:
            diff = num - len(selected_items)
            selected_items.extend(self.old_info.popular_items[:diff])
        return selected_items

    def save(self, path, model_name):
        """Save :class:`DataInfo` Object.

        Parameters
        ----------
        path : str
            File folder path to save :class:`DataInfo`.
        model_name : str
            Name of the saved file.
        """
        if not os.path.isdir(path):
            print(f"file folder {path} doesn't exists, creating a new one...")
            os.makedirs(path)
        if self.col_name_mapping is not None:
            name_mapping_path = os.path.join(
                path, f"{model_name}_data_info_name_mapping.json"
            )
            with open(name_mapping_path, "w") as f:
                json.dump(
                    self.all_args["col_name_mapping"],
                    f,
                    separators=(",", ":"),
                    indent=4,
                )

        other_path = os.path.join(path, f"{model_name}_data_info")
        hparams = dict()
        arg_names = inspect.signature(self.__init__).parameters.keys()
        for arg in arg_names:
            if (
                arg == "col_name_mapping"
                or arg not in self.all_args
                or self.all_args[arg] is None
            ):
                continue
            if arg == "interaction_data":
                hparams[arg] = self.all_args[arg].to_numpy()
            elif arg == "sparse_unique_vals":
                sparse_unique_vals = self.all_args[arg]
                for col, val in sparse_unique_vals.items():
                    hparams["unique_" + str(col)] = np.asarray(val)
            elif arg == "multi_sparse_unique_vals":
                multi_sparse_unique_vals = self.all_args[arg]
                for col, val in multi_sparse_unique_vals.items():
                    hparams["munique_" + str(col)] = np.asarray(val)
            else:
                hparams[arg] = self.all_args[arg]

        np.savez_compressed(other_path, **hparams)

    @classmethod
    def load(cls, path, model_name):
        """Load saved :class:`DataInfo`.

        Parameters
        ----------
        path : str
            File folder path to save :class:`DataInfo`.
        model_name : str
            Name of the saved file.
        """
        if not os.path.exists(path):
            raise OSError(f"file folder {path} doesn't exists...")

        hparams = dict()
        name_mapping_path = os.path.join(
            path, f"{model_name}_data_info_name_mapping.json"
        )
        if os.path.exists(name_mapping_path):
            with open(name_mapping_path, "r") as f:
                hparams["col_name_mapping"] = json.load(f)

        other_path = os.path.join(path, f"{model_name}_data_info.npz")
        info = np.load(other_path, allow_pickle=True)
        info = dict(info.items())
        for arg in info:
            if arg == "interaction_data":
                hparams[arg] = pd.DataFrame(
                    info[arg], columns=["user", "item", "label"]
                )
            elif arg == "multi_sparse_combine_info":
                # numpy can save MultiSparseInfo in 0-d array.
                hparams[arg] = info[arg].item()
            elif arg.startswith("unique_"):
                if "sparse_unique_vals" not in hparams:
                    hparams["sparse_unique_vals"] = dict()
                hparams["sparse_unique_vals"][arg[7:]] = info[arg]
            elif arg.startswith("munique_"):
                if "multi_sparse_unique_vals" not in hparams:
                    hparams["multi_sparse_unique_vals"] = dict()
                hparams["multi_sparse_unique_vals"][arg[8:]] = info[arg]
            else:
                hparams[arg] = info[arg]

        return cls(**hparams)


@dataclass
class OldInfo:
    n_users: int
    n_items: int
    sparse_len: List[int]
    sparse_oov: List[int]
    popular_items: List[Any]


def store_old_info(data_info):
    sparse_len = list()
    sparse_oov = list()
    sparse_unique = data_info.sparse_unique_vals
    multi_sparse_unique = data_info.multi_sparse_unique_vals
    for i, col in enumerate(data_info.sparse_col.name):
        if sparse_unique is not None and col in sparse_unique:
            sparse_len.append(len(sparse_unique[col]))
            sparse_oov.append(data_info.sparse_oov[i])
        elif multi_sparse_unique is not None and col in multi_sparse_unique:
            sparse_len.append(len(multi_sparse_unique[col]))
            sparse_oov.append(data_info.sparse_oov[i])
        elif (
            multi_sparse_unique is not None
            and "multi_sparse" in data_info.col_name_mapping
            and col in data_info.col_name_mapping["multi_sparse"]
        ):
            # multi_sparse case, second to last cols are redundant.
            # Used in `rebuild_tf_model`, `rebuild_torch_model`
            sparse_len.append(-1)
    return OldInfo(
        data_info.n_users,
        data_info.n_items,
        sparse_len,
        sparse_oov,
        data_info.popular_items,
    )
