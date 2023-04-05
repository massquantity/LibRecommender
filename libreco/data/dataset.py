"""Classes for Transforming and Building Data."""
import functools
import itertools

import numpy as np

from .consumed import update_consumed
from .data_info import DataInfo, store_old_info
from .transformed import TransformedSet
from ..feature.column_mapping import col_name2index
from ..feature.multi_sparse import (
    get_multi_sparse_info,
    multi_sparse_col_map,
    recover_sparse_cols,
)
from ..feature.sparse import (
    get_id_indices,
    get_oov_pos,
    merge_offset,
    merge_sparse_col,
    merge_sparse_indices,
)
from ..feature.unique import construct_unique_feat
from ..feature.update import (
    update_id_unique,
    update_multi_sparse_unique,
    update_sparse_unique,
    update_unique_feats,
)


class _Dataset(object):
    """Base class for loading dataset."""

    user_unique_vals = None
    item_unique_vals = None
    train_called = False

    @staticmethod
    def _check_col_names(data, is_train):
        if not np.all(["user" == data.columns[0], "item" == data.columns[1]]):
            raise ValueError("'user', 'item' must be the first two columns of the data")
        if is_train:
            assert "label" in data.columns, "train data should contain label column"

    @classmethod
    def _check_subclass(cls):
        if not issubclass(cls, _Dataset):
            raise NameError("Please use 'DatasetPure' or 'DatasetFeat' to call method")

    @staticmethod
    def shuffle_data(data, seed):
        """Shuffle data randomly.

        Parameters
        ----------
        data : pandas.DataFrame
            Data to shuffle.
        seed : int
            Random seed.

        Returns
        -------
        pandas.DataFrame
            Shuffled data.
        """
        data = data.sample(frac=1, random_state=seed)
        return data.reset_index(drop=True)

    @classmethod
    def _transform_test_factory(cls, test_data, shuffle, seed, data_info=None):
        if not cls.train_called:
            raise RuntimeError(
                "Must first build trainset before building evalset or testset"
            )
        cls._check_subclass()
        cls._check_col_names(test_data, is_train=False)
        if shuffle:
            test_data = cls.shuffle_data(test_data, seed)

        if cls.__name__ == "DatasetPure":
            return _build_transformed_set(
                test_data,
                cls.user_unique_vals,
                cls.item_unique_vals,
                is_train=False,
                is_ordered=False,
            )
        else:
            return _build_transformed_set_feat(
                test_data,
                cls.user_unique_vals,
                cls.item_unique_vals,
                is_train=False,
                is_ordered=False,
                data_info=data_info,
            )

    @classmethod
    def build_evalset(cls, eval_data, shuffle=False, seed=42):
        """Build transformed eval data from original data.

        .. versionchanged:: 1.0.0
           Data construction in :ref:`Model Retrain <retrain_data>` has been moved
           to :meth:`merge_evalset`

        Parameters
        ----------
        eval_data : pandas.DataFrame
            Data must contain at least two columns, i.e. `user`, `item`.
        shuffle : bool, default: False
            Whether to fully shuffle data.
        seed: int, default: 42
            Random seed.

        Returns
        -------
        :class:`~libreco.data.TransformedSet`
            Transformed Data object used for evaluating.
        """
        return cls._transform_test_factory(eval_data, shuffle, seed)

    @classmethod
    def build_testset(cls, test_data, shuffle=False, seed=42):
        """Build transformed test data from original data.

        .. versionchanged:: 1.0.0
           Data construction in :ref:`Model Retrain <retrain_data>` has been moved
           to :meth:`merge_testset`

        Parameters
        ----------
        test_data : pandas.DataFrame
            Data must contain at least two columns, i.e. `user`, `item`.
        shuffle : bool, default: False
            Whether to fully shuffle data.
        seed: int, default: 42
            Random seed.

        Returns
        -------
        :class:`~libreco.data.TransformedSet`
            Transformed Data object used for testing.
        """
        return cls._transform_test_factory(test_data, shuffle, seed)

    @classmethod
    def merge_evalset(cls, eval_data, data_info, shuffle=False, seed=42):
        """Build transformed data by merging new train data with old data.

        .. versionadded:: 1.0.0

        Parameters
        ----------
        eval_data : pandas.DataFrame
            Data must contain at least two columns, i.e. `user`, `item`.
        data_info : DataInfo
            Object that contains past data information.
        shuffle : bool, default: False
            Whether to fully shuffle data.
        seed: int, default: 42
            Random seed.

        Returns
        -------
        :class:`~libreco.data.TransformedSet`
            Transformed Data object used for testing.
        """
        return cls._transform_test_factory(eval_data, shuffle, seed, data_info)

    @classmethod
    def merge_testset(cls, test_data, data_info, shuffle=False, seed=42):
        """Build transformed data by merging new train data with old data.

        .. versionadded:: 1.0.0

        Parameters
        ----------
        test_data : pandas.DataFrame
            Data must contain at least two columns, i.e. `user`, `item`.
        data_info : DataInfo
            Object that contains past data information.
        shuffle : bool, default: False
            Whether to fully shuffle data.
        seed: int, default: 42
            Random seed.

        Returns
        -------
        :class:`~libreco.data.TransformedSet`
            Transformed Data object used for testing.
        """
        return cls._transform_test_factory(test_data, shuffle, seed, data_info)


class DatasetPure(_Dataset):
    """Dataset class used for building pure collaborative filtering data.

    Examples
    --------
    >>> from libreco.data import DatasetPure
    >>> train_data, data_info = DatasetPure.build_trainset(train_data)
    >>> eval_data = DatasetPure.build_evalset(eval_data)
    >>> test_data = DatasetPure.build_testset(test_data)
    """

    @classmethod
    def build_trainset(cls, train_data, shuffle=False, seed=42):
        """Build transformed train data and data_info from original data.

        .. versionchanged:: 1.0.0
           Data construction in :ref:`Model Retrain <retrain_data>` has been moved
           to :meth:`merge_trainset`

        Parameters
        ----------
        train_data : pandas.DataFrame
            Data must contain at least three columns, i.e. ``user``, ``item``, ``label``.
        shuffle : bool, default: False
            Whether to fully shuffle data.

            .. Warning::
                If your data is order or time dependent, it is not recommended to shuffle data.

        seed: int, default: 42
            Random seed.

        Returns
        -------
        trainset : :class:`~libreco.data.TransformedSet`
            Transformed Data object used for training.
        data_info : :class:`~libreco.data.DataInfo`
            Object that contains some useful information.
        """
        cls._check_subclass()
        cls._check_col_names(train_data, is_train=True)
        cls.user_unique_vals = np.sort(train_data["user"].unique())
        cls.item_unique_vals = np.sort(train_data["item"].unique())
        if shuffle:
            train_data = cls.shuffle_data(train_data, seed)

        train_transformed, user_indices, item_indices = _build_transformed_set(
            train_data,
            cls.user_unique_vals,
            cls.item_unique_vals,
            is_train=True,
            is_ordered=True,
        )
        data_info = DataInfo(
            interaction_data=train_data[["user", "item", "label"]],
            user_indices=user_indices,
            item_indices=item_indices,
            user_unique_vals=cls.user_unique_vals,
            item_unique_vals=cls.item_unique_vals,
        )
        cls.train_called = True
        return train_transformed, data_info

    @classmethod
    def merge_trainset(
        cls, train_data, data_info, merge_behavior=True, shuffle=False, seed=42
    ):
        """Build transformed data by merging new train data with old data.

        .. versionadded:: 1.0.0

        .. versionchanged:: 1.1.0
           Applying a more functional approach. A new ``data_info`` will be constructed
           and returned, and the passed old ``data_info`` should be discarded.

        Parameters
        ----------
        train_data : pandas.DataFrame
            Data must contain at least three columns, i.e. ``user``, ``item``, ``label``.
        data_info : DataInfo
            Object that contains past data information.
        merge_behavior : bool, default: True
            Whether to merge the user behavior in old and new data.
        shuffle : bool, default: False
            Whether to fully shuffle data.
        seed: int, default: 42
            Random seed.

        Returns
        -------
        new_trainset : :class:`~libreco.data.TransformedSet`
            New transformed Data object used for training.
        new_data_info : :class:`~libreco.data.DataInfo`
            New ``data_info`` that contains some useful information.
        """
        assert isinstance(data_info, DataInfo), "Invalid passed `data_info`."
        cls._check_col_names(train_data, is_train=True)
        cls.user_unique_vals, cls.item_unique_vals = update_id_unique(
            train_data, data_info
        )
        if shuffle:
            train_data = cls.shuffle_data(train_data, seed)

        merge_transformed, user_indices, item_indices = _build_transformed_set(
            train_data,
            cls.user_unique_vals,
            cls.item_unique_vals,
            is_train=True,
            is_ordered=False,
        )
        new_data_info = DataInfo(
            interaction_data=train_data[["user", "item", "label"]],
            user_indices=user_indices,
            item_indices=item_indices,
            user_unique_vals=cls.user_unique_vals,
            item_unique_vals=cls.item_unique_vals,
        )
        new_data_info = update_consumed(new_data_info, data_info, merge_behavior)
        new_data_info.old_info = store_old_info(data_info)
        cls.train_called = True
        return merge_transformed, new_data_info


class DatasetFeat(_Dataset):
    """Dataset class used for building data contains features.

    Examples
    --------
    >>> from libreco.data import DatasetFeat
    >>> train_data, data_info = DatasetFeat.build_trainset(train_data)
    >>> eval_data = DatasetFeat.build_evalset(eval_data)
    >>> test_data = DatasetFeat.build_testset(test_data)
    """

    sparse_unique_vals = None
    multi_sparse_unique_vals = None
    sparse_col = None
    multi_sparse_col = None
    dense_col = None

    @classmethod
    def _set_feature_col(cls, sparse_col, dense_col, multi_sparse_col):
        cls.sparse_col = sparse_col or None
        cls.dense_col = dense_col or None
        if multi_sparse_col:
            if not all(isinstance(field, list) for field in multi_sparse_col):
                cls.multi_sparse_col = [multi_sparse_col]
            else:
                cls.multi_sparse_col = multi_sparse_col
        else:
            cls.multi_sparse_col = None

    @classmethod
    def _check_feature_cols(cls, user_col, item_col):
        all_sparse_col = (
            merge_sparse_col(cls.sparse_col, cls.multi_sparse_col)
            if cls.multi_sparse_col is not None
            else cls.sparse_col
        )
        sparse_cols = all_sparse_col or []
        dense_cols = cls.dense_col or []
        user_cols = user_col or []
        item_cols = item_col or []
        if len(sparse_cols) + len(dense_cols) != len(user_cols) + len(item_cols):
            len_str = "len(sparse_cols) + len(dense_cols) == len(user_cols) + len(item_cols)"  # fmt: skip
            raise ValueError(
                f"Please make sure length of columns match, i.e. `{len_str}`, got "
                f"sparse columns: {sparse_cols}, "
                f"dense columns: {dense_cols}, "
                f"user columns: {user_cols}, "
                f"item columns: {item_cols}"
            )
        columns1, columns2 = sparse_cols + dense_cols, user_cols + item_cols
        mis_match_cols = np.setxor1d(columns1, columns2)
        if len(mis_match_cols) > 0:
            raise ValueError(
                f"Got inconsistent columns: {mis_match_cols}, please check the column names"
            )

    @classmethod  # TODO: pseudo pure
    def build_trainset(
        cls,
        train_data,
        user_col=None,
        item_col=None,
        sparse_col=None,
        dense_col=None,
        multi_sparse_col=None,
        unique_feat=False,
        pad_val="missing",
        shuffle=False,
        seed=42,
    ):
        """Build transformed feat train data and data_info from original data.

        .. versionchanged:: 1.0.0
           Data construction in :ref:`Model Retrain <retrain_data>` has been moved
           to :meth:`merge_trainset`

        Parameters
        ----------
        train_data : pandas.DataFrame
            Data must contain at least three columns, i.e. ``user``, ``item``, ``label``.
        user_col : list of str or None, default: None
            List of user feature column names.
        item_col : list of str or None, default: None
            List of item feature column names.
        sparse_col : list of str or None, default: None
            List of sparse feature columns names.
        multi_sparse_col : nested lists of str or None, default: None
            Nested lists of multi_sparse feature columns names.
            For example, ``[["a", "b", "c"], ["d", "e"]]``
        dense_col : list of str or None, default: None
            List of dense feature column names.
        unique_feat : bool, default: False
            Whether the features of users and items are unique in train data.
        pad_val : int or str or list, default: "missing"
            Padding value in multi_sparse columns to ensure same length of all samples.

            .. Warning::
                If the ``pad_val`` is a single value, it will be used in all ``multi_sparse`` columns.
                So if you want to use different ``pad_val`` for different ``multi_sparse`` columns,
                the ``pad_val`` should be a list.

        shuffle : bool, default: False
            Whether to fully shuffle data.

            .. Warning::
                If your data is order or time dependent, it is not recommended to shuffle data.

        seed: int, default: 42
            Random seed.

        Returns
        -------
        trainset : :class:`~libreco.data.TransformedSet`
            Transformed Data object used for training.
        data_info : :class:`~libreco.data.DataInfo`
            Object that contains some useful information.

        Raises
        ------
        ValueError
            If the feature columns specified by the user are inconsistent.
        """
        cls._check_subclass()
        cls._check_col_names(train_data, is_train=True)
        cls._set_feature_col(sparse_col, dense_col, multi_sparse_col)
        cls._check_feature_cols(user_col, item_col)
        cls.user_unique_vals = np.sort(train_data["user"].unique())
        cls.item_unique_vals = np.sort(train_data["item"].unique())
        cls.sparse_unique_vals = _get_sparse_unique_vals(cls.sparse_col, train_data)
        cls.multi_sparse_unique_vals, pad_val_dict = _get_multi_sparse_unique_vals(
            cls.multi_sparse_col, train_data, pad_val
        )
        if shuffle:
            train_data = cls.shuffle_data(train_data, seed)

        (
            train_transformed,
            user_indices,
            item_indices,
            train_sparse_indices,
            train_dense_values,
        ) = _build_transformed_set_feat(
            train_data,
            cls.user_unique_vals,
            cls.item_unique_vals,
            is_train=True,
            is_ordered=True,
        )

        all_sparse_col = (
            merge_sparse_col(cls.sparse_col, cls.multi_sparse_col)
            if cls.multi_sparse_col
            else sparse_col
        )
        col_name_mapping = col_name2index(
            user_col, item_col, all_sparse_col, cls.dense_col
        )
        (
            user_sparse_unique,
            user_dense_unique,
            item_sparse_unique,
            item_dense_unique,
        ) = construct_unique_feat(
            user_indices,
            item_indices,
            train_sparse_indices,
            train_dense_values,
            col_name_mapping,
            unique_feat,
        )

        sparse_offset = merge_offset(
            cls.sparse_col,
            cls.multi_sparse_col,
            cls.sparse_unique_vals,
            cls.multi_sparse_unique_vals,
        )
        sparse_oov = get_oov_pos(
            cls.sparse_col,
            cls.multi_sparse_col,
            cls.sparse_unique_vals,
            cls.multi_sparse_unique_vals,
        )
        multi_sparse_info = get_multi_sparse_info(
            all_sparse_col,
            cls.sparse_col,
            cls.multi_sparse_col,
            cls.sparse_unique_vals,
            cls.multi_sparse_unique_vals,
            pad_val_dict,
        )
        if cls.multi_sparse_col:
            col_name_mapping["multi_sparse"] = multi_sparse_col_map(multi_sparse_col)

        interaction_data = train_data[["user", "item", "label"]]
        data_info = DataInfo(
            col_name_mapping,
            interaction_data,
            user_sparse_unique,
            user_dense_unique,
            item_sparse_unique,
            item_dense_unique,
            user_indices,
            item_indices,
            cls.user_unique_vals,
            cls.item_unique_vals,
            cls.sparse_unique_vals,
            sparse_offset,
            sparse_oov,
            cls.multi_sparse_unique_vals,
            multi_sparse_info,
        )
        cls.train_called = True
        return train_transformed, data_info

    @classmethod
    def merge_trainset(
        cls, train_data, data_info, merge_behavior=True, shuffle=False, seed=42
    ):
        """Build transformed data by merging new train data with old data.

        .. versionadded:: 1.0.0

        .. versionchanged:: 1.1.0
           Applying a more functional approach. A new ``data_info`` will be constructed
           and returned, and the passed old ``data_info`` should be discarded.

        Parameters
        ----------
        train_data : pandas.DataFrame
            Data must contain at least three columns, i.e. ``user``, ``item``, ``label``.
        data_info : DataInfo
            Object that contains past data information.
        merge_behavior : bool, default: True
            Whether to merge the user behavior in old and new data.
        shuffle : bool, default: False
            Whether to fully shuffle data.
        seed: int, default: 42
            Random seed.

        Returns
        -------
        new_trainset : :class:`~libreco.data.TransformedSet`
            New transformed Data object used for training.
        new_data_info : :class:`~libreco.data.DataInfo`
            New ``data_info`` that contains some useful information.
        """
        assert isinstance(data_info, DataInfo), "Invalid passed `data_info`."
        cls._check_col_names(train_data, is_train=True)
        cls.user_unique_vals, cls.item_unique_vals = update_id_unique(
            train_data, data_info
        )
        cls.sparse_unique_vals = update_sparse_unique(train_data, data_info)
        cls.multi_sparse_unique_vals = update_multi_sparse_unique(train_data, data_info)
        if shuffle:
            train_data = cls.shuffle_data(train_data, seed)

        (
            merge_transformed,
            user_indices,
            item_indices,
            sparse_cols,
            multi_sparse_cols,
        ) = _build_transformed_set_feat(
            train_data,
            cls.user_unique_vals,
            cls.item_unique_vals,
            is_train=True,
            is_ordered=False,
            data_info=data_info,
        )
        sparse_offset = merge_offset(
            sparse_cols,
            multi_sparse_cols,
            cls.sparse_unique_vals,
            cls.multi_sparse_unique_vals,
        )
        sparse_oov = get_oov_pos(
            sparse_cols,
            multi_sparse_cols,
            cls.sparse_unique_vals,
            cls.multi_sparse_unique_vals,
        )

        all_sparse_col = data_info.sparse_col.name
        pad_val = (
            data_info.multi_sparse_combine_info.pad_val
            if cls.multi_sparse_unique_vals
            else dict()
        )
        multi_sparse_info = get_multi_sparse_info(
            all_sparse_col,
            cls.sparse_col,
            cls.multi_sparse_col,
            cls.sparse_unique_vals,
            cls.multi_sparse_unique_vals,
            pad_val,
        )

        _update_func = functools.partial(
            update_unique_feats,
            train_data,
            data_info,
            sparse_unique=cls.sparse_unique_vals,
            multi_sparse_unique=cls.multi_sparse_unique_vals,
            sparse_offset=sparse_offset,
            sparse_oov=sparse_oov,
        )
        user_sparse_unique, user_dense_unique = _update_func(
            unique_ids=cls.user_unique_vals, is_user=True
        )
        item_sparse_unique, item_dense_unique = _update_func(
            unique_ids=cls.item_unique_vals, is_user=False
        )

        interaction_data = train_data[["user", "item", "label"]]
        new_data_info = DataInfo(
            data_info.col_name_mapping,
            interaction_data,
            user_sparse_unique,
            user_dense_unique,
            item_sparse_unique,
            item_dense_unique,
            user_indices,
            item_indices,
            cls.user_unique_vals,
            cls.item_unique_vals,
            cls.sparse_unique_vals,
            sparse_offset,
            sparse_oov,
            cls.multi_sparse_unique_vals,
            multi_sparse_info,
        )
        new_data_info = update_consumed(new_data_info, data_info, merge_behavior)
        new_data_info.old_info = store_old_info(data_info)
        cls.train_called = True
        return merge_transformed, new_data_info


def _get_sparse_unique_vals(sparse_col, train_data):
    if not sparse_col:
        return
    sparse_unique_vals = dict()
    for col in sparse_col:
        sparse_unique_vals[col] = np.sort(train_data[col].unique())
    return sparse_unique_vals


def _get_multi_sparse_unique_vals(multi_sparse_col, train_data, pad_val):
    if not multi_sparse_col:
        return None, None
    multi_sparse_unique_vals = dict()
    if not isinstance(pad_val, (list, tuple)):
        pad_val = [pad_val] * len(multi_sparse_col)
    if len(multi_sparse_col) != len(pad_val):
        raise ValueError("Length of `multi_sparse_col` and `pad_val` doesn't match")
    pad_val_dict = dict()
    for i, field in enumerate(multi_sparse_col):
        unique_vals = set(itertools.chain.from_iterable(train_data[field].to_numpy().T))
        if pad_val[i] in unique_vals:
            unique_vals.remove(pad_val[i])
        # use name of a field's first column as representative
        multi_sparse_unique_vals[field[0]] = np.sort(list(unique_vals))
        pad_val_dict[field[0]] = pad_val[i]
    return multi_sparse_unique_vals, pad_val_dict


def _build_transformed_set(
    data,
    user_unique_vals,
    item_unique_vals,
    is_train,
    is_ordered,
    has_feats=False,
):
    user_indices, item_indices = get_id_indices(
        data,
        user_unique_vals,
        item_unique_vals,
        is_train,
        is_ordered,
    )
    if "label" in data.columns:
        labels = data["label"].to_numpy(dtype=np.float32)
    else:
        # in case test_data has no label column, create dummy labels for consistency
        labels = np.zeros(len(data), dtype=np.float32)

    transformed_data = TransformedSet(
        user_indices, item_indices, labels, train=is_train
    )
    if has_feats:
        return user_indices, item_indices, labels

    if is_train:
        return transformed_data, user_indices, item_indices
    else:
        return transformed_data


def _build_transformed_set_feat(
    data,
    user_unique_vals,
    item_unique_vals,
    is_train,
    is_ordered,
    data_info=None,
):
    user_indices, item_indices, labels = _build_transformed_set(
        data, user_unique_vals, item_unique_vals, is_train, is_ordered, has_feats=True
    )

    sparse_indices, dense_values, sparse_cols, multi_sparse_cols = _build_features(
        data, is_train, is_ordered, data_info
    )
    transformed_data = TransformedSet(
        user_indices, item_indices, labels, sparse_indices, dense_values, train=is_train
    )
    if not is_train:
        return transformed_data

    pure_data = transformed_data, user_indices, item_indices
    if not data_info:
        return pure_data + (sparse_indices, dense_values)  # noqa: RUF005
    else:
        return pure_data + (sparse_cols, multi_sparse_cols)  # noqa: RUF005


def _build_features(data, is_train, is_ordered, data_info):
    sparse_indices, dense_values = None, None
    if data_info:
        sparse_cols, multi_sparse_cols = recover_sparse_cols(data_info)
        dense_cols = data_info.dense_col.name
    else:
        sparse_cols = DatasetFeat.sparse_col
        multi_sparse_cols = DatasetFeat.multi_sparse_col
        dense_cols = DatasetFeat.dense_col

    sparse_unique = DatasetFeat.sparse_unique_vals
    multi_sparse_unique = DatasetFeat.multi_sparse_unique_vals
    if sparse_cols or multi_sparse_cols:
        sparse_indices = merge_sparse_indices(
            data,
            sparse_cols,
            multi_sparse_cols,
            sparse_unique,
            multi_sparse_unique,
            is_train,
            is_ordered,
        )
    if dense_cols:
        dense_values = data[dense_cols].to_numpy(dtype=np.float32)
    return sparse_indices, dense_values, sparse_cols, multi_sparse_cols
