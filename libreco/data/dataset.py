import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from .data_info import DataInfo
from .transformed import TransformedSet
from ..utils.column_mapping import col_name2index
from ..utils.unique_features import construct_unique_feat
import warnings
warnings.filterwarnings("ignore")


class Dataset(object):
    """Base class for loading dataset.

    Warning: This class should not be used directly. Use derived class instead.
    """

    sparse_unique_vals = dict()
    user_unique_vals = None
    item_unique_vals = None
#    dense_col = None
#    sparse_col = None
#    multi_sparse_col = None

    @classmethod
    def load_builtin(cls, name="ml-1m") -> pd.DataFrame:
        pass

#    @classmethod
#    def load_from_file(cls, data, kind="pure"):
#        if kind == "pure":
#            return DatasetPure(data)
#        elif kind == "feat":
#            return DatasetFeat(data)
#        else:
#            raise ValueError("data kind must either be 'pure' or 'feat'.")

    @staticmethod
    def _check_col_names(data, mode):
        if not np.all(["user" == data.columns[0], "item" == data.columns[1]]):
            raise ValueError(
                "'user', 'item' must be the first two columns of the data")
        if mode == "train":
            assert ("label" in data.columns,
                    "train data should contain label column")

    @classmethod
    def _check_subclass(cls):
        if not issubclass(cls, Dataset):
            raise NameError(
                "Please use 'DatasetPure' or 'DatasetFeat' to call method")

    @classmethod
    def _set_sparse_unique_vals(cls, train_data, sparse_col):
        if sparse_col is not None:
            for col in sparse_col:
                cls.sparse_unique_vals[col] = np.unique(train_data[col])
        cls.user_unique_vals = np.unique(train_data["user"])
        cls.item_unique_vals = np.unique(train_data["item"])

    @classmethod
    def _get_feature_offset(cls, sparse_col):
        if cls.__name__.lower().endswith("pure"):
            unique_values = [
                len(cls.sparse_unique_vals[col]) for col in sparse_col
            ]
        elif cls.__name__.lower().endswith("feat"):
            # plus one for value only in test data
            unique_values = [
                len(cls.sparse_unique_vals[col]) + 1 for col in sparse_col
            ]
        return np.cumsum(np.array([0] + unique_values))

    @staticmethod
    def check_unknown(values, uniques):
        diff = list(np.setdiff1d(values, uniques, assume_unique=True))
        mask = np.in1d(values, uniques, invert=True)
        return diff, mask

    @classmethod
    def _sparse_indices(cls, values, unique, mode="train"):
        if mode == "test":
            diff, not_in_mask = cls.check_unknown(values, unique)
            col_indices = np.searchsorted(unique, values)
            col_indices[not_in_mask] = len(unique)
        elif mode == "train":
            col_indices = np.searchsorted(unique, values)
        else:
            raise ValueError("mode must either be \"train\" or \"test\" ")
        return col_indices

    @classmethod
    def _get_user_item_sparse_indices(cls, data, mode="train"):
        user_indices = cls._sparse_indices(
            data.user.to_numpy(), cls.user_unique_vals, mode)
        item_indices = cls._sparse_indices(
            data.item.to_numpy(), cls.item_unique_vals, mode)
        return user_indices, item_indices

    @classmethod
    def _get_sparse_indices_matrix(cls, data, sparse_col, mode="train"):
        n_samples, n_features = len(data), len(sparse_col)
        sparse_indices = np.zeros((n_samples, n_features), dtype=np.int32)
        for i, col in enumerate(sparse_col):
            col_values = data[col].to_numpy()
            unique_values = cls.sparse_unique_vals[col]
            sparse_indices[:, i] = cls._sparse_indices(
                col_values, unique_values, mode)

        feature_offset = cls._get_feature_offset(sparse_col)
        return sparse_indices + feature_offset[:-1]

    @classmethod
    def _get_dense_indices_matrix(cls, data, dense_col):
        n_samples, n_features = len(data), len(dense_col)
        dense_indices = np.tile(np.arange(n_features), [n_samples, 1])
        return dense_indices


class DatasetPure(Dataset):
    """A derived class from :class:`Dataset`, used for pure
    collaborative filtering
    """

    @classmethod
    def build_trainset(cls, train_data, shuffle=False, seed=42):
        """Build transformed pure train_data from original data.

        Normally, pure data only contains `user` and `item` columns,
        so only `sparse_col` is needed.

        Parameters
        ----------
        train_data : `pandas.DataFrame`
            Data must at least contains three columns,
            i.e. `user`, `item`, `label`.
        shuffle : bool, optional
            Whether to fully shuffle data.
        seed: int, optional
            random seed.

        Returns
        -------
        trainset : `TransformedSet` object
            Data object used for training.
        data_info : `DataInfo` object
            Object that contains some useful information
            for training and predicting
        """

        cls._check_subclass()
        cls._check_col_names(train_data, mode="train")
        cls._set_sparse_unique_vals(train_data, None)
        if shuffle:
            train_data = train_data.sample(
                frac=1, random_state=seed).reset_index(drop=True)

        user_indices, item_indices = cls._get_user_item_sparse_indices(
            train_data, mode="train")
        labels = train_data["label"].to_numpy(dtype=np.float32)

        interaction_data = train_data[["user", "item", "label"]]
        train_transformed = TransformedSet(user_indices,
                                           item_indices,
                                           labels,
                                           train=True)
        data_info = DataInfo(interaction_data=interaction_data)
        return train_transformed, data_info

    @classmethod
    def build_testset(cls, test_data, shuffle=False, seed=42):
        """Build transformed pure eval_data or test_data from original data.

        Normally, pure data only contains `user` and `item` columns,
        so only `sparse_col` is needed.

        Parameters
        ----------
        test_data : `pandas.DataFrame`
            Data must at least contains two columns, i.e. `user`, `item`.
        shuffle : bool, optional
            Whether to fully shuffle data.
        seed: int, optional
            random seed.

        Returns
        -------
        testset : `TransformedSet` object
            Data object used for evaluate and test.
        """

        cls._check_subclass()
        cls._check_col_names(test_data, mode="test")
        if shuffle:
            test_data = test_data.sample(
                frac=1, random_state=seed).reset_index(drop=True)

        (test_user_indices,
         test_item_indices) = cls._get_user_item_sparse_indices(
            test_data, mode="test")
        if "label" in test_data.columns:
            labels = test_data["label"].to_numpy(dtype=np.float32)
        else:
            # in case test_data has no label column,
            # create dummy labels for consistency
            labels = np.zeros(len(test_data))

        test_transformed = TransformedSet(test_user_indices,
                                          test_item_indices,
                                          labels,
                                          train=False)
        return test_transformed

    @classmethod
    def build_train_test(cls, train_data, test_data,
                         shuffle=(False, False), seed=42):
        """Build transformed pure train_data and test_data from original data.

        Normally, pure data only contains `user` and `item` columns,
        so only `sparse_col` is needed.

        Parameters
        ----------
        train_data : `pandas.DataFrame`
            Data must at least contains three columns,
            i.e. `user`, `item`, `label`.
        test_data : `pandas.DataFrame`
            Data must at least contains two columns,
            i.e. `user`, `item`.
        shuffle : list of bool, optional
            Whether to fully shuffle train and test data
        seed: int, optional
            random seed

        Returns
        -------
        trainset : `TransformedSet` object
            Data object used for training.
        testset : `TransformedSet` object
            Data object used for evaluation and test.
        data_info : `DataInfo` object
            Object that contains some useful information for
            training and predicting
        """

        trainset, data_info = cls.build_trainset(train_data, shuffle[0], seed)
        testset = cls.build_testset(test_data, shuffle[1], seed)
        return trainset, testset, data_info


class DatasetFeat(Dataset):
    """A derived class from :class:`Dataset`, used for data that
    contains features
    """

    @classmethod   # TODO: pseudo pure
    def build_trainset(cls, train_data, user_col=None, item_col=None,
                       sparse_col=None, dense_col=None, shuffle=False,
                       seed=42):
        """Build transformed feat train_data from original data.

        Normally, `user` and `item` column will be transformed into
        sparse indices, so `sparse_col` must be provided.

        Parameters
        ----------
        train_data : `pandas.DataFrame`
            Data must at least contains three columns,
            i.e. `user`, `item`, `label`.
        user_col : list of str
            List of user feature column names.
        item_col : list of str
            List of item feature column names.
        sparse_col : list of str
            List of sparse feature columns names,
            usually include `user` and `item`, so it must be provided.
        dense_col : list of str, optional
            List of dense feature column names.
        shuffle : bool, optional
            Whether to fully shuffle data.
        seed: int, optional
            random seed.

        Returns
        -------
        trainset : `TransformedSet` object
            Data object used for training.
        data_info : `DataInfo` object
            Object that contains some useful information
            for training and predicting
        """

        cls._check_subclass()
        cls._check_col_names(train_data, mode="train")
        cls._set_sparse_unique_vals(train_data, sparse_col)
        if shuffle:
            train_data = train_data.sample(
                frac=1, random_state=seed).reset_index(drop=True)

        user_indices, item_indices = cls._get_user_item_sparse_indices(
            train_data, mode="train")
        train_sparse_indices = cls._get_sparse_indices_matrix(
            train_data, sparse_col, mode="train") if sparse_col else None
        train_dense_values = (
            train_data[dense_col].to_numpy() if dense_col else None)
        labels = train_data["label"].to_numpy(dtype=np.float32)

        train_transformed = TransformedSet(user_indices,
                                           item_indices,
                                           labels,
                                           train_sparse_indices,
                                           train_dense_values,
                                           train=True)

        col_name_mapping = col_name2index(
            user_col, item_col, sparse_col, dense_col)
        user_sparse_col_indices = list(
            col_name_mapping["user_sparse_col"].values())
        user_dense_col_indices = list(
            col_name_mapping["user_dense_col"].values())
        item_sparse_col_indices = list(
            col_name_mapping["item_sparse_col"].values())
        item_dense_col_indices = list(
            col_name_mapping["item_dense_col"].values())

        (user_sparse_unique,
         user_dense_unique,
         item_sparse_unique,
         item_dense_unique) = construct_unique_feat(
            user_indices, item_indices, train_sparse_indices,
            train_dense_values, user_sparse_col_indices,
            user_dense_col_indices, item_sparse_col_indices,
            item_dense_col_indices
        )

        interaction_data = train_data[["user", "item", "label"]]
        data_info = DataInfo(col_name_mapping,
                             interaction_data,
                             user_sparse_unique,
                             user_dense_unique,
                             item_sparse_unique,
                             item_dense_unique)

        return train_transformed, data_info

    @classmethod
    def build_testset(cls, test_data, sparse_col=None, dense_col=None,
                      shuffle=False, seed=42):
        """Build transformed feat eval_data or test_data from original data.

        Normally, `user` and `item` column will be transformed
        into sparse indices, so `sparse_col` must be provided.

        Parameters
        ----------
        test_data : `pandas.DataFrame`
            Data must at least contains two columns, i.e. `user`, `item`.
        sparse_col : list of str
            List of sparse feature columns names,
            usually include `user` and `item`, so it must be provided.
        dense_col : list of str, optional
            List of dense feature column names.
        shuffle : bool, optional
            Whether to fully shuffle data.
        seed: int, optional
            random seed.

        Returns
        -------
        testset : `TransformedSet` object
            Data object used for evaluation and test.
        """

        cls._check_subclass()
        cls._check_col_names(test_data, "test")
        if shuffle:
            test_data = test_data.sample(
                frac=1, random_state=seed).reset_index(drop=True)

        (test_user_indices,
         test_item_indices) = cls._get_user_item_sparse_indices(
            test_data, mode="test")
        test_sparse_indices = cls._get_sparse_indices_matrix(
            test_data, sparse_col, mode="test") if sparse_col else None
        test_dense_values = (
            test_data[dense_col].to_numpy() if dense_col else None)

        if "label" in test_data.columns:
            labels = test_data["label"].to_numpy(dtype=np.float32)
        else:
            # in case test_data has no label column,
            # create dummy labels for consistency
            labels = np.zeros(len(test_data), dtype=np.float32)

        test_transformed = TransformedSet(test_user_indices,
                                          test_item_indices,
                                          labels,
                                          test_sparse_indices,
                                          test_dense_values,
                                          train=False)

        return test_transformed

    @classmethod
    def build_train_test(cls, train_data, test_data, user_col=None,
                         item_col=None, sparse_col=None, dense_col=None,
                         shuffle=(False, False), seed=42):
        """Build transformed feat train_data and test_data from original data.

        Normally, `user` and `item` column will be transformed into
        sparse indices, so `sparse_col` must be provided.

        Parameters
        ----------
        train_data : `pandas.DataFrame`
            Data must at least contains three columns,
            i.e. `user`, `item`, `label`.
        test_data : `pandas.DataFrame`
            Data must at least contains two columns,
            i.e. `user`, `item`.
        user_col : list of str
            List of user feature column names.
        item_col : list of str
            List of item feature column names.
        sparse_col : list of str
            List of sparse feature columns names,
            usually include `user` and `item`, so it must be provided.
        dense_col : list of str, optional
            List of dense feature column names.
        shuffle : list of bool, optional
            Whether to fully shuffle data.
        seed: int, optional
            random seed.

        Returns
        -------
        trainset : `TransformedSet` object
            Data object used for training.
        testset : `TransformedSet` object
            Data object used for evaluation and test.
        data_info : `DataInfo` object
            Object that contains some useful information
            for training and predicting
        """
        trainset, data_info = cls.build_trainset(
            train_data, user_col, item_col, sparse_col, dense_col,
            shuffle[0], seed)
        testset = cls.build_testset(
            test_data, sparse_col, dense_col, shuffle[1], seed)
        return trainset, testset, data_info

