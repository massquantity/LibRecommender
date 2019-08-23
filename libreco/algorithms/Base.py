from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import numpy as np


class BasePure(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def build_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def fit(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def recommend_user(self, *args, **kwargs):
        raise NotImplementedError


class BaseFeat(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def build_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def fit(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def recommend_user(self, *args, **kwargs):
        raise NotImplementedError

    def get_predict_indices_and_values(self, data, user, item):
        user_col = data.train_feat_indices.shape[1] - 2
        item_col = data.train_feat_indices.shape[1] - 1

        user_repr = user + data.user_offset
        user_cols = data.user_feature_cols + [user_col]
        user_features = data.train_feat_indices[:, user_cols]
        user = user_features[user_features[:, -1] == user_repr][0]

        item_repr = item + data.user_offset + data.n_users
        item_cols = [item_col] + data.item_feature_cols
        item_features = data.train_feat_indices[:, item_cols]
        item = item_features[item_features[:, 0] == item_repr][0]

        orig_cols = user_cols + item_cols
        col_reindex = np.array(range(len(orig_cols)))[np.argsort(orig_cols)]
        concat_indices = np.concatenate([user, item])[col_reindex]

        feat_values = np.ones(len(concat_indices))
        if data.numerical_col is not None:
            for col in range(len(data.numerical_col)):
                if col in data.user_feature_cols:
                    user_indices = np.where(data.train_feat_indices[:, user_col] == user_repr)[0]
                    numerical_values = data.train_feat_values[user_indices, col][0]
                    feat_values[col] = numerical_values
                elif col in data.item_feature_cols:
                    item_indices = np.where(data.train_feat_indices[:, item_col] == item_repr)[0]
                    numerical_values = data.train_feat_values[item_indices, col][0]
                    feat_values[col] = numerical_values

        return concat_indices.reshape(1, -1), feat_values.reshape(1, -1)

    def get_recommend_indices_and_values(self, data, user, items_unique):
        user_col = data.train_feat_indices.shape[1] - 2
        item_col = data.train_feat_indices.shape[1] - 1

        user_repr = user + data.user_offset
        user_cols = data.user_feature_cols + [user_col]
        user_features = data.train_feat_indices[:, user_cols]
        user_unique = user_features[user_features[:, -1] == user_repr][0]
        users = np.tile(user_unique, (data.n_items, 1))

        #   np.unique is sorted from starting with the first element, so put item col first
        item_cols = [item_col] + data.item_feature_cols
        orig_cols = user_cols + item_cols
        col_reindex = np.array(range(len(orig_cols)))[np.argsort(orig_cols)]

        assert users.shape[0] == items_unique.shape[0], "user shape must equal to num of candidate items"
        concat_indices = np.concatenate([users, items_unique], axis=-1)[:, col_reindex]

        #   construct feature values, mainly fill numerical columns
        feat_values = np.ones(shape=(data.n_items, concat_indices.shape[1]))
        if data.numerical_col is not None:
            numerical_dict = OrderedDict()
            for col in range(len(data.numerical_col)):
                if col in data.user_feature_cols:
                    user_indices = np.where(data.train_feat_indices[:, user_col] == user_repr)[0]
                    numerical_values = data.train_feat_values[user_indices, col][0]
                    numerical_dict[col] = numerical_values
                elif col in data.item_feature_cols:
                    # order according to item indices
                    numerical_map = OrderedDict(
                                        sorted(
                                            zip(data.train_feat_indices[:, -1],
                                                data.train_feat_values[:, col]), key=lambda x: x[0]))
                    numerical_dict[col] = [v for v in numerical_map.values()]

            for k, v in numerical_dict.items():
                feat_values[:, k] = np.array(v)

        return concat_indices, feat_values