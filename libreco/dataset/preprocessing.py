from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


class FeatureBuilder:
    def __init__(self, value_sharing=False, include_user=True, include_item=True,
                 n_users=None, n_items=None):
        self.value_sharing = value_sharing
        self.include_user = include_user
        self.include_item = include_item
        self.n_users = n_users
        self.n_items = n_items

    def fit(self, categorical_features, numerical_features, merged_cat_features,
            train_size, user_features=None, item_features=None):
        self.total_count = 0  # add user & item indices before/after
        feature_indices = []
        feature_values = []
        for k in sorted(numerical_features):
            v = numerical_features[k]
            feature_indices.append([self.total_count] * train_size)
            feature_values.append(v)
            self.total_count += 1

        self.val_index_dict = defaultdict(dict)
        for k in sorted(categorical_features):  # preserve column orders
            v = categorical_features[k]
            unique_vals, indices = np.unique(v, return_inverse=True)
            unique_vals_length = len(unique_vals)
            indices += self.total_count
            self.val_index_dict[k].update(zip(unique_vals, np.unique(indices)))
            feature_indices.append(indices.tolist())
            feature_values.append([1.0] * train_size)
            self.total_count += unique_vals_length

        self.merge_val_index_dict = defaultdict(dict)
        for k in sorted(merged_cat_features):
            v = merged_cat_features[k]
            if len(v) % train_size == 0:
                unique_vals, indices = np.unique(v, return_inverse=True)
                unique_vals_length = len(unique_vals)
                indices += self.total_count
                self.merge_val_index_dict[k].update(zip(unique_vals, np.unique(indices)))

                feature_count = len(v) // train_size
                for i in range(feature_count):
                    feature_indices.append(indices.tolist()[train_size * i: train_size * (i + 1)])
                    feature_values.append([1.0] * train_size)
                self.total_count += unique_vals_length
            else:
                raise ValueError("feature total length must be integral multiple of train data size")

        self.feature_size = self.total_count  # preserve total_count for transform function
        if self.include_user:
            feature_indices.append(user_features + self.feature_size)
            self.feature_size += self.n_users
            feature_values.append([1.0] * train_size)
        if self.include_item:
            feature_indices.append(item_features + self.feature_size)
            self.feature_size += self.n_items
            feature_values.append([1.0] * train_size)

        feature_indices = np.array(feature_indices).T.astype(np.int32)
        feature_values = np.array(feature_values).T.astype(np.float32)
        return feature_indices, feature_values, self.feature_size

    def transform(self, test_cat_feat, test_num_feat, test_merge_feat,
                  test_size, test_user_features=None, test_item_features=None):
        test_feature_indices = []
        test_feature_values = []
        test_total_count = 0
        for k, v in test_num_feat.items():
            test_feature_indices.append([test_total_count] * test_size)
            test_feature_values.append(v)
            test_total_count += 1

        for k, v in test_cat_feat.items():
            indices = pd.Series(v).map(self.val_index_dict[k])
            indices = indices.fillna(self.feature_size)  # fillna with last single index
            test_feature_indices.append(indices.tolist())
            test_feature_values.append([1.0] * test_size)

        for k, v in test_merge_feat.items():
            if len(v) % test_size == 0:
                indices = pd.Series(v).map(self.merge_val_index_dict[k])
                indices = indices.fillna(self.feature_size)

                feature_count = len(v) // test_size
                for i in range(feature_count):
                    test_feature_indices.append(indices.tolist()[test_size * i: test_size * (i + 1)])
                    test_feature_values.append([1.0] * test_size)
            else:
                raise ValueError("feature total length must be integral multiple of test data size")

        if self.include_user:
            test_feature_indices.append(test_user_features + self.total_count)
            test_feature_values.append([1.0] * test_size)
        if self.include_item:
            test_feature_indices.append(test_item_features + self.total_count + self.n_users)
            test_feature_values.append([1.0] * test_size)

        test_feature_indices = np.array(test_feature_indices).T.astype(np.int32)
        test_feature_values = np.array(test_feature_values).T.astype(np.float32)
        return test_feature_indices, test_feature_values










