from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


class FeatureBuilder:
    def __init__(self, value_sharing=False):
        self.value_sharing = value_sharing

    def fit(self, categorical_features, numerical_features, train_size):
        total_count = 0  # add user & item indices before/after
        feature_indices = []
        feature_values = []
        for k, v in numerical_features.items():
            feature_indices.append([total_count] * train_size)
            feature_values.append(v)
            total_count += 1

        self.val_index_dict = defaultdict(dict)
        for k, v in categorical_features.items():
            unique_vals, indices = np.unique(v, return_inverse=True)
            unique_vals_length = len(unique_vals)
            indices += total_count
            self.val_index_dict[k].update(zip(unique_vals, indices))
            feature_indices.append(indices.tolist())
            feature_values.append([1.0] * train_size)
            total_count += unique_vals_length

        feature_indices = np.array(feature_indices).T.astype(np.int32)
        feature_values = np.array(feature_values).T.astype(np.float32)
        self.feature_size = total_count
        return feature_indices, feature_values, self.feature_size

    def transform(self, test_cat_feat, test_num_feat, test_size):
        test_feature_indices = []
        test_feature_values = []
        total_count = 0
        for k, v in test_num_feat.items():
            test_feature_indices.append([total_count] * test_size)
            test_feature_values.append(v)
            total_count += 1

        for k, v in test_cat_feat.items():
            indices = pd.Series(v).map(self.val_index_dict[k])
            indices = indices.fillna(self.feature_size)
            test_feature_indices.append(indices.tolist())
            test_feature_values.append([1.0] * test_size)

        test_feature_indices = np.array(test_feature_indices).T.astype(np.int32)
        test_feature_values = np.array(test_feature_values).T.astype(np.float32)
        return test_feature_indices, test_feature_values















