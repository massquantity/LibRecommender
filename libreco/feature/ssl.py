"""Feature Generation for Self-Supervised Learning."""
import numpy as np


def rfm_mask_features(model, batch_size, ssl_pattern):
    ssl_feats = dict()
    rng, n_items = model.data_info.np_rng, model.n_items
    replace = False if batch_size < n_items else True
    item_indices = rng.choice(n_items, size=batch_size, replace=replace)
    feat_indices = model.data_info.item_sparse_unique[item_indices]
    # add offset since default embedding has 0 index
    sparse_indices = np.hstack(
        [np.expand_dims(item_indices + 1, 1), feat_indices + n_items + 1]
    )
    feat_num = sparse_indices.shape[1]
    mid_point = feat_num // 2
    if ssl_pattern.endswith("complementary"):
        random_cols = rng.permutation(feat_num)
        left_cols, right_cols = np.split(random_cols, [mid_point])
    else:
        left_cols = rng.permutation(feat_num)[:mid_point]
        right_cols = rng.permutation(feat_num)[:mid_point]

    left_sparse_indices = sparse_indices.copy()
    left_sparse_indices[:, left_cols] = 0
    ssl_feats.update({model.ssl_left_sparse_indices: left_sparse_indices})
    right_sparse_indices = sparse_indices.copy()
    right_sparse_indices[:, right_cols] = 0
    ssl_feats.update({model.ssl_right_sparse_indices: right_sparse_indices})

    if model.item_dense:
        dense_values = model.data_info.item_dense_unique[item_indices]
        ssl_feats.update({model.ssl_left_dense_values: dense_values})
        ssl_feats.update({model.ssl_right_dense_values: dense_values})
    return ssl_feats
