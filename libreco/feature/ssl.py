"""Feature Generation for Self-Supervised Learning."""
import numpy as np
from sklearn.metrics import mutual_info_score


def get_ssl_features(model, batch_size):
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
    if model.ssl_pattern.startswith("cfm"):
        seed_col = rng.integers(feat_num)
        left_cols = model.sparse_feat_mutual_info[seed_col]
        right_cols = np.setdiff1d(range(feat_num), left_cols)
    elif model.ssl_pattern.endswith("complementary"):
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


def get_mutual_info(data, data_info):
    """Compute mutual information for each pair of item sparse features."""
    item_indices = np.expand_dims(data.item_indices, 1)
    feat_indices = data.sparse_indices[:, data_info.item_sparse_col.index]
    sparse_indices = np.hstack([item_indices, feat_indices])
    feat_num = sparse_indices.shape[1]
    pairwise_mutual_info = np.zeros((feat_num, feat_num))
    # assign self mutual info to impossible value
    np.fill_diagonal(pairwise_mutual_info, -1)
    for i in range(feat_num):
        for j in range(i + 1, feat_num):
            mi = mutual_info_score(sparse_indices[:, i], sparse_indices[:, j])
            pairwise_mutual_info[i][j] = pairwise_mutual_info[j][i] = mi

    n = feat_num // 2
    topn_mutual_info = np.argsort(pairwise_mutual_info, axis=1)[:, -n:]
    return {i: topn_mutual_info[i] for i in range(feat_num)}
