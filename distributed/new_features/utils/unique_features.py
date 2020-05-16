import numpy as np


def construct_unique_item_feat(sparse_indices, dense_values, item_sparse_col):
    neg_item_sparse_matrix = _item_sparse_unique(sparse_indices, item_sparse_col)
#    assert len(neg_item_sparse_matrix) == data_info.n_items, "number of unique negative items should match n_items"
    assert len(sparse_indices) == len(dense_values), "length of sparse and dense columns must equal"
    if dense_values is not None:
        neg_item_dense_matrix = _item_dense_unique(sparse_indices, dense_values)
        return neg_item_sparse_matrix, neg_item_dense_matrix
    return neg_item_sparse_matrix, None


def _item_sparse_unique(sparse_indices, item_sparse_col):
#    item_sparse_mapping = dict()
#    all_items_sparse_unique = np.unique(train_data.sparse_indices[:, data_info.item_sparse_col], axis=0)
#    all_items = all_items_sparse_unique[:, 0] - self.n_items - 1
#    for item, feat in enumerate(all_items_sparse_unique):
#        item_sparse_mapping[item] = feat.tolist()
#    return item_sparse_mapping

    # np.unique(axis=0) will sort the data based on first column, so we can do direct mapping
    return np.unique(np.take(sparse_indices, item_sparse_col, axis=1), axis=0)


def _item_dense_unique(sparse_indices, dense_values):
#    item_dense_mapping = dict()
    item_indices = sparse_indices[:, 0].reshape(-1, 1)
    dense_values = dense_values.reshape(-1, 1) if dense_values.ndim == 1 else dense_values
    indices_plus_dense_values = np.concatenate([item_indices, dense_values], axis=-1)
#    all_items_dense_unique = np.unique(indices_plus_dense_values, axis=0)
#    for item, feat in enumerate(all_items_dense_unique[:, 1:]):
#        item_dense_mapping[item] = feat.tolist()
#    return item_dense_mapping
    return np.unique(indices_plus_dense_values, axis=0)[:, 1:]  # remove item_indices


