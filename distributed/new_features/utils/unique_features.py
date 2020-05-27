import numpy as np


def construct_unique_item_feat(sparse_indices, dense_values, item_sparse_col, item_dense_col):
    neg_item_sparse_matrix = _item_sparse_unique(sparse_indices, item_sparse_col)
    if dense_values is None or not item_dense_col:
        return neg_item_sparse_matrix, None
    assert len(sparse_indices) == len(dense_values), "length of sparse and dense columns must equal"
    neg_item_dense_matrix = _item_dense_unique(sparse_indices, dense_values, item_dense_col)
    return neg_item_sparse_matrix, neg_item_dense_matrix


def _item_sparse_unique(sparse_indices, item_sparse_col):
    # np.unique(axis=0) will sort the data based on first column, so we can do direct mapping
    return np.unique(np.take(sparse_indices, item_sparse_col, axis=1), axis=0)


def _item_dense_unique(sparse_indices, dense_values, item_dense_col):
    item_indices = sparse_indices[:, 1].reshape(-1, 1)
    dense_values = np.take(dense_values, item_dense_col, axis=1)
    dense_values = dense_values.reshape(-1, 1) if dense_values.ndim == 1 else dense_values
    indices_plus_dense_values = np.concatenate([item_indices, dense_values], axis=-1)
    return np.unique(indices_plus_dense_values, axis=0)[:, 1:]  # remove redundant item_indices

