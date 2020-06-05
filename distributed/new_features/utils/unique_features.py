import numbers
import numpy as np

"""
def construct_unique_item_feat(sparse_indices, dense_values, item_sparse_col, item_dense_col):
    neg_item_sparse_matrix = _item_sparse_unique(sparse_indices, item_sparse_col)
    if dense_values is None or not item_dense_col:
        return neg_item_sparse_matrix, None

    assert len(sparse_indices) == len(dense_values), "length of sparse and dense columns must be equal"
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
"""


def construct_unique_feat(sparse_indices, dense_values, user_sparse_col,
                          user_dense_col, item_sparse_col, item_dense_col):
    user_sparse_matrix = _sparse_unique(sparse_indices, user_sparse_col)
    item_sparse_matrix = _sparse_unique(sparse_indices, item_sparse_col)
    if dense_values is None or not item_dense_col:
        return user_sparse_matrix, None, item_sparse_matrix, None

    assert len(sparse_indices) == len(dense_values), (
        "length of sparse and dense columns must equal")

    user_dense_matrix = _dense_unique(sparse_indices, dense_values, user_dense_col, "user")
    item_dense_matrix = _dense_unique(sparse_indices, dense_values, item_dense_col, "item")
    return user_sparse_matrix, user_dense_matrix, item_sparse_matrix, item_dense_matrix


def _sparse_unique(sparse_indices, sparse_col):
    # np.unique(axis=0) will sort the data based on first column, so we can do direct mapping
    return np.unique(np.take(sparse_indices, sparse_col, axis=1), axis=0)


def _dense_unique(sparse_indices, dense_values, dense_col, unique_col):
    if unique_col == "user":
        unique_indices = sparse_indices[:, 0].reshape(-1, 1)
    elif unique_col == "item":
        unique_indices = sparse_indices[:, 1].reshape(-1, 1)

    dense_values = np.take(dense_values, dense_col, axis=1)
    dense_values = dense_values.reshape(-1, 1) if dense_values.ndim == 1 else dense_values
    indices_plus_dense_values = np.concatenate([unique_indices, dense_values], axis=-1)
    return np.unique(indices_plus_dense_values, axis=0)[:, 1:]  # remove redundant unique_indices


def get_predict_indices_and_values(data_info, user, item, n_items):
    if isinstance(user, numbers.Integral):
        user = list([user])
    if isinstance(item, numbers.Integral):
        item = list([item])
    sparse_indices = get_sparse_indices(data_info, user, item, mode="predict")
    dense_indices = get_dense_indices(data_info, n_items, mode="predict")
    dense_values = get_dense_values(data_info, user, item, mode="predict")
    assert len(sparse_indices) == len(dense_indices) == len(dense_values), (
        "indices and values length must equal")
    return sparse_indices, dense_indices, dense_values


def get_recommend_indices_and_values(data_info, user, n_items):
    sparse_indices = get_sparse_indices(data_info, user, n_items=n_items, mode="recommend")
    dense_indices = get_dense_indices(data_info, n_items=n_items, mode="recommend")
    dense_values = get_dense_values(data_info, user, n_items=n_items, mode="recommend")
    assert len(sparse_indices) == len(dense_indices) == len(dense_values), (
        "indices and values length must equal")
    return sparse_indices, dense_indices, dense_values


def get_sparse_indices(data_info, user, item=None, n_items=None, mode="predict"):
    user_sparse_col = data_info.user_sparse_col.index
    item_sparse_col = data_info.item_sparse_col.index
    orig_cols = user_sparse_col + item_sparse_col
    # keep column names in original order
    col_reindex = np.arange(len(orig_cols))[np.argsort(orig_cols)]

    if mode == "predict":
        user_sparse_part = data_info.user_sparse_unique[user]
        item_sparse_part = data_info.item_sparse_unique[item]
    elif mode == "recommend":
        user_sparse_part = np.tile(data_info.user_sparse_unique[user], (n_items, 1))
        item_sparse_part = data_info.item_sparse_unique

    sparse_indices = np.concatenate(
        [user_sparse_part, item_sparse_part], axis=-1)[:, col_reindex]
    return sparse_indices


def get_dense_indices(data_info, n_items=None, mode="predict"):
    user_dense_col = data_info.user_dense_col.index
    item_dense_col = data_info.item_dense_col.index
    total_dense_cols = len(user_dense_col) + len(item_dense_col)
    if mode == "predict":
        return np.arange(total_dense_cols).reshape(1, -1)
    elif mode == "recommend":
        return np.tile(np.arange(total_dense_cols), (n_items, 1))


def get_dense_values(data_info, user, item=None, n_items=None, mode="predict"):
    user_dense_col = data_info.user_dense_col.index
    item_dense_col = data_info.item_dense_col.index
    # keep column names in original order
    orig_cols = user_dense_col + item_dense_col
    col_reindex = np.arange(len(orig_cols))[np.argsort(orig_cols)]

    if mode == "predict":
        if user_dense_col and item_dense_col:
            user_dense_part = data_info.user_dense_unique[user]
            item_dense_part = data_info.item_dense_unique[item]
            dense_values = np.concatenate(
                [user_dense_part, item_dense_part], axis=-1)[:, col_reindex]
            return dense_values
        elif user_dense_col:
            return data_info.user_dense_unique[user]
        elif item_dense_col:
            return data_info.item_dense_unique[item]

    elif mode == "recommend":
        if user_dense_col and item_dense_col:
            user_dense_part = np.tile(data_info.user_dense_unique[user], (n_items, 1))
            item_dense_part = data_info.item_dense_unique
            dense_values = np.concatenate([user_dense_part, item_dense_part], axis=-1)[:, col_reindex]
            return dense_values
        elif user_dense_col:
            return np.tile(data_info.user_dense_unique[user], (n_items, 1))
        elif item_dense_col:
            return data_info.item_dense_unique



