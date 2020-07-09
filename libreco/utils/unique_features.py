import numbers
import numpy as np


def construct_unique_feat(user_indices, item_indices, sparse_indices,
                          dense_values, user_sparse_col, user_dense_col,
                          item_sparse_col, item_dense_col):

    if user_sparse_col:
        user_sparse_matrix = _compress_unique_values(
            sparse_indices, user_sparse_col, user_indices)
    else:
        user_sparse_matrix = None

    if item_sparse_col:
        item_sparse_matrix = _compress_unique_values(
            sparse_indices, item_sparse_col, item_indices)
    else:
        item_sparse_matrix = None

    if user_dense_col:
        user_dense_matrix = _compress_unique_values(
            dense_values, user_dense_col, user_indices)
    else:
        user_dense_matrix = None

    if item_dense_col:
        item_dense_matrix = _compress_unique_values(
            dense_values, item_dense_col, item_indices)
    else:
        item_dense_matrix = None

    rets = (user_sparse_matrix, user_dense_matrix,
            item_sparse_matrix, item_dense_matrix)
    return rets


def _compress_unique_values(orig_val, col, unique_indices):
    values = np.take(orig_val, col, axis=1)
    values = values.reshape(-1, 1) if orig_val.ndim == 1 else values
    unique_indices = unique_indices.reshape(-1, 1)
    indices_plus_values = np.concatenate([unique_indices, values], axis=-1)
    # np.unique(axis=0) will sort the data based on first column,
    # so we can do direct mapping, then remove redundant unique_indices
    return np.unique(indices_plus_values, axis=0)[:, 1:]


def get_predict_indices_and_values(data_info, user, item, n_items,
                                   sparse, dense):
    if isinstance(user, numbers.Integral):
        user = list([user])
    if isinstance(item, numbers.Integral):
        item = list([item])

    sparse_indices = get_sparse_indices(
        data_info, user, item, mode="predict") if sparse else None
    dense_values = get_dense_values(
        data_info, user, item, mode="predict") if dense else None
    if sparse and dense:
        assert len(sparse_indices) == len(dense_values), (
            "indices and values length must equal")

    return user, item, sparse_indices, dense_values


def get_recommend_indices_and_values(data_info, user, n_items, sparse, dense):
    user_indices = np.repeat(user, n_items)
    item_indices = np.arange(n_items)

    sparse_indices = get_sparse_indices(
        data_info, user, n_items=n_items, mode="recommend") if sparse else None
    dense_values = get_dense_values(
        data_info, user, n_items=n_items, mode="recommend") if dense else None
    if sparse and dense:
        assert len(sparse_indices) == len(dense_values), (
            "indices and values length must equal")

    return user_indices, item_indices, sparse_indices, dense_values


def get_sparse_indices(data_info, user, item=None, n_items=None,
                       mode="predict"):
    user_sparse_col = data_info.user_sparse_col.index
    item_sparse_col = data_info.item_sparse_col.index
    orig_cols = user_sparse_col + item_sparse_col
    # keep column names in original order
    col_reindex = np.arange(len(orig_cols))[np.argsort(orig_cols)]

    if mode == "predict":
        if user_sparse_col and item_sparse_col:
            user_sparse_part = data_info.user_sparse_unique[user]
            item_sparse_part = data_info.item_sparse_unique[item]
            sparse_indices = np.concatenate(
                [user_sparse_part, item_sparse_part], axis=-1)[:, col_reindex]
            return sparse_indices
        elif user_sparse_col:
            return data_info.user_sparse_unique[user]
        elif item_sparse_col:
            return data_info.item_sparse_unique[item]

    elif mode == "recommend":
        if user_sparse_col and item_sparse_col:
            user_sparse_part = np.tile(data_info.user_sparse_unique[user],
                                       (n_items, 1))
            item_sparse_part = data_info.item_sparse_unique
            sparse_indices = np.concatenate(
                [user_sparse_part, item_sparse_part], axis=-1)[:, col_reindex]
            return sparse_indices
        elif user_sparse_col:
            return np.tile(data_info.user_sparse_unique[user], (n_items, 1))
        elif item_sparse_col:
            return data_info.item_sparse_unique


def get_dense_indices(data_info, user, n_items=None, mode="predict"):
    user_dense_col = data_info.user_dense_col.index
    item_dense_col = data_info.item_dense_col.index
    total_dense_cols = len(user_dense_col) + len(item_dense_col)
    if mode == "predict":
        return np.tile(np.arange(total_dense_cols), (len(user), 1))
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
            user_dense_part = np.tile(data_info.user_dense_unique[user],
                                      (n_items, 1))
            item_dense_part = data_info.item_dense_unique
            dense_values = np.concatenate(
                [user_dense_part, item_dense_part], axis=-1)[:, col_reindex]
            return dense_values
        elif user_dense_col:
            return np.tile(data_info.user_dense_unique[user], (n_items, 1))
        elif item_dense_col:
            return data_info.item_dense_unique

