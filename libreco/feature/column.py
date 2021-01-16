import itertools
import numpy as np


def get_user_item_sparse_indices(data_class, data, mode="train"):
    user_indices = column_sparse_indices(
        data.user.to_numpy(), data_class.user_unique_vals, mode
    )
    item_indices = column_sparse_indices(
        data.item.to_numpy(), data_class.item_unique_vals, mode
    )
    return user_indices, item_indices


def merge_sparse_col(sparse_col, multi_sparse_col):
    flatten_cols = list(itertools.chain.from_iterable(multi_sparse_col))
    return flatten_cols if not sparse_col else sparse_col + flatten_cols


def merge_sparse_indices(data_class, data, sparse_col, multi_sparse_col, mode):
    if sparse_col and multi_sparse_col:
        sparse_indices = get_sparse_indices_matrix(
            data_class, data, sparse_col, mode
        )
        sparse_offset = get_sparse_offset(data_class, sparse_col)
        sparse_indices = sparse_indices + sparse_offset[:-1]

        multi_sparse_indices = get_multi_sparse_indices_matrix(
            data_class, data, multi_sparse_col, mode
        )
        multi_sparse_offset = get_multi_sparse_offset(
            data_class, multi_sparse_col
        )
        multi_sparse_indices = (
            multi_sparse_indices + sparse_offset[-1] + multi_sparse_offset
        )
        return np.concatenate([sparse_indices, multi_sparse_indices], axis=1)

    elif sparse_col:
        sparse_indices = get_sparse_indices_matrix(
            data_class, data, sparse_col, mode)
        sparse_offset = get_sparse_offset(data_class, sparse_col)
        return sparse_indices + sparse_offset[:-1]

    elif multi_sparse_col:
        multi_sparse_indices = get_multi_sparse_indices_matrix(
            data_class, data, multi_sparse_col, mode
        )
        multi_sparse_offset = get_multi_sparse_offset(
            data_class, multi_sparse_col
        )
        return multi_sparse_indices + multi_sparse_offset


def get_sparse_indices_matrix(data_class, data, sparse_col, mode="train"):
    n_samples, n_features = len(data), len(sparse_col)
    sparse_indices = np.zeros((n_samples, n_features), dtype=np.int32)
    for i, col in enumerate(sparse_col):
        col_values = data[col].to_numpy()
        unique_values = data_class.sparse_unique_vals[col]
        sparse_indices[:, i] = column_sparse_indices(
            col_values, unique_values, mode
        )
    return sparse_indices


def get_multi_sparse_indices_matrix(data_class, data, multi_sparse_col,
                                    mode="train"):
    n_samples = len(data)
    # n_fields = len(multi_sparse_col)
    n_features = len(list(itertools.chain.from_iterable(multi_sparse_col)))
    multi_sparse_indices = np.zeros((n_samples, n_features), dtype=np.int32)
    i = 0
    while i < n_features:
        for field in multi_sparse_col:
            unique_values = data_class.multi_sparse_unique_vals[field[0]]
            for col in field:
                col_values = data[col].to_numpy()
                multi_sparse_indices[:, i] = column_sparse_indices(
                    col_values, unique_values, mode
                )
                i += 1
    return multi_sparse_indices


def get_dense_indices_matrix(data, dense_col):
    n_samples, n_features = len(data), len(dense_col)
    dense_indices = np.tile(np.arange(n_features), [n_samples, 1])
    return dense_indices


def get_sparse_offset(data_class, sparse_col):
    # plus one for value only in test data
    unique_values = [
        len(data_class.sparse_unique_vals[col]) + 1
        for col in sparse_col
    ]
    return np.cumsum(np.array([0] + unique_values))


def get_multi_sparse_offset(data_class, multi_sparse_col):
    unique_values = [
        len(data_class.multi_sparse_unique_vals[field[0]]) + 1
        for field
        in multi_sparse_col
    ]
    field_offset = np.cumsum(np.array([0] + unique_values)).tolist()[:-1]
    offset = []
    for i, field in enumerate(multi_sparse_col):
        offset.extend([field_offset[i]] * len(field))
    return np.array(offset)


def check_unknown(values, uniques):
    # diff = list(np.setdiff1d(values, uniques, assume_unique=True))
    mask = np.in1d(values, uniques, invert=True)
    return mask


def column_sparse_indices(values, unique, mode="train"):
    if mode == "test":
        not_in_mask = check_unknown(values, unique)
        col_indices = np.searchsorted(unique, values)
        col_indices[not_in_mask] = len(unique)
    elif mode == "train":
        col_indices = np.searchsorted(unique, values)
    else:
        raise ValueError("mode must either be \"train\" or \"test\" ")
    return col_indices
