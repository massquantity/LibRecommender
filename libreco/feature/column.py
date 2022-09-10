from array import array
from collections import defaultdict
import itertools

import numpy as np


def get_user_item_sparse_indices(
    data, user_unique_vals, item_unique_vals, mode, ordered
):
    user_indices = column_sparse_indices(
        data.user.to_numpy(), user_unique_vals, mode, ordered
    )
    item_indices = column_sparse_indices(
        data.item.to_numpy(), item_unique_vals, mode, ordered
    )
    return user_indices, item_indices


def merge_sparse_col(sparse_col, multi_sparse_col):
    flatten_cols = list(itertools.chain.from_iterable(multi_sparse_col))
    return flatten_cols if not sparse_col else sparse_col + flatten_cols


def merge_sparse_indices(data_class, data, sparse_col, multi_sparse_col, mode, ordered):
    if sparse_col and multi_sparse_col:
        sparse_indices = get_sparse_indices_matrix(
            data_class, data, sparse_col, mode, ordered
        )
        sparse_offset = get_sparse_offset(data_class, sparse_col)
        sparse_indices = sparse_indices + sparse_offset[:-1]

        multi_sparse_indices = get_multi_sparse_indices_matrix(
            data_class, data, multi_sparse_col, mode, ordered
        )
        multi_sparse_offset = get_multi_sparse_offset(data_class, multi_sparse_col)
        multi_sparse_indices = (
            multi_sparse_indices + sparse_offset[-1] + multi_sparse_offset
        )
        return np.concatenate([sparse_indices, multi_sparse_indices], axis=1)

    elif sparse_col:
        sparse_indices = get_sparse_indices_matrix(
            data_class, data, sparse_col, mode, ordered
        )
        sparse_offset = get_sparse_offset(data_class, sparse_col)
        return sparse_indices + sparse_offset[:-1]

    elif multi_sparse_col:
        multi_sparse_indices = get_multi_sparse_indices_matrix(
            data_class, data, multi_sparse_col, mode, ordered
        )
        multi_sparse_offset = get_multi_sparse_offset(data_class, multi_sparse_col)
        return multi_sparse_indices + multi_sparse_offset


def get_sparse_indices_matrix(data_class, data, sparse_col, mode, ordered):
    n_samples, n_features = len(data), len(sparse_col)
    sparse_indices = np.zeros((n_samples, n_features), dtype=np.int32)
    for i, col in enumerate(sparse_col):
        col_values = data[col].to_numpy()
        unique_values = data_class.sparse_unique_vals[col]
        sparse_indices[:, i] = column_sparse_indices(
            col_values, unique_values, mode, ordered
        )
    return sparse_indices


def get_multi_sparse_indices_matrix(data_class, data, multi_sparse_col, mode, ordered):
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
                    col_values, unique_values, mode, ordered, multi_sparse=True
                )
                i += 1
    return multi_sparse_indices


# def get_dense_indices_matrix(data, dense_col):
#    n_samples, n_features = len(data), len(dense_col)
#    dense_indices = np.tile(np.arange(n_features), [n_samples, 1])
#    return dense_indices


def get_sparse_offset(data_class, sparse_col):
    # plus one for oov value
    unique_values = [len(data_class.sparse_unique_vals[col]) + 1 for col in sparse_col]
    return np.cumsum(np.array([0] + unique_values))


def get_multi_sparse_offset(data_class, multi_sparse_col):
    unique_values = [
        len(data_class.multi_sparse_unique_vals[field[0]]) + 1
        for field in multi_sparse_col
    ]
    field_offset = np.cumsum(np.array([0] + unique_values)).tolist()[:-1]
    offset = []
    for i, field in enumerate(multi_sparse_col):
        offset.extend([field_offset[i]] * len(field))
    return np.array(offset)


def merge_offset(data_class, sparse_col, multi_sparse_col):
    if sparse_col and multi_sparse_col:
        sparse_offset = get_sparse_offset(data_class, sparse_col)
        multi_sparse_offset = (
            get_multi_sparse_offset(data_class, multi_sparse_col) + sparse_offset[-1]
        )
        return np.concatenate([sparse_offset[:-1], multi_sparse_offset])
    elif sparse_col:
        sparse_offset = get_sparse_offset(data_class, sparse_col)
        return sparse_offset[:-1]
    elif multi_sparse_col:
        multi_sparse_offset = get_multi_sparse_offset(data_class, multi_sparse_col)
        return multi_sparse_offset


def sparse_oov(data_class, sparse_col):
    unique_values = [len(data_class.sparse_unique_vals[col]) + 1 for col in sparse_col]
    return np.cumsum(unique_values) - 1


def multi_sparse_oov(data_class, multi_sparse_col, extend=True):
    unique_values = [
        len(data_class.multi_sparse_unique_vals[field[0]]) + 1
        for field in multi_sparse_col
    ]
    # return np.cumsum(unique_values) - 1
    field_oov = np.cumsum(unique_values) - 1
    if extend:
        oov = []
        for i, field in enumerate(multi_sparse_col):
            oov.extend([field_oov[i]] * len(field))
        return np.array(oov)
    else:
        return field_oov


def get_oov_pos(data_class, sparse_col, multi_sparse_col):
    if sparse_col and multi_sparse_col:
        sparse = sparse_oov(data_class, sparse_col)
        multi_sparse = multi_sparse_oov(data_class, multi_sparse_col) + sparse[-1] + 1
        return np.concatenate([sparse, multi_sparse])
    elif sparse_col:
        return sparse_oov(data_class, sparse_col)
    elif multi_sparse_col:
        return multi_sparse_oov(data_class, multi_sparse_col)


def multi_sparse_combine_info(
    data_class, all_sparse_cols, sparse_col, multi_sparse_col
):
    from ..data import MultiSparseInfo

    # sparse_last_offset = get_sparse_offset(data_class, sparse_col)[-1]
    # unique_values = [
    #    len(data_class.multi_sparse_unique_vals[field[0]]) + 1
    #    for field
    #    in multi_sparse_col
    # ]
    # offset = np.cumsum(np.array([0] + unique_values))
    # offset = offset.tolist()[:-1] + sparse_last_offset

    # all_sparse_cols = data_class.sparse_col.name
    field_offset = [all_sparse_cols.index(field[0]) for field in multi_sparse_col]
    field_length = [len(col) for col in multi_sparse_col]
    sparse_offset = sparse_oov(data_class, sparse_col)[-1] + 1 if sparse_col else 0
    feat_oov = multi_sparse_oov(data_class, multi_sparse_col, extend=False)
    return MultiSparseInfo(field_offset, field_length, feat_oov + sparse_offset)


def multi_sparse_true_size(data, multi_sparse_col, pad_val):
    sizes = [
        np.sum(data[field].to_numpy() != pad_val, axis=1) for field in multi_sparse_col
    ]
    return np.vstack(sizes)


def check_unknown(values, uniques):
    # diff = list(np.setdiff1d(values, uniques, assume_unique=True))
    mask = np.in1d(values, uniques, invert=True)
    return mask


def column_sparse_indices(
    values, unique, mode="train", ordered=True, multi_sparse=False
):
    if mode not in ("train", "test"):
        raise ValueError('mode must either be "train" or "test" ')
    if ordered:
        if mode == "test" or multi_sparse:
            not_in_mask = check_unknown(values, unique)
            col_indices = np.searchsorted(unique, values)
            col_indices[not_in_mask] = len(unique)
        else:
            col_indices = np.searchsorted(unique, values)
    else:
        map_vals = dict(zip(unique, range(len(unique))))
        oov_val = len(unique)
        if mode == "test" or multi_sparse:
            col_indices = np.array(
                [map_vals[v] if v in map_vals else oov_val for v in values]
            )
        else:
            col_indices = np.array([map_vals[v] for v in values])
    return col_indices


def interaction_consumed(user_indices, item_indices):
    user_consumed = defaultdict(lambda: array("I"))
    item_consumed = defaultdict(lambda: array("I"))
    for u, i in zip(user_indices, item_indices):
        user_consumed[u].append(i)
        item_consumed[i].append(u)
    return user_consumed, item_consumed


def multi_sparse_col_map(multi_sparse_col):
    multi_sparse_map = dict()
    for field in multi_sparse_col:
        if len(field) > 1:
            for col in field[1:]:
                multi_sparse_map[col] = field[0]
    return multi_sparse_map


def recover_sparse_cols(data_info):
    total_sparse_cols = data_info.sparse_col.name
    if data_info.sparse_unique_vals:
        sparse_cols = [
            col for col in total_sparse_cols if col in data_info.sparse_unique_vals
        ]
    else:
        sparse_cols = None

    if data_info.multi_sparse_unique_vals:
        multi_sparse_cols = []
        i, field = 0, 0
        while i < len(total_sparse_cols):
            col = total_sparse_cols[i]
            if col in data_info.multi_sparse_unique_vals:
                field_len = data_info.multi_sparse_combine_info.field_len[field]
                multi_sparse_cols.append(
                    [total_sparse_cols[k] for k in range(i, i + field_len)]
                )
                i += field_len
                field += 1
            else:
                i += 1
    else:
        multi_sparse_cols = None

    return sparse_cols, multi_sparse_cols
