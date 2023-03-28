import itertools

import numpy as np


def get_multi_sparse_indices_matrix(
    data, multi_sparse_col, multi_sparse_unique, is_train, is_ordered
):
    """Get all multi_sparse indices for all samples in data.

    The function will consider each sub-feature of multi_sparse columns.

    Parameters
    ----------
    data : pandas.DataFrame
        The original data.
    multi_sparse_col : list of str
        All multi_sparse feature names.
    multi_sparse_unique : dict of {str : numpy.ndarray}
        Unique values of all multi_sparse features. Each sub-feature will use the first
        feature as representative.
    is_train : bool
        Whether the data is training data.
    is_ordered : bool
        Whether the `unique` values are sorted.

    Returns
    -------
    multi_sparse_indices : numpy.ndarray
        Multi_sparse indices of all multi_sparse features.
    """
    from .sparse import column_sparse_indices

    n_samples = len(data)
    n_features = len(list(itertools.chain.from_iterable(multi_sparse_col)))
    multi_sparse_indices = np.zeros((n_samples, n_features), dtype=np.int32)
    i = 0
    while i < n_features:
        for field in multi_sparse_col:
            unique_values = multi_sparse_unique[field[0]]
            for col in field:
                col_values = data[col].to_numpy()
                multi_sparse_indices[:, i] = column_sparse_indices(
                    col_values, unique_values, is_train, is_ordered, multi_sparse=True
                )
                i += 1
    return multi_sparse_indices


def get_multi_sparse_offset(multi_sparse_col, multi_sparse_unique):
    unique_values = [
        len(multi_sparse_unique[field[0]]) + 1 for field in multi_sparse_col
    ]
    field_offset = np.cumsum(np.array([0, *unique_values])).tolist()[:-1]
    offset = []
    # each sub-feature will use same offset
    for i, field in enumerate(multi_sparse_col):
        offset.extend([field_offset[i]] * len(field))
    return np.array(offset)


def multi_sparse_oov(multi_sparse_col, multi_sparse_unique, extend=True):
    unique_values = [
        len(multi_sparse_unique[field[0]]) + 1 for field in multi_sparse_col
    ]
    field_oov = np.cumsum(unique_values) - 1
    if extend:
        oov = []
        for i, field in enumerate(multi_sparse_col):
            oov.extend([field_oov[i]] * len(field))
        return np.array(oov)
    else:
        return field_oov


def get_multi_sparse_info(
    all_sparse_cols,
    sparse_col,
    multi_sparse_col,
    sparse_unique,
    multi_sparse_unique,
    pad_val,
):
    from .sparse import get_last_offset
    from ..data import MultiSparseInfo

    if not multi_sparse_col:
        return
    field_offset = [all_sparse_cols.index(field[0]) for field in multi_sparse_col]
    field_length = [len(col) for col in multi_sparse_col]
    feat_oov = multi_sparse_oov(multi_sparse_col, multi_sparse_unique, extend=False)
    if sparse_col:
        sparse_last_offset = get_last_offset(sparse_col, sparse_unique)
        feat_oov += sparse_last_offset
    return MultiSparseInfo(field_offset, field_length, feat_oov, pad_val)


def multi_sparse_col_map(multi_sparse_col):
    """Map sub-features in multi-sparse features to main-features.

    Each multi-sparse feature will use its first sub-feature as representative.
    This function maps the rest sub-features to the representative sub-feature.

    Parameters
    ----------
    multi_sparse_col : list of [list of str]
        All multi_sparse feature names.

    Returns
    -------
    dict of {str : str}
    """
    multi_sparse_map = dict()
    for field in multi_sparse_col:
        if len(field) > 1:
            for col in field[1:]:
                multi_sparse_map[col] = field[0]
    return multi_sparse_map


def recover_sparse_cols(data_info):
    """Get the original nested multi_sparse columns from data_info."""
    total_sparse_cols = data_info.sparse_col.name
    sparse_cols, multi_sparse_cols = None, None
    if data_info.sparse_unique_vals:
        sparse_cols = [
            col for col in total_sparse_cols if col in data_info.sparse_unique_vals
        ]
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
    return sparse_cols, multi_sparse_cols


def true_sparse_field_size(data_info, sparse_field_size, combiner):
    """Get the real sparse field size.

    When using multi_sparse_combiner, field size will decrease.
    """
    if data_info.multi_sparse_combine_info and combiner in ("sum", "mean", "sqrtn"):
        field_length = data_info.multi_sparse_combine_info.field_len
        return sparse_field_size - (sum(field_length) - len(field_length))
    else:
        return sparse_field_size
