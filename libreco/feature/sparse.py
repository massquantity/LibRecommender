import itertools

import numpy as np

from .multi_sparse import (
    get_multi_sparse_indices_matrix,
    get_multi_sparse_offset,
    multi_sparse_oov,
)


def column_sparse_indices(values, unique, is_train, is_ordered, multi_sparse=False):
    """Assign indices for each value in a feature.

    The indices are mapped into integer indices based on the unique values of the feature.
    The generated indices are commonly used in an embedding layer.

    Parameters
    ----------
    values : array_like
        Feature values of all samples.
    unique : numpy.ndarray
        All unique values of a feature.
    is_train : bool
        Whether the feature values are from the training data.
        Values should all belong to `unique` values if they come from the training data.
    is_ordered : bool
        Whether the `unique` values are sorted. If they are sorted, `numpy.searchsorted`
        will be used, which has O(logN) time complexity.
    multi_sparse : bool
        Whether values come from multi_sparse features. Multi_sparse features may contain
        padding values, which don't exist in `unique` values.

    Returns
    -------
    indices : numpy.ndarray
        The mapped indices.
    """
    oov_val = len(unique)
    if is_ordered:
        col_indices = np.searchsorted(unique, values)
        if not is_train or multi_sparse:
            not_in_mask = np.in1d(values, unique, invert=True)
            col_indices[not_in_mask] = oov_val
    else:
        idx_mapping = dict(zip(unique, range(len(unique))))
        if not is_train or multi_sparse:
            col_indices = np.array(
                [idx_mapping[v] if v in idx_mapping else oov_val for v in values]
            )
        else:
            col_indices = np.array([idx_mapping[v] for v in values])
    return col_indices


def get_id_indices(data, user_unique_vals, item_unique_vals, is_train, is_ordered):
    user_indices = column_sparse_indices(
        data["user"].to_numpy(), user_unique_vals, is_train, is_ordered
    )
    item_indices = column_sparse_indices(
        data["item"].to_numpy(), item_unique_vals, is_train, is_ordered
    )
    return user_indices, item_indices


def merge_sparse_col(sparse_col, multi_sparse_col):
    """Merge all sparse and multi_sparse columns together."""
    flatten_cols = list(itertools.chain.from_iterable(multi_sparse_col))
    return flatten_cols if not sparse_col else sparse_col + flatten_cols


def merge_sparse_indices(
    data,
    sparse_col,
    multi_sparse_col,
    sparse_unique,
    multi_sparse_unique,
    is_train,
    is_ordered,
):
    """Merge all sparse and multi_sparse indices together.

    Sparse features will always be ahead of multi_sparse features.

    Parameters
    ----------
    data : pandas.DataFrame
        The original data.
    sparse_col : list of str
        All sparse feature names.
    multi_sparse_col : list of [list of str]
        All multi_sparse feature names.
    sparse_unique : dict of {str : numpy.ndarray}
        Unique values of all sparse features.
    multi_sparse_unique : dict of {str : numpy.ndarray}
        Unique values of all multi_sparse features.
    is_train : bool
        Whether the data is training data.
    is_ordered : bool
        Whether the `unique` values are sorted.

    Returns
    -------
    sparse_indices : numpy.ndarray
        Sparse indices of all sparse and multi_sparse features.
    """
    sparse_indices, multi_sparse_indices = None, None
    if sparse_col:
        sparse_indices = get_sparse_indices_matrix(
            data, sparse_col, sparse_unique, is_train, is_ordered
        )
        sparse_offset = get_sparse_offset(sparse_col, sparse_unique)
        sparse_indices = sparse_indices + sparse_offset
    if multi_sparse_col:
        multi_sparse_indices = get_multi_sparse_indices_matrix(
            data, multi_sparse_col, multi_sparse_unique, is_train, is_ordered
        )
        multi_sparse_offset = get_multi_sparse_offset(
            multi_sparse_col, multi_sparse_unique
        )
        multi_sparse_indices = multi_sparse_offset + multi_sparse_indices
    if sparse_col and multi_sparse_col:
        sparse_last_offset = get_last_offset(sparse_col, sparse_unique)
        multi_sparse_indices += sparse_last_offset
        return np.concatenate([sparse_indices, multi_sparse_indices], axis=1)
    return sparse_indices if sparse_col else multi_sparse_indices


def get_sparse_indices_matrix(data, sparse_col, sparse_unique, is_train, is_ordered):
    """Get all sparse indices for all samples in data.

    Parameters
    ----------
    data : pandas.DataFrame
        The original data.
    sparse_col : list of str
        All sparse feature names.
    sparse_unique : dict of {str : numpy.ndarray}
        Unique values of all sparse features.
    is_train : bool
        Whether the data is training data.
    is_ordered : bool
        Whether the `unique` values are sorted.

    Returns
    -------
    sparse_indices : numpy.ndarray
        Sparse indices of all sparse features.
    """
    n_samples, n_features = len(data), len(sparse_col)
    sparse_indices = np.zeros((n_samples, n_features), dtype=np.int32)
    for i, col in enumerate(sparse_col):
        col_values = data[col].to_numpy()
        unique_values = sparse_unique[col]
        sparse_indices[:, i] = column_sparse_indices(
            col_values, unique_values, is_train, is_ordered
        )
    return sparse_indices


def get_sparse_offset(sparse_col, sparse_unique):
    # plus one for oov value
    unique_values = [len(sparse_unique[col]) + 1 for col in sparse_col]
    return np.cumsum(np.array([0, *unique_values]))[:-1]


def get_last_offset(sparse_col, sparse_unique):
    """Last offset is used as the starting point of multi_sparse features."""
    return np.sum([len(sparse_unique[col]) + 1 for col in sparse_col])


def merge_offset(sparse_col, multi_sparse_col, sparse_unique, multi_sparse_unique):
    if not sparse_col and not multi_sparse_col:
        return
    sparse_offset = get_sparse_offset(sparse_col, sparse_unique) if sparse_col else None
    multi_sparse_offset = (
        get_multi_sparse_offset(multi_sparse_col, multi_sparse_unique)
        if multi_sparse_col
        else None
    )
    if sparse_col and multi_sparse_col:
        sparse_last_offset = get_last_offset(sparse_col, sparse_unique)
        multi_sparse_offset += sparse_last_offset
        return np.concatenate([sparse_offset, multi_sparse_offset])
    return sparse_offset if sparse_col else multi_sparse_offset


def sparse_oov(sparse_col, sparse_unique):
    """Get oov position for all sparse features."""
    unique_values = [len(sparse_unique[col]) + 1 for col in sparse_col]
    return np.cumsum(unique_values) - 1


def get_oov_pos(sparse_col, multi_sparse_col, sparse_unique, multi_sparse_unique):
    if not sparse_col and not multi_sparse_col:
        return
    sparse = sparse_oov(sparse_col, sparse_unique) if sparse_col else None
    multi_sparse = (
        multi_sparse_oov(multi_sparse_col, multi_sparse_unique)
        if multi_sparse_col
        else None
    )
    if sparse_col and multi_sparse_col:
        sparse_last_offset = get_last_offset(sparse_col, sparse_unique)
        multi_sparse += sparse_last_offset
        return np.concatenate([sparse, multi_sparse])
    return sparse if sparse_col else multi_sparse
