from collections import defaultdict

import numpy as np

from .sparse import column_sparse_indices


def update_unique_vals(data, old_unique_vals, pad_val=None):
    diff = np.setdiff1d(data, old_unique_vals)
    if pad_val is not None:
        diff = diff[diff != pad_val]
    if len(diff) > 0:
        return np.append(old_unique_vals, diff)
    else:
        return old_unique_vals


def update_id_unique(data, data_info):
    user_data = np.unique(data["user"])
    item_data = np.unique(data["item"])
    user_unique = update_unique_vals(user_data, data_info.user_unique_vals)
    item_unique = update_unique_vals(item_data, data_info.item_unique_vals)
    return user_unique, item_unique


def update_sparse_unique(data, data_info):
    if not data_info.sparse_unique_vals:
        return
    all_sparse_col = data_info.sparse_col.name
    old_sps = data_info.sparse_unique_vals
    sp_unique = dict()
    for col in all_sparse_col:
        if col not in data.columns:
            raise ValueError(f"Old column `{col}` doesn't exist in new data")
        if col in old_sps:
            sp_unique[col] = update_unique_vals(np.unique(data[col]), old_sps[col])
    return sp_unique


def update_multi_sparse_unique(data, data_info):
    if not data_info.multi_sparse_unique_vals:
        return
    all_sparse_col = data_info.sparse_col.name
    old_multi_sps = data_info.multi_sparse_unique_vals
    multi_sp_unique = dict()
    multi_sp_col_mapping = data_info.col_name_mapping["multi_sparse"]
    multi_sp_data = defaultdict(list)
    for col in all_sparse_col:
        if col not in data.columns:
            raise ValueError(f"Old column `{col}` doesn't exist in new data")
        if col in old_multi_sps:
            multi_sp_data[col].extend(np.unique(data[col]))
        elif col in multi_sp_col_mapping:
            main_col = multi_sp_col_mapping[col]
            multi_sp_data[main_col].extend(np.unique(data[col]))

    pad_val_dict = data_info.multi_sparse_combine_info.pad_val
    for col, multi_data in multi_sp_data.items():
        multi_sp_unique[col] = update_unique_vals(
            multi_data, old_multi_sps[col], pad_val_dict[col]
        )
    return multi_sp_unique


def update_unique_feats(
    data,
    data_info,
    unique_ids,
    sparse_unique,
    multi_sparse_unique,
    sparse_offset,
    sparse_oov,
    is_user,
):
    """Update and add unique features for all users or items.

    If a user or item has duplicate features, only the last one will be used for updating.

    Parameters
    ----------
    data : pandas.DataFrame
        The original data.
    data_info : :class:`~libreco.data.DataInfo` object
        Object that contains useful information for training and inference.
    unique_ids : numpy.ndarray
        All the unique users or items.
    sparse_unique : dict of {str : numpy.ndarray}
        Unique values of all sparse features.
    multi_sparse_unique : dict of {str : numpy.ndarray}
        Unique values of all multi_sparse features.
    sparse_offset : numpy.ndarray
        Offset for all sparse and multi_sparse features, ordered by [sparse, multi_sparse].
    sparse_oov : numpy.ndarray
        Padding indices(out-of-vocabulary, oov) for all sparse and multi_sparse features.
    is_user : bool
        Whether to update user or item features.

    Returns
    -------
    tuple of (numpy.ndarray,)
        Updated unique feature matrices.
    """
    col = "user" if is_user else "item"
    data = data.drop_duplicates(subset=[col], keep="last")
    new_num = len(unique_ids)
    sp_col_info = data_info.user_sparse_col if is_user else data_info.item_sparse_col
    ds_col_info = data_info.user_dense_col if is_user else data_info.item_dense_col
    sparse_feats = get_sparse_feats(
        data_info, sparse_offset, sparse_oov, new_num, sp_col_info.index, is_user
    )
    dense_feats = get_dense_feats(data_info, new_num, is_user)
    row_idx, id_mask = get_row_id_masks(data[col], unique_ids)
    sparse_feats = update_new_sparse_feats(
        data,
        row_idx,
        id_mask,
        sparse_feats,
        sparse_unique,
        multi_sparse_unique,
        sp_col_info,
        data_info.col_name_mapping,
        sparse_offset,
    )
    dense_feats = update_new_dense_feats(
        data, row_idx, id_mask, dense_feats, ds_col_info
    )
    return sparse_feats, dense_feats


def get_sparse_feats(data_info, sparse_offset, sparse_oov, new_num, col_idxs, is_user):
    old_sp = data_info.user_sparse_unique if is_user else data_info.item_sparse_unique
    if old_sp is None:
        return
    old_sp = old_sp[:-1]  # exclude last oov unique values
    new_sp = adjust_offsets(data_info, old_sp, sparse_offset, col_idxs)
    new_sp = update_oovs(data_info, old_sp, new_sp, sparse_oov, col_idxs)
    assert new_num >= len(old_sp)
    if new_num > len(old_sp):
        diff = new_num - len(old_sp)
        oovs = sparse_oov[col_idxs]
        new_vals = np.full([diff, old_sp.shape[1]], oovs, old_sp.dtype)
        new_sp = np.vstack([new_sp, new_vals])
    return new_sp


def get_dense_feats(data_info, new_num, is_user):
    old_ds = data_info.user_dense_unique if is_user else data_info.item_dense_unique
    if old_ds is None:
        return
    new_ds = old_ds[:-1]
    if new_num > len(new_ds):
        diff = new_num - len(new_ds)
        new_vals = np.zeros([diff, old_ds.shape[1]], old_ds.dtype)
        new_ds = np.vstack([new_ds, new_vals])
    return new_ds


# sparse indices and offset will increase if sparse features encounter new categories
def adjust_offsets(data_info, old_sparse, sparse_offset, col_idxs):
    old_offset = data_info.sparse_offset
    diff = sparse_offset[col_idxs] - old_offset[col_idxs]
    return old_sparse + diff


def update_oovs(data_info, old_sparse, new_sparse, sparse_oov, col_idxs):
    old_oov = data_info.sparse_oov
    for i, col in enumerate(col_idxs):
        mask = old_sparse[:, i] == old_oov[col]
        new_sparse[mask, i] = sparse_oov[col]
    return new_sparse


def get_row_id_masks(data_ids, unique_ids):
    id_mask = np.isin(data_ids, unique_ids)
    id_mapping = dict(zip(unique_ids, range(len(unique_ids))))
    row_idxs = np.array([id_mapping[i] if i in id_mapping else -1 for i in data_ids])
    return row_idxs, id_mask


def update_new_sparse_feats(
    data,
    row_idxs,
    id_mask,
    unique_matrix,
    sparse_unique_vals,
    multi_sparse_unique_vals,
    col_info,
    col_mapping,
    sparse_offset,
):
    if unique_matrix is None:
        return
    for feat_idx, (col, col_index) in enumerate(zip(col_info.name, col_info.index)):
        # used in `data_info.assign_features()`, skip unknown features in new data.
        if col not in data.columns:
            continue
        if "multi_sparse" in col_mapping and col in col_mapping["multi_sparse"]:
            main_col = col_mapping["multi_sparse"][col]
            unique_vals = multi_sparse_unique_vals[main_col]
        elif multi_sparse_unique_vals and col in multi_sparse_unique_vals:
            unique_vals = multi_sparse_unique_vals[col]
        else:
            unique_vals = sparse_unique_vals[col]

        # used in `data_info.assign_features()`, new data may contain oov values.
        col_values = data[col].to_numpy()
        col_mask = id_mask & np.isin(col_values, unique_vals)
        indices, values = row_idxs[col_mask], col_values[col_mask]
        assert np.all(indices != -1)  # oov is marked as -1 in `id_mask`

        sparse_indices = column_sparse_indices(
            values, unique_vals, is_train=True, is_ordered=False
        )
        unique_matrix[indices, feat_idx] = sparse_offset[col_index] + sparse_indices
    return unique_matrix


def update_new_dense_feats(data, row_indices, id_mask, unique_matrix, col_info):
    if unique_matrix is None:
        return
    for feat_idx, col in enumerate(col_info.name):
        if col not in data.columns:
            continue
        index = row_indices[id_mask]
        dense_values = data[col].to_numpy(np.float32)
        unique_matrix[index, feat_idx] = dense_values[id_mask]
    return unique_matrix
