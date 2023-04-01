import numpy as np


def convert_id(model, user, item, inner_id=False):
    user = [user] if np.isscalar(user) else user
    item = [item] if np.isscalar(item) else item
    if not inner_id:
        user = [model.data_info.user2id.get(u, model.n_users) for u in user]
        item = [model.data_info.item2id.get(i, model.n_items) for i in item]
    return np.array(user), np.array(item)


def get_original_feats(data_info, user, item, sparse, dense):
    """Get original features from data_info to predict using feat models."""
    user = [user] if np.isscalar(user) else user
    item = [item] if np.isscalar(item) else item
    sparse_indices = (
        _extract_feats(
            user,
            item,
            data_info.user_sparse_col.index,
            data_info.item_sparse_col.index,
            data_info.user_sparse_unique,
            data_info.item_sparse_unique,
        )
        if sparse
        else None
    )
    dense_values = (
        _extract_feats(
            user,
            item,
            data_info.user_dense_col.index,
            data_info.item_dense_col.index,
            data_info.user_dense_unique,
            data_info.item_dense_unique,
        )
        if dense
        else None
    )
    return user, item, sparse_indices, dense_values


def _extract_feats(user, item, user_col, item_col, user_unique, item_unique):
    user_feats = user_unique[user] if user_col else None
    item_feats = item_unique[item] if item_col else None
    if user_col and item_col:
        orig_cols = user_col + item_col
        # keep column names in original order
        col_reindex = np.arange(len(orig_cols))[np.argsort(orig_cols)]
        features = np.concatenate([user_feats, item_feats], axis=1)
        return features[:, col_reindex]
    return user_feats if user_col else item_feats


def set_temp_feats(data_info, sparse_indices, dense_values, feat_dict):
    """Set temporary features in data.

    This function is used in predicting or recommending with user provided features,
    and it will try not to modify the original numpy arrays.
    """
    sparse_indices_copy = None if sparse_indices is None else sparse_indices.copy()
    dense_values_copy = None if dense_values is None else dense_values.copy()
    _set_sparse_indices(
        sparse_indices_copy,
        data_info.col_name_mapping,
        data_info.sparse_idx_mapping,
        data_info.sparse_offset,
        feat_dict,
    )
    _set_dense_values(dense_values_copy, data_info.col_name_mapping, feat_dict)
    return sparse_indices_copy, dense_values_copy


def _set_sparse_indices(
    sparse_indices, col_mapping, sparse_idx_mapping, offsets, feat_dict
):
    if "sparse_col" not in col_mapping:
        return
    field_mapping = col_mapping["sparse_col"]
    for col, val in feat_dict.items():
        if col not in field_mapping:
            continue
        if "multi_sparse" in col_mapping and col in col_mapping["multi_sparse"]:
            main_col = col_mapping["multi_sparse"][col]
            idx_mapping = sparse_idx_mapping[main_col]
        else:
            idx_mapping = sparse_idx_mapping[col]
        if val in idx_mapping:
            field_idx = field_mapping[col]
            feat_idx = idx_mapping[val]
            offset = offsets[field_idx]
            sparse_indices[:, field_idx] = feat_idx + offset


def _set_dense_values(dense_values, col_mapping, feat_dict):
    if "dense_col" not in col_mapping:
        return
    field_mapping = col_mapping["dense_col"]
    for col, val in feat_dict.items():
        if col not in field_mapping:
            continue
        field_idx = field_mapping[col]
        dense_values[:, field_idx] = val


def get_cached_seqs(model, user_id, repeat):
    if model.model_category != "sequence":
        return None, None
    seqs = model.recent_seqs[user_id]
    seq_len = model.recent_seq_lens[user_id]
    if repeat:
        seqs = np.repeat(seqs, model.n_items, axis=0)
        seq_len = np.repeat(seq_len, model.n_items)
    return seqs, seq_len


def features_from_batch(data_info, sparse, dense, data):
    sparse_indices, dense_values = None, None
    if sparse:
        sparse_col_mapping = data_info.col_name_mapping["sparse_col"]
        sparse_indices = np.zeros((len(data), len(sparse_col_mapping)), np.int32)
        for col, field_idx in sparse_col_mapping.items():
            if col not in data.columns:
                raise ValueError(f"Column `{col}` doesn't exist in data")
            sparse_indices[:, field_idx] = _compute_sparse_feat_indices(
                data_info, data, field_idx, col
            )
    if dense:
        dense_cols = list(data_info.col_name_mapping["dense_col"])
        for col in dense_cols:
            if col not in data.columns:
                raise ValueError(f"Column `{col}` doesn't exist in data")
        dense_values = data[dense_cols].to_numpy(dtype=np.float32)
    return sparse_indices, dense_values


def _compute_sparse_feat_indices(data_info, data, field_idx, col):
    offset = data_info.sparse_offset[field_idx]
    oov_val = data_info.sparse_oov[field_idx]
    values = data[col].tolist()
    if (
        "multi_sparse" in data_info.col_name_mapping
        and col in data_info.col_name_mapping["multi_sparse"]
    ):
        main_col = data_info.col_name_mapping["multi_sparse"][col]
        idx_mapping = data_info.sparse_idx_mapping[main_col]
    else:
        idx_mapping = data_info.sparse_idx_mapping[col]
    return np.array(
        [idx_mapping[v] + offset if v in idx_mapping else oov_val for v in values]
    )
