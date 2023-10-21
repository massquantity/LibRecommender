import numpy as np

from ..prediction.preprocess import get_cached_dual_seq, get_cached_seqs, set_temp_feats
from ..tfops.features import get_dual_seq_feed_dict, get_feed_dict


def process_sparse_embed_seq(model, user_id, seq, inner_id):
    if user_id is None:
        seq_indices = model.recent_seq_indices
        seq_values = model.recent_seq_values
    elif seq is not None and len(seq) > 0:
        seq_indices, seq_values = build_sparse_seq(seq, model, inner_id)
    else:
        # user_id is array_like
        user_id = user_id.item()
        if user_id != model.n_users:
            seq = model.user_consumed[user_id]
            seq_indices, seq_values = build_sparse_seq(seq, model, inner_id=True)
        else:
            # -1 will be pruned in `tf.nn.safe_embedding_lookup_sparse`
            seq_indices = np.zeros((1, 2), dtype=np.int64)
            seq_values = np.array([-1], dtype=np.int64)
    return seq_indices, seq_values


def process_embed_seq(model, user_id, seq, inner_id):
    if user_id is None:
        seq = model.recent_seqs[:-1]
        seq_len = model.recent_seq_lens[:-1]
    elif seq is not None and len(seq) > 0:
        seq, seq_len = build_rec_seq(seq, model, inner_id)
    else:
        # embed cached seqs include oov
        seq, seq_len = get_cached_seqs(model, user_id, repeat=False)
    return seq, seq_len


def build_rec_seq(seq, model, inner_id, repeat=False):
    seq, seq_len = _extract_seq(seq, model, inner_id)
    recent_seq = np.full((1, model.max_seq_len), model.n_items, dtype=np.int32)
    recent_seq[0, :seq_len] = seq[-seq_len:]
    seq_len = np.array([seq_len], dtype=np.int32)
    if repeat:
        recent_seq = np.repeat(recent_seq, model.n_items, axis=0)
        seq_len = np.repeat(seq_len, model.n_items)
    return recent_seq, seq_len


def build_dual_seq(seq, model, inner_id, repeat=False):
    assert isinstance(seq, (list, np.ndarray)), "`seq` must be list or numpy.ndarray."
    if not inner_id:
        seq = [model.data_info.item2id.get(i, model.n_items) for i in seq]

    total_max_len = model.long_max_len + model.short_max_len
    long_seq = np.full((1, model.long_max_len), model.n_items, dtype=np.int32)
    if len(seq) >= total_max_len:
        long_len = model.long_max_len
        start_index = len(seq) - total_max_len
        long_seq[0] = seq[start_index : start_index + long_len]
    elif len(seq) > model.short_max_len:
        long_len = len(seq) - model.short_max_len
        long_seq[0, :long_len] = seq[:long_len]
    else:
        long_len = 1
    long_len = np.array([long_len], dtype=np.int32)

    short_seq = np.full((1, model.short_max_len), model.n_items, dtype=np.int32)
    short_len = min(model.short_max_len, len(seq))
    short_seq[0, :short_len] = seq[-short_len:]
    short_len = np.array([short_len], dtype=np.int32)
    if repeat:
        long_seq = np.repeat(long_seq, model.n_items, axis=0)
        long_len = np.repeat(long_len, model.n_items)
        short_seq = np.repeat(short_seq, model.n_items, axis=0)
        short_len = np.repeat(short_len, model.n_items)
    return long_seq, long_len, short_seq, short_len


def build_sparse_seq(seq, model, inner_id):
    seq, seq_len = _extract_seq(seq, model, inner_id)
    sparse_tensor_indices = np.zeros((seq_len, 2), dtype=np.int64)
    # -1 will be pruned in `tf.nn.safe_embedding_lookup_sparse`
    seq = [i if i < model.n_items else -1 for i in seq[-seq_len:]]
    sparse_tensor_values = np.array(seq, dtype=np.int64)
    return sparse_tensor_indices, sparse_tensor_values


def process_embed_feat(data_info, user_id, user_feats):
    sparse_indices = dense_values = None
    if user_id is not None:
        if data_info.user_sparse_unique is not None:
            sparse_indices = data_info.user_sparse_unique[user_id]
            if user_feats is not None:
                # do not modify the original features
                sparse_indices = sparse_indices.copy()
                _set_user_sparse_indices(data_info, sparse_indices, user_feats)
        if data_info.user_dense_unique is not None:
            dense_values = data_info.user_dense_unique[user_id]
            if user_feats is not None:
                dense_values = dense_values.copy()
                _set_user_dense_values(data_info, dense_values, user_feats)
    else:
        if data_info.user_sparse_unique is not None:
            sparse_indices = data_info.user_sparse_unique[:-1]
        if data_info.user_dense_unique is not None:
            dense_values = data_info.user_dense_unique[:-1]
    return sparse_indices, dense_values


def process_tf_feat(model, user_ids, user_feats, seq, inner_id):
    user_indices, item_indices, sparse_indices, dense_values = [], [], [], []
    has_sparse = model.sparse if hasattr(model, "sparse") else None
    has_dense = model.dense if hasattr(model, "dense") else None
    all_items = np.arange(model.n_items)
    for u in user_ids:
        user_indices.append(np.repeat(u, model.n_items))
        item_indices.append(all_items)
        sparse, dense = _get_original_feats(
            model.data_info, u, model.n_items, has_sparse, has_dense
        )
        if sparse is not None:
            sparse_indices.append(sparse)
        if dense is not None:
            dense_values.append(dense)
    user_indices = np.concatenate(user_indices, axis=0)
    item_indices = np.concatenate(item_indices, axis=0)
    sparse_indices = np.concatenate(sparse_indices, axis=0) if sparse_indices else None
    dense_values = np.concatenate(dense_values, axis=0) if dense_values else None

    if user_feats is not None:
        assert isinstance(user_feats, dict), "`user_feats` must be `dict`."
        sparse_indices, dense_values = set_temp_feats(
            model.data_info, sparse_indices, dense_values, user_feats
        )

    if model.model_name == "SIM":
        if seq is not None and len(seq) > 0:
            long_seq, long_len, short_seq, short_len = build_dual_seq(
                seq, model, inner_id, repeat=True
            )
        else:
            long_seq, long_len, short_seq, short_len = get_cached_dual_seq(
                model, user_ids, repeat=True
            )
        return get_dual_seq_feed_dict(
            model,
            user_indices,
            item_indices,
            sparse_indices,
            dense_values,
            long_seq,
            long_len,
            short_seq,
            short_len,
            is_training=False,
        )
    else:
        if seq is not None and len(seq) > 0:
            seqs, seq_len = build_rec_seq(seq, model, inner_id, repeat=True)
        else:
            # tf cached seqs include oov
            seqs, seq_len = get_cached_seqs(model, user_ids, repeat=True)
        return get_feed_dict(
            model=model,
            user_indices=user_indices,
            item_indices=item_indices,
            sparse_indices=sparse_indices,
            dense_values=dense_values,
            user_interacted_seq=seqs,
            user_interacted_len=seq_len,
            is_training=False,
        )


def _get_original_feats(data_info, user, n_items, sparse, dense):
    sparse_indices = (
        _extract_feats(
            user,
            n_items,
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
            n_items,
            data_info.user_dense_col.index,
            data_info.item_dense_col.index,
            data_info.user_dense_unique,
            data_info.item_dense_unique,
        )
        if dense
        else None
    )
    return sparse_indices, dense_values


def _extract_feats(user, n_items, user_col, item_col, user_unique, item_unique):
    user_feats = np.tile(user_unique[user], (n_items, 1)) if user_col else None
    item_feats = item_unique[:-1] if item_col else None  # remove oov
    if user_col and item_col:
        orig_cols = user_col + item_col
        # keep column names in original order
        col_reindex = np.arange(len(orig_cols))[np.argsort(orig_cols)]
        features = np.concatenate([user_feats, item_feats], axis=1)
        return features[:, col_reindex]
    return user_feats if user_col else item_feats


def _extract_seq(seq, model, inner_id):
    assert isinstance(seq, (list, np.ndarray)), "`seq` must be list or numpy.ndarray."
    if not inner_id:
        seq = [model.data_info.item2id.get(i, model.n_items) for i in seq]
    seq_len = min(model.max_seq_len, len(seq))
    return seq, seq_len


def _set_user_sparse_indices(data_info, user_sparse_indices, feat_dict):
    """Set user sparse features according to the provided `feat_dict`.

    If the provided feature is not a user sparse feature or the value in a provided feature
    doesn't exist in training data(through `idx_mapping`), this feature will be ignored.
    """
    col_mapping = data_info.col_name_mapping
    sparse_idx_mapping = data_info.sparse_idx_mapping
    offsets = data_info.sparse_offset
    user_sparse_cols = data_info.user_sparse_col.name
    for col, val in feat_dict.items():
        if col not in user_sparse_cols:
            continue
        if "multi_sparse" in col_mapping and col in col_mapping["multi_sparse"]:
            main_col = col_mapping["multi_sparse"][col]
            idx_mapping = sparse_idx_mapping[main_col]
        else:
            idx_mapping = sparse_idx_mapping[col]

        if val in idx_mapping:
            user_field_idx = user_sparse_cols.index(col)
            feat_idx = idx_mapping[val]
            all_field_idx = col_mapping["sparse_col"][col]
            # shape: [1, d]
            user_sparse_indices[0, user_field_idx] = feat_idx + offsets[all_field_idx]


def _set_user_dense_values(data_info, user_dense_values, feat_dict):
    user_dense_cols = data_info.user_dense_col.name
    for col, val in feat_dict.items():
        if col not in user_dense_cols:
            continue
        user_field_idx = user_dense_cols.index(col)
        user_dense_values[0, user_field_idx] = val
