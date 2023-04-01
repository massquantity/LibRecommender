import numpy as np

from ..prediction.preprocess import get_cached_seqs, set_temp_feats
from ..tfops import get_feed_dict


def embed_from_seq(model, user_ids, seq, inner_id):
    seq, seq_len = build_rec_seq(seq, model, inner_id)
    feed_dict = {
        model.user_interacted_seq: seq,
        model.user_interacted_len: seq_len,
    }
    if hasattr(model, "user_indices"):
        feed_dict[model.user_indices] = np.array(user_ids, dtype=np.int32)
    embed = model.sess.run(model.user_vector, feed_dict)
    # embed = embed / np.linalg.norm(embed, axis=1, keepdims=True)
    bias = np.ones([1, 1], dtype=embed.dtype)
    return np.hstack([embed, bias])


def build_rec_seq(seq, model, inner_id, repeat=False):
    assert isinstance(seq, (list, np.ndarray)), "`seq` must be list or numpy.ndarray."
    if not inner_id:
        seq = [model.data_info.item2id.get(i, model.n_items) for i in seq]
    recent_seq = np.full((1, model.max_seq_len), model.n_items, dtype=np.int32)
    seq_len = min(model.max_seq_len, len(seq))
    recent_seq[0, -seq_len:] = seq[-seq_len:]
    seq_len = np.array([seq_len], dtype=np.float32)
    if repeat:
        recent_seq = np.repeat(recent_seq, model.n_items, axis=0)
        seq_len = np.repeat(seq_len, model.n_items)
    return recent_seq, seq_len


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
    if seq is not None and len(seq) > 0:
        seqs, seq_len = build_rec_seq(seq, model, inner_id, repeat=True)
    else:
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
