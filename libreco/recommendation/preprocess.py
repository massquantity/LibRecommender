import numpy as np

from ..prediction.preprocess import get_seq_feats, set_temp_feats
from ..tfops import get_feed_dict


def process_tf_feat(model, user_ids, user_feats):
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
    seqs, seq_len = get_seq_feats(model, user_ids, repeat=True)
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
