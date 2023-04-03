import numpy as np
import pandas as pd
from scipy.special import expit
from tqdm import tqdm

from .preprocess import (
    convert_id,
    features_from_batch,
    get_cached_seqs,
    get_original_feats,
    set_temp_feats,
)
from ..tfops import get_feed_dict
from ..utils.validate import check_unknown


def normalize_prediction(preds, model, cold_start, unknown_num, unknown_index):
    if model.task == "rating":
        preds = np.clip(preds, model.lower_bound, model.upper_bound)
    elif model.task == "ranking":
        # preds = 1 / (1 + np.exp(-z))
        preds = expit(preds)

    if unknown_num > 0 and cold_start == "popular":
        if isinstance(preds, np.ndarray):
            preds[unknown_index] = model.default_pred
        elif isinstance(preds, list):
            for i in unknown_index:
                preds[i] = model.default_pred
        else:
            preds = model.default_pred
    return preds


def predict_from_embedding(model, user, item, cold_start, inner_id):
    user, item = convert_id(model, user, item, inner_id)
    unknown_num, unknown_index, user, item = check_unknown(model, user, item)
    preds = np.sum(np.multiply(model.user_embed[user], model.item_embed[item]), axis=1)
    return normalize_prediction(preds, model, cold_start, unknown_num, unknown_index)


def predict_tf_feat(model, user, item, feats, cold_start, inner_id):
    user, item = convert_id(model, user, item, inner_id)
    unknown_num, unknown_index, user, item = check_unknown(model, user, item)
    has_sparse = model.sparse if hasattr(model, "sparse") else None
    has_dense = model.dense if hasattr(model, "dense") else None
    (
        user_indices,
        item_indices,
        sparse_indices,
        dense_values,
    ) = get_original_feats(model.data_info, user, item, has_sparse, has_dense)

    if feats is not None:
        assert isinstance(feats, dict), "`feats` must be `dict`."
        assert len(user_indices) == 1, "Predict with feats only supports single user."
        sparse_indices, dense_values = set_temp_feats(
            model.data_info, sparse_indices, dense_values, feats
        )

    seqs, seq_len = get_cached_seqs(model, user_indices, repeat=False)
    feed_dict = get_feed_dict(
        model=model,
        user_indices=user_indices,
        item_indices=item_indices,
        sparse_indices=sparse_indices,
        dense_values=dense_values,
        user_interacted_seq=seqs,
        user_interacted_len=seq_len,
        is_training=False,
    )
    preds = model.sess.run(model.output, feed_dict)
    return normalize_prediction(preds, model, cold_start, unknown_num, unknown_index)


def predict_data_with_feats(
    model, data, batch_size=None, cold_start="average", inner_id=False
):
    assert isinstance(data, pd.DataFrame), "Data must be pandas DataFrame"
    user, item = convert_id(model, data.user, data.item, inner_id)
    unknown_num, unknown_index, user, item = check_unknown(model, user, item)
    if not batch_size:
        batch_size = len(data)
    preds = np.zeros(len(data), dtype=np.float32)
    for i in tqdm(range(0, len(data), batch_size), "pred_data"):
        batch_slice = slice(i, i + batch_size)
        batch_data = data.iloc[batch_slice]
        user_indices = user[batch_slice]
        item_indices = item[batch_slice]
        sparse_indices, dense_values = features_from_batch(
            model.data_info, model.sparse, model.dense, batch_data
        )
        seqs, seq_len = get_cached_seqs(model, user_indices, repeat=False)
        feed_dict = get_feed_dict(
            model=model,
            user_indices=user_indices,
            item_indices=item_indices,
            sparse_indices=sparse_indices,
            dense_values=dense_values,
            user_interacted_seq=seqs,
            user_interacted_len=seq_len,
            is_training=False,
        )
        preds[batch_slice] = model.sess.run(model.output, feed_dict)
    return normalize_prediction(preds, model, cold_start, unknown_num, unknown_index)
