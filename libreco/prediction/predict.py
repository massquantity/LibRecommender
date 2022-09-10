import numpy as np
import pandas as pd
from tqdm import tqdm

from ..feature import (
    features_from_batch_data,
    features_from_dict,
    get_predict_indices_and_values,
)
from ..tfops import get_feed_dict
from ..utils.validate import convert_id, check_unknown


def normalize_prediction(preds, model, cold_start, unknown_num, unknown_index):
    if model.task == "rating":
        preds = np.clip(preds, model.lower_bound, model.upper_bound)
    elif model.task == "ranking":
        preds = 1 / (1 + np.exp(-preds))

    if unknown_num > 0 and cold_start == "popular":
        if isinstance(preds, np.ndarray):
            preds[unknown_index] = model.default_prediction
        elif isinstance(preds, list):
            for i in unknown_index:
                preds[i] = model.default_prediction
        else:
            preds = model.default_prediction
    return preds


def predict_from_embedding(model, user, item, cold_start, inner_id):
    user, item = convert_id(model, user, item, inner_id)
    unknown_num, unknown_index, user, item = check_unknown(model, user, item)
    preds = np.sum(np.multiply(model.user_embed[user], model.item_embed[item]), axis=1)
    return normalize_prediction(preds, model, cold_start, unknown_num, unknown_index)


def predict_tf_feat(model, user, item, feats, cold_start, inner_id):
    user, item = convert_id(model, user, item, inner_id)
    unknown_num, unknown_index, user, item = check_unknown(model, user, item)
    (
        user_indices,
        item_indices,
        sparse_indices,
        dense_values,
    ) = get_predict_indices_and_values(
        model.data_info, user, item, model.n_items, model.sparse, model.dense
    )

    if feats is not None:
        assert isinstance(
            feats, (dict, pd.Series)
        ), "feats must be `dict` or `pandas.Series`."
        assert len(user_indices) == 1, "only support single user for feats"
        sparse_indices, dense_values = features_from_dict(
            model.data_info, sparse_indices, dense_values, feats, "predict"
        )

    if model.model_category == "sequence":
        feed_dict = get_feed_dict(
            model=model,
            user_indices=user_indices,
            item_indices=item_indices,
            sparse_indices=sparse_indices,
            dense_values=dense_values,
            user_interacted_seq=model.user_last_interacted[user_indices],
            user_interacted_len=model.last_interacted_len[user_indices],
            is_training=False,
        )
    else:
        feed_dict = get_feed_dict(
            model=model,
            user_indices=user_indices,
            item_indices=item_indices,
            sparse_indices=sparse_indices,
            dense_values=dense_values,
            is_training=False,
        )
    preds = model.sess.run(model.output, feed_dict)
    return normalize_prediction(preds, model, cold_start, unknown_num, unknown_index)


def predict_data_with_feats(
    model, data, batch_size=None, cold_start="average", inner_id=False
):
    assert isinstance(data, pd.DataFrame), "data must be pandas DataFrame"
    user, item = convert_id(model, data.user, data.item, inner_id)
    unknown_num, unknown_index, user, item = check_unknown(model, user, item)
    if not batch_size:
        batch_size = len(data)
    preds = np.zeros(len(data), dtype=np.float32)
    for index in tqdm(range(0, len(data), batch_size), "pred_data"):
        batch_slice = slice(index, index + batch_size)
        batch_data = data.iloc[batch_slice]
        user_indices = user[batch_slice]
        item_indices = item[batch_slice]
        sparse_indices, dense_values = features_from_batch_data(
            model.data_info, model.sparse, model.dense, batch_data
        )
        if model.model_category == "sequence":
            feed_dict = get_feed_dict(
                model=model,
                user_indices=user_indices,
                item_indices=item_indices,
                sparse_indices=sparse_indices,
                dense_values=dense_values,
                user_interacted_seq=model.user_last_interacted[user_indices],
                user_interacted_len=model.last_interacted_len[user_indices],
                is_training=False,
            )
        else:
            feed_dict = get_feed_dict(
                model=model,
                user_indices=user_indices,
                item_indices=item_indices,
                sparse_indices=sparse_indices,
                dense_values=dense_values,
                is_training=False,
            )
        preds[batch_slice] = model.sess.run(model.output, feed_dict)
    return normalize_prediction(preds, model, cold_start, unknown_num, unknown_index)
