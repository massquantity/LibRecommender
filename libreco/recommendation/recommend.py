import numpy as np
import pandas as pd

from .ranking import rank_recommendations
from ..feature import (
    add_item_features,
    features_from_dict,
    get_recommend_indices_and_values,
)
from ..tfops import get_feed_dict


def popular_recommendations(data_info, inner_id, n_rec):
    if not inner_id:
        return np.array(data_info.popular_items[:n_rec])
    else:
        top_populars = data_info.popular_items[:n_rec]
        return np.array([data_info.item2id[i] for i in top_populars])


# def rank_recommendations(preds, model, user_id, n_rec, inner_id):
#    if model.task == "ranking":
#        preds = expit(preds)
#    consumed = set(model.user_consumed[user_id])
#    count = n_rec + len(consumed)
#    ids = np.argpartition(preds, -count)[-count:]
#    rank = sorted(zip(ids, preds[ids]), key=lambda x: -x[1])
#    recs_and_scores = islice(
#        (
#            rec if inner_id else (model.data_info.id2item[rec[0]], rec[1])
#            for rec in rank
#            if rec[0] not in consumed and rec[0] in model.data_info.id2item
#        ),
#        n_rec,
#    )
#    return list(recs_and_scores)


def recommend_from_embedding(
    task,
    data_info,
    user_embed,
    item_embed,
    user_id,
    n_rec,
    inner_id,
    filter_consumed,
    random_rec,
    return_scores,
):
    preds = user_embed[user_id] @ item_embed[:-1].T  # exclude item oov
    return rank_recommendations(
        task,
        preds,
        n_rec,
        data_info.n_items,
        data_info.user_consumed[user_id],
        data_info.id2item,
        inner_id,
        filter_consumed,
        random_rec,
        return_scores,
    )


def recommend_tf_feat(
    model,
    user_id,
    n_rec,
    user_feats,
    item_data,
    inner_id,
    filter_consumed,
    random_rec,
    return_scores,
):
    (
        user_indices,
        item_indices,
        sparse_indices,
        dense_values,
    ) = get_recommend_indices_and_values(
        model.data_info, user_id, model.n_items, model.sparse, model.dense
    )
    if user_feats is not None:
        assert isinstance(
            user_feats, (dict, pd.Series)
        ), "feats must be `dict` or `pandas.Series`."
        sparse_indices, dense_values = features_from_dict(
            model.data_info, sparse_indices, dense_values, user_feats, mode="recommend"
        )
    if item_data is not None:
        assert isinstance(
            item_data, pd.DataFrame
        ), "item_data must be `pandas DataFrame`"
        assert "item" in item_data.columns, "item_data must contain 'item' column"
        sparse_indices, dense_values = add_item_features(
            model.data_info, sparse_indices, dense_values, item_data
        )

    params = {
        "model": model,
        "user_indices": user_indices,
        "item_indices": item_indices,
        "sparse_indices": sparse_indices,
        "dense_values": dense_values,
        "is_training": False,
    }
    if model.model_category == "sequence":
        u_last_interacted = np.tile(
            model.user_last_interacted[user_id], (model.n_items, 1)
        )
        u_interacted_len = np.repeat(model.last_interacted_len[user_id], model.n_items)
        params["user_interacted_seq"] = u_last_interacted
        params["user_interacted_len"] = u_interacted_len

    preds = model.sess.run(model.output, get_feed_dict(**params))
    return rank_recommendations(
        model.task,
        preds,
        n_rec,
        model.n_items,
        model.user_consumed[user_id],
        model.data_info.id2item,
        inner_id,
        filter_consumed,
        random_rec,
        return_scores,
    )
