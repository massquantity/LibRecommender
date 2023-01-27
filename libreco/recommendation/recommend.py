import numpy as np
import pandas as pd

from ..feature import (
    add_item_features,
    features_from_dict,
    get_recommend_indices_and_values,
)
from ..tfops import get_feed_dict
from .ranking import rank_recommendations


def construct_rec(data_info, user_ids, computed_recs, inner_id):
    result_recs = dict()
    for i, u in enumerate(user_ids):
        if inner_id:
            result_recs[u] = computed_recs[i]
        else:
            u = data_info.id2user[u]
            result_recs[u] = np.array(
                [data_info.id2item[ri] for ri in computed_recs[i]]
            )
    return result_recs


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
    user_ids,
    n_rec,
    data_info,
    user_embed,
    item_embed,
    filter_consumed,
    random_rec,
):
    preds = user_embed[user_ids] @ item_embed[:data_info.n_items].T  # exclude item oov
    return rank_recommendations(
        task,
        user_ids,
        preds,
        n_rec,
        data_info.n_items,
        data_info.user_consumed,
        filter_consumed,
        random_rec,
    )


def recommend_tf_feat(
    model,
    user_ids,
    n_rec,
    user_feats,
    item_data,
    filter_consumed,
    random_rec,
):
    user_indices, item_indices, sparse_indices, dense_values = [], [], [], []
    has_sparse = model.sparse if hasattr(model, "sparse") else None
    has_dense = model.dense if hasattr(model, "dense") else None
    for u in user_ids:
        u, i, s, d = get_recommend_indices_and_values(
            model.data_info, u, model.n_items, has_sparse, has_dense
        )
        user_indices.append(u)
        item_indices.append(i)
        if s is not None:
            sparse_indices.append(s)
        if d is not None:
            dense_values.append(d)
    user_indices = np.concatenate(user_indices, axis=0)
    item_indices = np.concatenate(item_indices, axis=0)
    sparse_indices = np.concatenate(sparse_indices, axis=0) if sparse_indices else None
    dense_values = np.concatenate(dense_values, axis=0) if dense_values else None

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
        u_last_interacted = np.repeat(
            model.user_last_interacted[user_ids], model.n_items, axis=0
        )
        u_interacted_len = np.repeat(model.last_interacted_len[user_ids], model.n_items)
        params["user_interacted_seq"] = u_last_interacted
        params["user_interacted_len"] = u_interacted_len

    preds = model.sess.run(model.output, get_feed_dict(**params))
    return rank_recommendations(
        model.task,
        user_ids,
        preds,
        n_rec,
        model.n_items,
        model.user_consumed,
        filter_consumed,
        random_rec,
    )
