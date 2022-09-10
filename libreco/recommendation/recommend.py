from itertools import islice

import numpy as np
import pandas as pd

from ..feature import (
    add_item_features,
    features_from_dict,
    get_recommend_indices_and_values,
)
from ..tfops import get_feed_dict
from ..utils.validate import check_unknown_user


def popular_recommendations(data_info, inner_id, n_rec):
    if not inner_id:
        return data_info.popular_items[:n_rec]
    else:
        top_populars = data_info.popular_items[:n_rec]
        return [data_info.item2id[i] for i in top_populars]


def rank_recommendations(recos, model, user_id, n_rec, inner_id):
    if model.task == "ranking":
        recos = 1 / (1 + np.exp(-recos))
    consumed = set(model.user_consumed[user_id])
    count = n_rec + len(consumed)
    ids = np.argpartition(recos, -count)[-count:]
    rank = sorted(zip(ids, recos[ids]), key=lambda x: -x[1])
    recs_and_scores = islice(
        (
            rec if inner_id else (model.data_info.id2item[rec[0]], rec[1])
            for rec in rank
            if rec[0] not in consumed
        ),
        n_rec,
    )
    return list(recs_and_scores)


def recommend_from_embedding(model, user, n_rec, cold_start, inner_id):
    user_id = check_unknown_user(model, user, inner_id)
    if user_id is None:
        if cold_start == "average":
            user_id = model.n_users
        elif cold_start == "popular":
            return popular_recommendations(model.data_info, inner_id, n_rec)
        else:
            raise ValueError(user)

    recos = model.user_embed[user_id] @ model.item_embed.T
    return rank_recommendations(recos, model, user_id, n_rec, inner_id)


def recommend_tf_feat(model, user, n_rec, user_feats, item_data, cold_start, inner_id):
    user_id = check_unknown_user(model, user, inner_id)
    if user_id is None:
        if cold_start == "average":
            user_id = model.n_users
        elif cold_start == "popular":
            return popular_recommendations(model.data_info, inner_id, n_rec)
        else:
            raise ValueError(user)

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
            model.data_info, sparse_indices, dense_values, user_feats, "recommend"
        )
    if item_data is not None:
        assert isinstance(
            item_data, pd.DataFrame
        ), "item_data must be `pandas DataFrame`"
        assert "item" in item_data.columns, "item_data must contain 'item' column"
        sparse_indices, dense_values = add_item_features(
            model.data_info, sparse_indices, dense_values, item_data
        )

    if model.model_category == "sequence":
        u_last_interacted = np.tile(
            model.user_last_interacted[user_id], (model.n_items, 1)
        )
        u_interacted_len = np.repeat(model.last_interacted_len[user_id], model.n_items)
        feed_dict = get_feed_dict(
            model=model,
            user_indices=user_indices,
            item_indices=item_indices,
            sparse_indices=sparse_indices,
            dense_values=dense_values,
            user_interacted_seq=u_last_interacted,
            user_interacted_len=u_interacted_len,
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

    recos = model.sess.run(model.output, feed_dict)
    return rank_recommendations(recos, model, user_id, n_rec, inner_id)
