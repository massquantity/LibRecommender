import numpy as np

from .preprocess import process_tf_feat
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
    filter_consumed,
    random_rec,
):
    feed_dict = process_tf_feat(model, user_ids, user_feats)
    preds = model.sess.run(model.output, feed_dict)
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
