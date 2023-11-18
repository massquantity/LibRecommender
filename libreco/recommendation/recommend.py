import numpy as np

from .preprocess import process_tf_feat
from .ranking import rank_recommendations
from ..utils.constants import SequenceModels


def construct_rec(data_info, user_ids, computed_recs, inner_id):
    result_recs = dict()
    for i, u in enumerate(user_ids):
        if inner_id:
            result_recs[u] = np.array(computed_recs[i])
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


def check_dynamic_rec_feats(model_name, user, user_feats, seq):
    if seq is not None and not SequenceModels.contains(model_name):
        raise ValueError(f"`{model_name}` doesn't support arbitrary seq inference.")
    if not np.isscalar(user):
        if user_feats is not None:
            raise ValueError(
                f"Batch inference doesn't support assigning arbitrary features: {user}"
            )
        if seq is not None:
            raise ValueError(
                f"Batch inference doesn't support arbitrary item sequence: {user}"
            )
    if seq is not None and not isinstance(seq, (list, np.ndarray)):
        raise ValueError("`seq` must be list or numpy.ndarray.")
    if user_feats is not None and not isinstance(user_feats, dict):
        raise ValueError("`user_feats` must be `dict`.")


def recommend_from_embedding(
    model,
    user_ids,  # shape should be [1, d] to preserve output shape
    n_rec,
    user_embeddings,
    item_embeddings,
    filter_consumed,
    random_rec,
):
    user_embed = user_embeddings[user_ids]
    item_embeds = item_embeddings[: model.n_items]  # exclude item oov
    preds = user_embed @ item_embeds.T
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


def recommend_tf_feat(
    model,
    user_ids,
    n_rec,
    user_feats,
    seq,
    filter_consumed,
    random_rec,
    inner_id=False,
):
    feed_dict = process_tf_feat(model, user_ids, user_feats, seq, inner_id)
    if model.model_name == "SIM":
        preds = model.sess.run(model.inference_output, feed_dict)
    else:
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
