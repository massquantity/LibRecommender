import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    balanced_accuracy_score,
    mean_squared_error,
    precision_recall_curve,
    roc_auc_score,
)

RATING_METRICS = {"loss", "rmse", "mae", "r2"}
POINTWISE_METRICS = {
    "loss",
    "log_loss",
    "balanced_accuracy",
    "roc_auc",
    "pr_auc",
    "roc_gauc",
}
LISTWISE_METRICS = {"precision", "recall", "map", "ndcg", "coverage"}
RANKING_METRICS = POINTWISE_METRICS | LISTWISE_METRICS


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html
def balanced_accuracy(y_true, y_prob):
    y_pred = np.round(y_prob)
    return balanced_accuracy_score(y_true, y_pred)


def roc_gauc_score(y_true, y_prob, user_indices):
    # gauc = 0
    # users = np.unique(user_indices)
    # y_true, y_prob = np.array(y_true), np.array(y_prob)
    # for u in users:
    #    index = np.where(user_indices == u)[0]
    #    user_auc = roc_auc_score(y_true[index], y_prob[index])
    #    gauc += len(index) * user_auc
    # return gauc / len(user_indices)

    def _safe_roc_auc(y_true, y_score):
        try:
            auc = roc_auc_score(y_true, y_score)
        except ValueError:  # only has one label
            auc = 0.0
        return auc

    roc_data = pd.DataFrame({"label": y_true, "prob": y_prob, "user": user_indices})
    gauc = (
        roc_data.groupby("user")
        .apply(lambda x: _safe_roc_auc(x["label"], x["prob"]) * len(x))
        .tolist()
    )
    return sum(gauc) / len(user_indices)


def pr_auc_score(y_true, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return auc(recall, precision)


def listwise_scores(fn, y_true_lists, y_reco_lists, users, k):
    scores = list()
    for u in users:
        y_true = y_true_lists[u]
        y_reco = y_reco_lists[u]
        scores.append(fn(y_true, y_reco, k))
    return np.mean(scores)


def precision_at_k(y_true, y_reco, k):
    common_items = set(y_reco).intersection(y_true)
    return len(common_items) / k


def recall_at_k(y_true, y_reco, _k):
    common_items = set(y_reco).intersection(y_true)
    return len(common_items) / len(y_true)


def average_precision_at_k(y_true, y_reco, k):
    common_items, _, indices_in_reco = np.intersect1d(
        y_true, y_reco, assume_unique=True, return_indices=True
    )
    if common_items.size == 0:
        return 0
    rank_list = np.zeros(k, np.float32)
    rank_list[indices_in_reco] = 1
    ap = [np.mean(rank_list[: i + 1]) for i in range(k) if rank_list[i]]
    assert len(ap) == common_items.size, "common size doesn't match..."
    return np.mean(ap)


def ndcg_at_k(y_true, y_reco, k):
    common_items, _, indices_in_reco = np.intersect1d(
        y_true, y_reco, assume_unique=True, return_indices=True
    )
    if common_items.size == 0:
        return 0
    rank_list = np.zeros(k, np.float32)
    rank_list[indices_in_reco] = 1
    ideal_list = np.sort(rank_list)[::-1]
    dcg = np.sum(rank_list / np.log2(np.arange(2, k + 2)))
    idcg = np.sum(ideal_list / np.log2(np.arange(2, k + 2)))
    return dcg / idcg


def rec_coverage(y_reco_lists, users, n_items):
    item_recs = set()
    for u in users:
        y_reco = y_reco_lists[u].tolist()
        item_recs.update(y_reco)
    return len(item_recs) / n_items * 100
