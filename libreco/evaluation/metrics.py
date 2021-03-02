import numpy as np

# pairwise_metrics = bpr ??
POINTWISE_METRICS = {"loss", "log_loss", "balanced_accuracy",
                     "roc_auc", "pr_auc"}
LISTWISE_METRICS = {"precision", "recall", "map", "ndcg"}
ALLOWED_METRICS = {
    "rating_metrics": ["loss", "rmse", "mae", "r2"],
    "ranking_metrics": ["loss", "log_loss", "balanced_accuracy",
                        "roc_auc", "pr_auc", "precision", "recall",
                        "map", "ndcg"]
}


def precision_at_k(y_true_list, y_reco_list, users, k):
    precision_all = list()
    for u in users:
        y_true = y_true_list[u]
        y_reco = y_reco_list[u]
        common_items = set(y_reco).intersection(y_true)
        precision = len(common_items) / k
        precision_all.append(precision)
    return np.mean(precision_all)


def recall_at_k(y_true_list, y_reco_list, users, k):
    recall_all = list()
    for u in users:
        y_true = y_true_list[u]
        y_reco = y_reco_list[u]
        common_items = set(y_reco).intersection(y_true)
        recall = len(common_items) / len(y_true)
        recall_all.append(recall)
    return np.mean(recall_all)


def average_precision_at_k(y_true, y_reco, k):
    common_items, indices_in_true, indices_in_reco = np.intersect1d(
        y_true, y_reco, assume_unique=True, return_indices=True)

    rank_list = np.zeros(k)
    if common_items.size > 0:
        rank_list[indices_in_reco] = 1
        ap = [np.mean(rank_list[:i+1]) for i in range(k) if rank_list[i]]
        assert len(ap) == common_items.size, "common size doesn't match..."
        return np.mean(ap)
    else:
        return 0


def map_at_k(y_true_list, y_reco_list, users, k):
    map_all = list()
    for u in users:
        y_true = list(set(y_true_list[u]))
        y_reco = y_reco_list[u]
        map_all.append(average_precision_at_k(y_true, y_reco, k))
    return np.mean(map_all)


def ndcg_at_k(y_true_list, y_reco_list, users, k):
    ndcg_all = list()
    for u in users:
        rank_list = np.zeros(k)
        y_true = list(set(y_true_list[u]))
        y_reco = y_reco_list[u]
        common_items, indices_in_true, indices_in_reco = np.intersect1d(
            y_true, y_reco, assume_unique=True, return_indices=True)

        if common_items.size > 0:
            rank_list[indices_in_reco] = 1
            ideal_list = np.sort(rank_list)[::-1]
            dcg = np.sum(rank_list / np.log2(np.arange(2, k + 2)))
            idcg = np.sum(ideal_list / np.log2(np.arange(2, k + 2)))
            ndcg = dcg / idcg
        else:
            ndcg = 0
        ndcg_all.append(ndcg)
    return np.mean(ndcg_all)

