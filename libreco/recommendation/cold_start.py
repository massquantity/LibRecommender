import numpy as np


def popular_recommendations(data_info, inner_id, n_rec):
    popular_recs = data_info.np_rng.choice(data_info.popular_items, n_rec)
    if inner_id:
        return np.array([data_info.item2id[i] for i in popular_recs])
    else:
        return popular_recs


def average_recommendations(data_info, default_recs, inner_id, n_rec):
    average_recs = data_info.np_rng.choice(default_recs, n_rec)
    if inner_id:
        return average_recs
    else:
        return np.array([data_info.id2item[i] for i in average_recs])


def cold_start_rec(data_info, default_recs, cold_start, users, n_rec, inner_id):
    if cold_start not in ("average", "popular"):
        raise ValueError(f"Unknown cold start strategy: {cold_start}")
    result_recs = dict()
    for u in users:
        if cold_start == "average":
            result_recs[u] = average_recommendations(
                data_info, default_recs, inner_id, n_rec
            )
        elif cold_start == "popular":
            result_recs[u] = popular_recommendations(data_info, inner_id, n_rec)
    return result_recs
