import numpy as np
from numpy.random import default_rng
from scipy.special import expit, softmax

# Numpy doc states that it is recommended to use new random API
# https://numpy.org/doc/stable/reference/random/index.html
np_rng = default_rng()


def rank_recommendations(
    task,
    user_ids,
    model_preds,
    n_rec,
    n_items,
    user_consumed,
    filter_consumed=True,
    random_rec=False,
    return_scores=False,
):
    if n_rec > n_items:
        raise ValueError(f"`n_rec` {n_rec} exceeds num of items {n_items}")
    if model_preds.ndim == 1:
        assert len(model_preds) % n_items == 0
        batch_size = int(len(model_preds) / n_items)
        all_preds = model_preds.reshape(batch_size, n_items)
    else:
        batch_size = len(model_preds)
        all_preds = model_preds
    all_ids = np.tile(np.arange(n_items), (batch_size, 1))

    batch_ids, batch_preds = [], []
    for i in range(batch_size):
        user = user_ids[i]
        ids = all_ids[i]
        preds = all_preds[i]
        consumed = user_consumed[user] if user in user_consumed else []
        if filter_consumed and consumed and n_rec + len(consumed) <= n_items:
            ids, preds = filter_items(ids, preds, consumed)
        if random_rec:
            ids, preds = random_select(ids, preds, n_rec)
        else:
            ids, preds = partition_select(ids, preds, n_rec)
        batch_ids.append(ids)
        batch_preds.append(preds)

    ids, preds = np.array(batch_ids), np.array(batch_preds)
    indices = np.argsort(preds, axis=1)[:, ::-1]
    ids = np.take_along_axis(ids, indices, axis=1)
    if return_scores:
        scores = np.take_along_axis(preds, indices, axis=1)
        if task == "ranking":
            scores = expit(scores)
        return ids, scores
    else:
        return ids


def filter_items(ids, preds, items):
    mask = np.isin(ids, items, assume_unique=True, invert=True)
    return ids[mask], preds[mask]


# add `**0.75` to lower probability of high score items
def get_reco_probs(preds):
    p = np.power(softmax(preds), 0.75) + 1e-8  # avoid zero probs
    return p / p.sum()


def random_select(ids, preds, n_rec):
    p = get_reco_probs(preds)
    mask = np_rng.choice(len(preds), n_rec, p=p, replace=False, shuffle=False)
    return ids[mask], preds[mask]


def partition_select(ids, preds, n_rec):
    mask = np.argpartition(preds, -n_rec)[-n_rec:]
    return ids[mask], preds[mask]
