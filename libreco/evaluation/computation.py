import numpy as np
import pandas as pd
from tqdm import tqdm

from ..data import TransformedSet
from ..prediction.preprocess import convert_id
from ..utils.validate import check_labels


def build_eval_transformed_data(model, data, neg_sampling, seed):
    if isinstance(data, pd.DataFrame):
        assert "user" in data and "item" in data and "label" in data
        users = data["user"].tolist()
        items = data["item"].tolist()
        user_indices, item_indices = convert_id(model, users, items, inner_id=False)
        labels = data["label"].to_numpy(dtype=np.float32)
        data = TransformedSet(user_indices, item_indices, labels, train=False)
    if neg_sampling and not data.has_sampled:
        num_neg = model.num_neg or 1 if hasattr(model, "num_neg") else 1
        data.build_negative_samples(model.data_info, num_neg, seed=seed)
    else:
        check_labels(model, data.labels, neg_sampling)
    return data


def compute_preds(model, data, batch_size):
    y_pred = list()
    y_label = list()
    for i in tqdm(range(0, len(data), batch_size), desc="eval_pointwise"):
        user_indices, item_indices, labels = data[i: i + batch_size]
        preds = model.predict(user_indices, item_indices, inner_id=True)
        y_pred.extend(preds)
        y_label.extend(labels)
    return y_pred, y_label


def compute_probs(model, data, batch_size):
    return compute_preds(model, data, batch_size)


def compute_recommends(model, users, k, num_batch_users):
    y_recommends = dict()
    for i in tqdm(range(0, len(users), num_batch_users), desc="eval_listwise"):
        batch_users = users[i: i + num_batch_users]
        batch_recs = model.recommend_user(
            user=batch_users,
            n_rec=k,
            inner_id=True,
            filter_consumed=True,
            random_rec=False,
        )
        y_recommends.update(batch_recs)
    return y_recommends
