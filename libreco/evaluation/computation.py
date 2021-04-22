import numpy as np
from tqdm import tqdm
from ..data import TransformedSet
from ..feature import features_from_batch_data


def build_transformed_data(model, data, negative_sample, update_features, seed):
    data_info = model.data_info
    n_users = data_info.n_users
    n_items = data_info.n_items
    users = data.user.tolist()
    items = data.item.tolist()
    user_indices = np.array([data_info.user2id.get(u, n_users) for u in users])
    item_indices = np.array([data_info.item2id.get(i, n_items) for i in items])
    labels = data.label.to_numpy(dtype=np.float32)
    sparse_indices, dense_values = None, None
    if data_info.col_name_mapping is not None:
        sparse_indices, dense_values = features_from_batch_data(
            data_info, model.sparse, model.dense, data
        )
    # todo: merge user_consumed
    transformed_data = TransformedSet(
        user_indices, item_indices, labels, sparse_indices,
        dense_values, train=False
    )
    if update_features:
        # if a user or item has duplicate features, will only update the last one.
        user_data = data.drop_duplicates(subset=["user"], keep="last")
        item_data = data.drop_duplicates(subset=["item"], keep="last")
        model.data_info.assign_user_features(user_data)
        model.data_info.assign_item_features(item_data)
    if negative_sample:
        transformed_data.build_negative_samples(
            data_info, item_gen_mode="random", seed=seed
        )
    return transformed_data


# def compute_preds(model, data, batch_size, mode, n_users=None, n_items=None):
#    y_pred = list()
#    y_label = list()
#    for batch_data in tqdm(range(0, len(data), batch_size), desc="eval_pred"):
#        batch_slice = slice(batch_data, batch_data + batch_size)
#        users = data.user_indices[batch_slice]
#        items = data.item_indices[batch_slice]
#        labels = data.labels[batch_slice]
#        if mode == "eval":
#            user_allowed = np.where(
#                np.logical_and(users >= 0, users < n_users))[0]
#            item_allowed = np.where(
#                np.logical_and(items >= 0, items < n_items))[0]
#            indices = np.intersect1d(user_allowed, item_allowed)
#            users = users[indices]
#            items = items[indices]
#            labels = labels[indices]
#        preds = list(model.predict(users, items, inner_id=True))
#        y_pred.extend(preds)
#        y_label.extend(labels)
#    return y_pred, y_label


def compute_preds(model, data, batch_size):
    y_pred = list()
    y_label = list()
    predict_func = choose_pred_func(model)
    for batch_data in tqdm(range(0, len(data), batch_size), desc="eval_pred"):
        batch_slice = slice(batch_data, batch_data + batch_size)
        labels = data.labels[batch_slice]
        preds = predict_func(model, data, batch_slice)
        y_pred.extend(preds)
        y_label.extend(labels)
    return y_pred, y_label


# def compute_probs(model, data, batch_size, mode, n_users=None, n_items=None):
#    return compute_preds11(model, data, batch_size, mode, n_users, n_items)


def compute_probs(model, data, batch_size):
    return compute_preds(model, data, batch_size)


def compute_recommends(model, users, k):
    y_recommends = dict()
    no_rec_num = 0
    no_rec_users = []
    for u in tqdm(users, desc="eval_rec"):
        reco = model.recommend_user(u, k, inner_id=True)
        # user_cf popular
        if not reco or isinstance(reco[0], int):
            # print("no recommend user: ", u)
            no_rec_num += 1
            no_rec_users.append(u)
            continue
        reco = [r[0] for r in reco]
        y_recommends[u] = reco
    if no_rec_num > 0:
        # print(f"{no_rec_num} users has no recommendation")
        users = list(set(users).difference(no_rec_users))
    return y_recommends, users


def choose_pred_func(model):
    pure_models = ["SVD", "SVDpp", "ALS", "BPR", "NCF", "YouTubeMatch",
                   "Caser", "RNN4Rec", "WaveNet", "UserCF", "ItemCF",
                   "KnnEmbedding", "KnnEmbeddingApproximate"]
    if model.__class__.__name__ in pure_models:
        pred_func = predict_pure
    else:
        pred_func = predict_tf_feat
    return pred_func


def predict_pure(model, transformed_data, batch_slice):
    user_indices, item_indices, labels, _, _ = transformed_data[batch_slice]
    preds = model.predict(user_indices, item_indices, inner_id=True)
    if isinstance(preds, np.ndarray):
        preds = preds.tolist()
    elif not isinstance(preds, list):
        preds = [preds]
    return preds


def predict_tf_feat(model, transformed_data, batch_slice):
    (
        user_indices,
        item_indices,
        labels,
        sparse_indices,
        dense_values
    ) = transformed_data[batch_slice]

    if hasattr(model, "user_last_interacted"):
        feed_dict = model._get_seq_feed_dict(
            model.user_last_interacted[user_indices],
            model.last_interacted_len[user_indices],
            user_indices, item_indices, None,
            sparse_indices, dense_values, False
        )
    else:
        feed_dict = model._get_feed_dict(
            user_indices, item_indices, sparse_indices,
            dense_values, None, False
        )

    preds = model.sess.run(model.output, feed_dict)
    if model.task == "rating":
        preds = np.clip(preds, model.lower_bound, model.upper_bound)
    elif model.task == "ranking":
        preds = 1 / (1 + np.exp(-preds))
    return preds.tolist() if isinstance(preds, np.ndarray) else [preds]
