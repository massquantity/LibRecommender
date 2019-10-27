from collections import defaultdict
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score


def rmse(model, dataset, mode="train"):
    if mode == "train":
        user_indices = dataset.train_user_indices
        item_indices = dataset.train_item_indices
        labels = dataset.train_labels
    elif mode == "test":
        user_indices = dataset.test_user_indices
        item_indices = dataset.test_item_indices
        labels = dataset.test_labels

    pred = []
    for j, (u, i) in enumerate(zip(user_indices, item_indices)):
        p = model.predict(u, i)
        pred.append(p)
    score = np.sqrt(np.mean(np.power(pred - labels, 2)))
    return score

def accuracy(model, user, item, labels):
    pred = []
    for j, (u, i) in enumerate(zip(user, item)):
        p = model.predict(u, i)[1]
        pred.append(p)

    score = np.sum(labels == np.array(pred)) / len(labels)
    return score

def rmse_tf(model, dataset, mode="train"):
    if mode == "train":
        user_indices = dataset.train_user_indices
        item_indices = dataset.train_item_indices
        ratings = dataset.train_ratings
    elif mode == "test":
        user_indices = dataset.test_user_indices
        item_indices = dataset.test_item_indices
        ratings = dataset.test_ratings

    rmse = model.sess.run(model.metrics, feed_dict={model.user_indices: user_indices,
                                                    model.item_indices: item_indices,
                                                    model.ratings: ratings})
    return rmse


def precision_tf(pred, y):
    one = tf.constant(1, dtype=tf.float32)
    mask = tf.equal(pred, one)
    y_chosen = tf.boolean_mask(y, mask)
    precision = tf.reduce_sum(y_chosen) / tf.cast(tf.shape(y_chosen), tf.float32)
    return precision[0]


def AP_at_k(model, dataset, u, k, **kwargs):
    true_items = dataset.test_item_indices[np.where(dataset.test_user_indices == u)]
    if len(true_items) == 0:
        return -1
    rank_list = model.recommend_user(u, k, **kwargs)
    if rank_list == -1 or len(rank_list) == 0:
        return -1

    top_k = [i[0] for i in rank_list]
    precision_k = 0
    count_relevant_k = 0
    for i in range(1, k + 1):
        if i > len(top_k):
            break
        precision_i = 0
        if top_k[i - 1] in true_items:
            count_relevant_k += 1
            for pred in top_k[:i]:
                if pred in true_items:
                    precision_i += 1
            precision_k += precision_i / i
        else:
            continue
    try:
        average_precision_at_k = precision_k / count_relevant_k
    except ZeroDivisionError:
        average_precision_at_k = 0.0
    return average_precision_at_k


def MAP_at_k(model, dataset, k, sample_user=None, **kwargs):
    average_precision_at_k = []
    if sample_user is not None:
        assert isinstance(sample_user, int), "sampled users must be integer"
        np.random.seed(42)
        users = np.random.choice(list(dataset.train_user), sample_user, replace=False)
    else:
        users = list(dataset.train_user)

    i = 0
    for u in users:
        AP = AP_at_k(model, dataset, u, k, **kwargs)
        if AP == -1:
            i += 1
            continue
        else:
            average_precision_at_k.append(AP)
    mean_average_precision_at_k = np.mean(average_precision_at_k)
    print("\tMAP None users: ", i)
    return mean_average_precision_at_k


def AR_at_k(model, dataset, u, k):
    """
    average recall at k
    """
    true_items = dataset.test_item_indices[np.where(dataset.test_user_indices == u)]
    if len(true_items) == 0:
        return -1
    rank_list = model.recommend_user(u, k)
    top_k = [i[0] for i in rank_list]
    recall_k = 0
    count_relevant_k = 0
    for i in range(1, k + 1):
        recall_i = 0
        if top_k[i-1] in true_items:
            count_relevant_k += 1
            for pred in top_k[:i]:
                if pred in true_items:
                    recall_i += 1
            recall_k += recall_i / len(true_items)
        else:
            continue
    try:
        average_recall_at_k = recall_k / count_relevant_k
    except ZeroDivisionError:
        average_recall_at_k = 0.0
    return average_recall_at_k


def MAR_at_k(model, dataset, k, sample_user=None):
    """
    mean average recall at k
    """
    average_recall_at_k = []
    if sample_user is not None:
        assert isinstance(sample_user, int), "sampled users must be integer"
        np.random.seed(42)
        users = np.random.choice(list(dataset.train_user), sample_user, replace=False)
    else:
        users = list(dataset.train_user)

    i = 0
    for u in users:
        AR = AR_at_k(model, dataset, u, k)
        if AR == -1:
            i += 1
            continue
        else:
            average_recall_at_k.append(AR)
    mean_average_recall_at_k = np.mean(average_recall_at_k)
    print("\tMAR None users: ", i)
    return mean_average_recall_at_k


def recall_at_k(model, dataset, k, sample_user=None):
    if sample_user is not None:
        assert isinstance(sample_user, int), "sampled users must be integer"
        np.random.seed(42)
        users = np.random.choice(list(dataset.train_user), sample_user, replace=False)
    else:
        users = list(dataset.train_user)

    recall = []
    i = 0
    for u in users:
        true_items = dataset.test_item_indices[np.where(dataset.test_user_indices == u)]
        if len(true_items) == 0:
            i += 1
            continue
        else:
            rank_list = model.recommend_user(u, k)
            if rank_list == -1 or len(rank_list) == 0:
                i += 1
                continue
            top_k = [i[0] for i in rank_list]
            recall_i = 0
            for pred in top_k:
                if pred in true_items:
                    recall_i += 1
            recall.append(recall_i / len(true_items))
    print("\trecall None users: ", i)
    return np.mean(recall)


def HitRatio_at_k(model, dataset, k):
    HitRatio = []
    for u in dataset.train_user:
        true_items = dataset.test_item_indices[np.where(dataset.test_user_indices == u)]
        if len(true_items) == 0:
            continue
        user_HitRatio = 0
        rank_list = model.recommend_user(u, k)
        top_k = [i[0] for i in rank_list]
        for i in top_k:
            if i in true_items:
                user_HitRatio += 1
        HitRatio.append(user_HitRatio / k)
    return np.mean(HitRatio)


def NDCG_at_k(model, dataset, k, sample_user=None, mode="normal"):
    if mode.lower() == "wide_deep":
        test_user_indices = dataset.test_data.loc[dataset.test_data.label == 1.0, "user"].to_numpy()
        test_item_indices = dataset.test_data.loc[dataset.test_data.label == 1.0, "item"].to_numpy()
        u_items = dataset.user_dict
    else:
        test_user_indices = dataset.test_user_indices
        test_item_indices = dataset.test_item_indices
        u_items = dataset.train_user

    if sample_user is not None:
        assert isinstance(sample_user, int), "sampled users must be integer"
        np.random.seed(42)
        users = np.random.choice(list(u_items), sample_user, replace=False)
    else:
        users = list(u_items)

    NDCG = []
    i = 0
    for u in users:
        DCG = 0
        IDCG = 0
        true_items = test_item_indices[np.where(test_user_indices == u)]
        if len(true_items) == 0:
            i += 1
            continue
        rank_list = model.recommend_user(u, k)
        if rank_list == -1 or len(rank_list) == 0:
            i += 1
            continue
        top_k = [i[0] for i in rank_list]
        for n, item in enumerate(top_k):
            if item in true_items:
                DCG += np.reciprocal(np.log2(n + 2))
        optimal_items = min(len(true_items), k)
        for n in range(optimal_items):
            IDCG += np.reciprocal(np.log2(n + 2))
        NDCG.append(DCG / IDCG)
    print("\tNDCG None users: ", i)
    return np.mean(NDCG)


def NDCG_at_k_tf(labels, predictions, k):
    _, indices = tf.nn.top_k(predictions, k, sorted=True)
    n = tf.cast(tf.range(1, k + 1), tf.float32)
    top_k = tf.gather(labels, indices)
    denominator = tf.log(n + 1) / tf.log(2.0) # logarithm base 2
    dcg_numerator = tf.pow(2.0, top_k) - 1.0
    DCG = tf.reduce_sum(dcg_numerator / denominator, axis=1, keep_dims=True)
    IDCG = tf.reduce_sum(1.0 / denominator, axis=1, keep_dims=True)
    NDCG = DCG / IDCG
    return tf.metrics.mean(NDCG)


def binary_cross_entropy(model, user, item, label):
    eps = 1e-7
    ce = []
    probs = []
    for u, i, l in zip(user, item, label):
        prob = model.predict(u, i)
        probs.append(prob)
    #    if prob == 0.0 or prob == 1.0:
    #        continue
        if l == 1.0 and prob >= eps:
            ce.append(-np.log(prob))
        elif l == 1.0 and prob < eps:
            ce.append(-np.log(eps))
        elif l == 0.0 and prob <= 1.0 - eps:
            ce.append(-np.log(1.0 - prob))
        elif l == 0.0 and prob > 1.0 - eps:
            ce.append(-np.log(1.0 - eps))
    return np.mean(ce), probs



