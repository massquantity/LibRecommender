import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score


def rmse_knn(model, dataset, mode="train"):
    if mode == "train":
        user_indices = dataset.train_user_indices
        item_indices = dataset.train_item_indices
        ratings = dataset.train_ratings
    elif mode == "test":
        user_indices = dataset.test_user_indices
        item_indices = dataset.test_item_indices
        ratings = dataset.test_ratings

    pred = []
    for j, (u, i) in enumerate(zip(user_indices, item_indices)):
        p = model.predict(u, i)
        pred.append(p)
    score = np.sqrt(np.mean(np.power(pred - ratings, 2)))
    return score


def rmse_svd(model, dataset, baseline=False, mode="train"):
    if mode == "train":
        user_indices = dataset.train_user_indices
        item_indices = dataset.train_item_indices
        ratings = dataset.train_ratings
    elif mode == "test":
        user_indices = dataset.test_user_indices
        item_indices = dataset.test_item_indices
        ratings = dataset.test_ratings

    pred = []
    for u, i in zip(user_indices, item_indices):
        p = model.predict(u, i)
        pred.append(p)
    score = np.sqrt(np.mean(np.power(pred - ratings, 2)))
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

#TODO

def precision_tf(pred, y):
    one = tf.constant(1, dtype=tf.float32)
    mask = tf.equal(pred, one)
    y_chosen = tf.boolean_mask(y, mask)
    precision = tf.reduce_sum(y_chosen) / tf.cast(tf.shape(y_chosen), tf.float32)
    return precision[0]

def AP_at_k(model, dataset, u, k):
#    user_indices = np.full(dataset.n_items, u)
#    item_indices = np.arange(dataset.n_items)
#    y_ranklist = self.sess.run(self.y_prob, feed_dict={self.user_indices: user_indices,
#                                                       self.item_indices: item_indices})

#    top_k, indices = tf.nn.top_k(ranklist, k, sorted=True)

#    ranklist = model.predict_user(u)
#    y_pred_k, indices = np.sort(ranklist)[::-1][:k], np.argsort(ranklist)[::-1][:k]
#    y_true_k = np.arange(dataset.n_items)[indices]
#    precision = 0
#    for i, pred in enumerate(y_pred_k, start=1):
#        if pred in dataset.train_user[u]:
#            precision += precision_score(y_true_k[:i], y_pred_k[:i])
#    average_precision_at_k = precision / k

    ranklist = model.predict_user(u)
    top_k = np.argsort(ranklist)[::-1][:k]
    precision_k = 0
    count_relevant_k = 0
    for i in range(1, k + 1):
        precision_i = 0
        if top_k[i-1] in dataset.train_user[u]:
            count_relevant_k += 1
            for pred in top_k[:i]:
                if pred in dataset.train_user[u]:
                    precision_i += 1
            precision_k += precision_i / i
        else:
            continue
    try:
        average_precision_at_k = precision_k / count_relevant_k
    except ZeroDivisionError:
        average_precision_at_k = 0.0
    return average_precision_at_k


def MAP_at_k(model, dataset, k):
    average_precision_at_k = 0
    for u in dataset.train_user:
        average_precision_at_k += AP_at_k(model, dataset, u, k)
    mean_average_precision_at_k = average_precision_at_k / dataset.n_users
    return mean_average_precision_at_k





# def recall

# def f1



