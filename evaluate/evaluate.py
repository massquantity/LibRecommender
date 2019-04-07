import numpy as np


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



'''
    # score = 0
    pred = []
    for u, i, r in zip(user_indices, item_indices, ratings):
        p = model.predict(u, i)
        pred.append(p)
    score = np.sqrt(np.mean(np.power(np.array(pred) - ratings, 2)))
    return score

        try:
            pred = model.predict(u, i)
        except IndexError:
            pred = dataset.global_mean
        score += np.power((r - pred), 2)
    return np.sqrt(score / len(user_indices))



        pred = global_mean + \
               model.bu[user_indices] + \
               model.bi[item_indices] + \
               np.dot(model.pu, model.qi.T)[user_indices, item_indices]
    else:
        pred = np.dot(model.pu, model.qi.T)[user_indices, item_indices]
    score = np.sqrt(np.mean(np.power(pred - ratings, 2)))
    return score
'''