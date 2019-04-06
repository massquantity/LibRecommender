import numpy as np


def rmse_user_knn(model, dataset, mode="train"):
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
    score = np.sqrt(np.mean(np.power(np.array(pred) - ratings, 2)))
    return score

