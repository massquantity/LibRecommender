import numpy as np


def baseline_als(dataset, n_epochs=100, reg_u=15, reg_i=10):
    train_user = dataset.train_user
    train_item = dataset.train_item

    bu = np.zeros((dataset.n_users))  # np.random.normal
    bi = np.zeros((dataset.n_items))
    global_mean = dataset.global_mean
    count = 0
    for epoch in range(n_epochs):
        for u in train_user.keys():
            before_bu = bu.copy()
            u_length = len(train_user[u])
            u_items = np.array(list(train_user[u].keys()))
            u_ratings = np.array(list(train_user[u].values()))
            bu[u] = (np.sum(u_ratings) - u_length * global_mean - np.sum(bi[u_items])) / (reg_u + u_length)

        for i in train_item.keys():
            before_bi = bi.copy()
            i_length = len(train_item[i])
            i_users = np.array(list(train_item[i].keys()))
            i_ratings = np.array(list(train_item[i].values()))
            bi[i] = (np.sum(i_ratings) - i_length * global_mean - np.sum(bu[i_users])) / (reg_i + i_length)

        if np.allclose(before_bu, bu, atol=1e-5) and np.allclose(before_bi, bi, atol=1e-5):
            count += 1
            if count > 3:
                print("epoch {} converged !".format(epoch))
                break
    return bu, bi


def ratings(dataset):
    for user, r in dataset.items():
        for item, rating in r.items():
            yield user, item, rating


def baseline_sgd(dataset, n_epochs=20, reg=0.01, lr=0.01):
    train_user = dataset.train_user
    train_item = dataset.train_item

    bu = np.zeros((dataset.n_users))
    bi = np.zeros((dataset.n_items))
    global_mean = dataset.global_mean
    for epoch in range(n_epochs):
        for u, i, r in ratings(train_user):
            before_bu, before_bi = bu.copy(), bi.copy()
            err = r - (global_mean + bu[u] + bi[i])
            bu[u] += lr * (err - reg * bu[u])
            bi[i] += lr * (err - reg * bi[i])

        if np.allclose(before_bu, bu, atol=1e-3) and np.allclose(before_bi, bi, atol=1e-3):
            print("epoch {} converged !".format(epoch))
            break
    return bu, bi
