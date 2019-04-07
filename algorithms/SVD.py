import time
from operator import itemgetter
import numpy as np
from ..evaluate import rmse_svd


class SVD:
    def __init__(self, n_factors=100, n_epochs=20, reg=5.0, seed=42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.reg = reg
        self.seed = seed

    def fit(self, dataset):
        np.random.seed(self.seed)
        self.dataset = dataset
        self.default_prediction = dataset.global_mean
        self.pu = np.random.normal(loc=0.0, scale=0.1,
                                   size=(dataset.n_users, self.n_factors))
        self.qi = np.random.normal(loc=0.0, scale=0.1,
                                   size=(dataset.n_items, self.n_factors))

        for epoch in range(1, self.n_epochs + 1):
            t0 = time.time()
            for u in dataset.train_user:
                u_items = np.array(list(dataset.train_user[u].keys()))
                u_ratings = np.array(list(dataset.train_user[u].values()))
                u_ratings_expand = np.expand_dims(u_ratings, axis=1)
                yy_reg = self.qi[u_items].T.dot(self.qi[u_items]) + \
                         self.reg * np.eye(self.n_factors)
                r_y = np.sum(np.multiply(u_ratings_expand, self.qi[u_items]), axis=0)
                self.pu[u] = np.linalg.inv(yy_reg).dot(r_y)

            for i in dataset.train_item:
                i_users = np.array(list(dataset.train_item[i].keys()))
            #    if len(i_users) == 0: continue
                i_ratings = np.array(list(dataset.train_item[i].values()))
                i_ratings_expand = np.expand_dims(i_ratings, axis=1)
                xx_reg = self.pu[i_users].T.dot(self.pu[i_users]) + \
                         self.reg * np.eye(self.n_factors)
                r_x = np.sum(np.multiply(i_ratings_expand, self.pu[i_users]), axis=0)
                self.qi[i] = np.linalg.inv(xx_reg).dot(r_x)


            if epoch % 5 == 0:
                print("Epoch {} time: {:.4f}".format(epoch, time.time() - t0))
                print("training rmse: ", self.rmse(dataset, "train"))
                print("test rmse: ", self.rmse(dataset, "test"))
        return self

    def predict(self, u, i):
        try:
            pred = np.dot(self.pu[u], self.qi[i])
            pred = np.clip(pred, 1, 5)
        except IndexError:
            pred = self.default_prediction
        return pred

    def topN(self, u, n_rec, random_rec=False):
        rank = [(j, self.predict(u, j)) for j in range(len(self.qi))
                if j not in self.dataset.train_user[u]]
        if random_rec:
            item_pred_dict = {j: r for j, r in rank if r >= 4}
            item_list = list(item_pred_dict.keys())
            pred_list = list(item_pred_dict.values())
            p = [p / np.sum(pred_list) for p in pred_list]
            item_candidates = np.random.choice(item_list, n_rec, replace=False, p=p)
            reco = [(item, item_pred_dict[item]) for item in item_candidates]
            return reco
        else:
            rank.sort(key=itemgetter(1), reverse=True)
            return rank[:n_rec]



    def rmse(self, dataset, mode="train"):
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
            p = self.predict(u, i)
            pred.append(p)
        score = np.sqrt(np.mean(np.power(pred - ratings, 2)))
        return score


'''
class SVD_tf:
    def __init__(self):
        import tensorflow as tf
'''