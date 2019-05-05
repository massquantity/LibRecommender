import time
import numpy as np
import tensorflow as tf
from ..utils.sampling import negative_sampling, pairwise_sampling
from ..utils.initializers import truncated_normal
from ..evaluate.evaluate import precision_tf, AP_at_k, MAP_at_k, HitRatio_at_k, NDCG_at_k, binary_cross_entropy
from sklearn.metrics import roc_auc_score, average_precision_score


class BPR:
    def __init__(self, n_factors=16, lr=0.01, n_epochs=20, reg=0.0,
                 iteration=1000, batch_size=64, seed=42):
        self.n_factors = n_factors
        self.lr = lr
        self.n_epochs = n_epochs
        self.reg = reg
        self.iteration = iteration
        self.batch_size = batch_size
        self.seed = seed

    def build_model(self, dataset):
        np.random.seed(self.seed)
        self.global_mean = dataset.global_mean
        self.pu = truncated_normal(shape=[dataset.n_users, self.n_factors], mean=0.0, scale=0.05)
        self.qi = truncated_normal(shape=[dataset.n_items, self.n_factors], mean=0.0, scale=0.05)

    def fit(self, dataset, mode="batch", verbose=1):
        if verbose > 0:
            sampling = pairwise_sampling(dataset)
            train_user, train_item, train_label = sampling(mode="train")
            test_user, test_item, test_label = sampling(mode="test")

        self.dataset = dataset
        self.build_model(self.dataset)
        if mode == "bootstrap":
            sampling = pairwise_sampling(self.dataset)
            t0 = time.time()
            for i in range(1, self.iteration + 1):
                user, item_i, item_j, x_uij = sampling.next_mf(self.pu, self.qi, bootstrap=True)
                sigmoid = 1.0 / (1.0 + np.exp(x_uij))
                self.pu[user] += self.lr * (sigmoid * (self.qi[item_i] - self.qi[item_j]) +
                                            self.reg * self.pu[user])
                self.qi[item_i] += self.lr * (sigmoid * self.pu[user] + self.reg * self.qi[item_i])
                self.qi[item_j] += self.lr * (sigmoid * (-self.pu[user]) + self.reg * self.qi[item_j])

                if i % 80000 == 0:
                    print("iter time: {:.2f}".format(time.time() - t0))

        elif mode == "sgd":
            for epoch in range(1, self.n_epochs + 1):
                t0 = time.time()
                sampling = pairwise_sampling(self.dataset, batch_size=1)
                for _ in range(len(self.dataset.train_user_indices)):
                    user, item_i, item_j, x_uij = sampling.next_mf(self.pu, self.qi, bootstrap=False)
                    sigmoid = 1.0 / (1.0 + np.exp(x_uij))
                    self.pu[user] += self.lr * (sigmoid * (self.qi[item_i] - self.qi[item_j]) +
                                                self.reg * self.pu[user])
                    self.qi[item_i] += self.lr * (sigmoid * self.pu[user] + self.reg * self.qi[item_i])
                    self.qi[item_j] += self.lr * (sigmoid * (-self.pu[user]) + self.reg * self.qi[item_j])

                if verbose > 0:
                    print("Epoch {}, fit time: {:.2f}".format(epoch, time.time() - t0))
                    train_loss, train_prob = binary_cross_entropy(self, train_user, train_item, train_label)
                    train_roc_auc = roc_auc_score(train_label, train_prob)
                    train_pr_auc = average_precision_score(train_label, train_prob)
                    print("train loss: {:.2f}, train roc auc: {:.2f}, train pr auc: {:.2f}".format(
                        train_loss, train_roc_auc, train_pr_auc))

                    test_loss, test_prob = binary_cross_entropy(self, test_user, test_item, test_label)
                    test_roc_auc = roc_auc_score(test_label, test_prob)
                    test_pr_auc = average_precision_score(test_label, test_prob)
                    print("test loss: {:.2f}, test auc: {:.2f}, test pr auc: {:.2f}".format(
                        test_loss, test_roc_auc, test_pr_auc))
                    print()

        elif mode == "batch":
            for epoch in range(1, self.n_epochs + 1):
                t0 = time.time()
                sampling = pairwise_sampling(self.dataset, batch_size=1)
                n_batches = len(self.dataset.train_user_indices) // self.batch_size
                for n in range(n_batches):
                    batch_user, \
                    batch_item_i, \
                    batch_item_j, \
                    batch_x_uij = sampling.next_mf(self.pu, self.qi, bootstrap=False)

                    sigmoids = 1.0 / (1.0 + np.exp(batch_x_uij))
                    self.pu[batch_user] += self.lr * (
                            sigmoids * (
                                self.qi[batch_item_i] - self.qi[batch_item_j]) +
                                    self.reg * self.pu[batch_user])
                    self.qi[batch_item_i] += self.lr * (sigmoids * self.pu[batch_user] +
                                                        self.reg * self.qi[batch_item_i])
                    self.qi[batch_item_j] += self.lr * (sigmoids * (-self.pu[batch_user]) +
                                                        self.reg * self.qi[batch_item_j])

                if verbose > 0:
                    print("Epoch {}, fit time: {:.2f}".format(epoch, time.time() - t0))
                    train_loss, train_prob = binary_cross_entropy(self, train_user, train_item, train_label)
                    train_roc_auc = roc_auc_score(train_label, train_prob)
                    train_pr_auc = average_precision_score(train_label, train_prob)
                    print("train loss: {:.2f}, train roc auc: {:.2f}, train pr auc: {:.2f}".format(
                        train_loss, train_roc_auc, train_pr_auc))

                    test_loss, test_prob = binary_cross_entropy(self, test_user, test_item, test_label)
                    test_roc_auc = roc_auc_score(test_label, test_prob)
                    test_pr_auc = average_precision_score(test_label, test_prob)
                    print("test loss: {:.2f}, test auc: {:.2f}, test pr auc: {:.2f}".format(
                        test_loss, test_roc_auc, test_pr_auc))
                    print()

        else:
            raise ValueError("mode must be one of these: bootstrap, sgd, batch")

    def predict(self, u, i):
        try:
            logits = np.dot(self.pu[u], self.qi[i])
            prob = 1.0 / (1.0 + np.exp(-logits))
            pred = float(np.where(prob >= 0.5, 1.0, 0.0))
        except IndexError:
            prob, pred = 0.0, 0.0
        return prob, pred































