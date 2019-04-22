import time
from collections import defaultdict
import numpy as np


class negative_sampling:
    def __init__(self, dataset, num_neg, batch_size=64, seed=42, replacement_sampling=False):
        self.dataset = dataset
        self.num_neg = num_neg
        self.batch_size = batch_size
        self.i = 0
        self.item_pool = defaultdict(set)
        if not replacement_sampling:
            self.__init_sampling()

    def __init_sampling(self):
        self.user_negative_pool = {}
        for u in self.dataset.train_user:
            self.user_negative_pool[u] = list(set(range(self.dataset.n_items)) - set(self.dataset.train_user[u]))

    def __call__(self, mode):
        if mode == "train":
            user_indices = self.dataset.train_user_indices
            item_indices = self.dataset.train_item_indices
            label_indices = self.dataset.train_labels
            timestamp_indices = self.dataset.train_timestamp_bin
        elif mode == "test":
            user_indices = self.dataset.test_user_indices
            item_indices = self.dataset.test_item_indices
            label_indices = self.dataset.test_labels
            timestamp_indices = self.dataset.test_timestamp_bin

        user, item, label, timestamp = [], [], [], []
        for i, u in enumerate(user_indices):
            user.append(user_indices[i])
            item.append(item_indices[i])
            label.append(label_indices[i])
            timestamp.append(timestamp_indices[i])
            for _ in range(self.num_neg):
                item_neg = np.random.randint(0, self.dataset.n_items - 1)
                while item_neg in self.dataset.train_user[u]:
                    item_neg = np.random.randint(0, self.dataset.n_items - 1)

                user.append(u)
                item.append(item_neg)
                label.append(0.0)
                timestamp.append(timestamp_indices[i])
        return np.array(user), \
               np.array(item), \
               np.array(label), \
               np.array(timestamp)



    def next_batch_99(self):
        end = min(len(self.dataset.train_user_indices), (self.i + 1) * self.batch_size)
        batch_pos_user = self.dataset.train_user_indices[self.i * self.batch_size: end]
        batch_pos_item = self.dataset.train_item_indices[self.i * self.batch_size: end]
        batch_pos_label = self.dataset.train_labels[self.i * self.batch_size: end]

        batch_user_indices, batch_item_indices, batch_label_indices = [], [], []
        t0 = time.time()
        for i, u in enumerate(batch_pos_user):
            batch_user_indices.append(batch_pos_user[i])
            batch_item_indices.append(batch_pos_item[i])
            batch_label_indices.append(batch_pos_label[i])

            batch_user_indices.extend([u] * self.num_neg)
            item_neg = np.random.choice(self.user_negative_pool[u], self.num_neg, replace=False)
            batch_item_indices.extend(item_neg)
            batch_label_indices.extend([0.0] * self.num_neg)
            self.user_negative_pool[u] = list(set(self.user_negative_pool[u]) - set(item_neg))
            if len(self.user_negative_pool[u]) < self.num_neg + 1:
                self.user_negative_pool[u] = list(set(range(self.dataset.n_items)) - set(self.dataset.train_user[u]))
                print("negative pool exhausted")
    #    print("batch time: {:.4f}".format(time.time() - t0))

        self.i += 1
    #    print("orig: {} \n {} \n {}".format(batch_user_indices, batch_item_indices, batch_label_indices))
    #    np.random.seed(42)
        indices = np.random.permutation(len(batch_user_indices))
        return np.array(batch_user_indices)[indices], \
               np.array(batch_item_indices)[indices], \
               np.array(batch_label_indices)[indices]


    def next_batch_67876(self):
        end = min(len(self.dataset.train_user_indices), (self.i + 1) * self.batch_size)
        batch_pos_user = self.dataset.train_user_indices[self.i * self.batch_size: end]
        batch_pos_item = self.dataset.train_item_indices[self.i * self.batch_size: end]
        batch_pos_label = self.dataset.train_labels[self.i * self.batch_size: end]

        batch_user_indices, batch_item_indices, batch_label_indices = [], [], []
        t0 = time.time()
        for i, u in enumerate(batch_pos_user):
            batch_user_indices.append(batch_pos_user[i])
            batch_item_indices.append(batch_pos_item[i])
            batch_label_indices.append(batch_pos_label[i])

            for _ in range(self.num_neg):
                item_neg = np.random.randint(0, self.dataset.n_items - 1)
        #        while item_neg in self.dataset.train_user[u]:
                while self.dataset.train_user[u].__contains__(item_neg) or item_neg in self.item_pool[u]:
                    item_neg = np.random.randint(0, self.dataset.n_items - 1)  # resample
                self.item_pool[u].add(item_neg)
                if len(self.item_pool[u]) > (len(self.dataset.train_user[u]) * 0.75):
                    self.item_pool[u].clear()
                #    print("negative pool exhausted")

                batch_user_indices.append(u)
                batch_item_indices.append(item_neg)
                batch_label_indices.append(0.0)
    #    print("batch time: {:.4f}".format(time.time() - t0))

        self.i += 1
        indices = np.random.permutation(len(batch_user_indices))
        return np.array(batch_user_indices)[indices], \
               np.array(batch_item_indices)[indices], \
               np.array(batch_label_indices)[indices]


    def next_batch(self):
        end = min(len(self.dataset.train_user_indices), (self.i + 1) * self.batch_size)
        batch_pos_user = self.dataset.train_user_indices[self.i * self.batch_size: end]
        batch_pos_item = self.dataset.train_item_indices[self.i * self.batch_size: end]
        batch_pos_label = self.dataset.train_labels[self.i * self.batch_size: end]

        batch_user_indices, batch_item_indices, batch_label_indices = [], [], []
        t0 = time.time()
        for i, u in enumerate(batch_pos_user):
            batch_user_indices.append(batch_pos_user[i])
            batch_item_indices.append(batch_pos_item[i])
            batch_label_indices.append(batch_pos_label[i])

            for _ in range(self.num_neg):
                item_neg = np.random.randint(0, self.dataset.n_items - 1)
        #        while item_neg in self.dataset.train_user[u]:
                while self.dataset.train_user[u].__contains__(item_neg):
                    item_neg = np.random.randint(0, self.dataset.n_items - 1)  # resample

                batch_user_indices.append(u)
                batch_item_indices.append(item_neg)
                batch_label_indices.append(0.0)
    #    print("batch time: {:.4f}".format(time.time() - t0))

        self.i += 1
        indices = np.random.permutation(len(batch_user_indices))
        return np.array(batch_user_indices)[indices], \
               np.array(batch_item_indices)[indices], \
               np.array(batch_label_indices)[indices]





