import numpy as np


class negative_sampling:
    def __init__(self, dataset, num_neg, batch_size):
        self.dataset = dataset
        self.num_neg = num_neg
        self.batch_size = batch_size
        self.i = 0
        self.__init_sampling()

    def __init_sampling(self):
        self.user_negative_pool = {}
        for u in self.dataset.train_user:
            self.user_negative_pool[u] = list(set(range(self.dataset.n_items)) - set(self.dataset.train_user[u]))

    def next_batch(self):
        end = min(len(self.dataset.train_user_indices), (self.i + 1) * self.batch_size)
        batch_pos_user = self.dataset.train_user_indices[self.i * self.batch_size: end]
        batch_pos_item = self.dataset.train_item_indices[self.i * self.batch_size: end]
        batch_pos_label = self.dataset.train_labels[self.i * self.batch_size: end]

        batch_user_indices, batch_item_indices, batch_label_indices = [], [], []
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

        self.i += 1
    #    print("orig: {} \n {} \n {}".format(batch_user_indices, batch_item_indices, batch_label_indices))
    #    np.random.seed(42)
        indices = np.random.permutation(len(batch_user_indices))
        return np.array(batch_user_indices)[indices], \
               np.array(batch_item_indices)[indices], \
               np.array(batch_label_indices)[indices]















