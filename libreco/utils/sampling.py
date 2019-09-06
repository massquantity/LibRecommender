import time
from collections import defaultdict
from multiprocessing import Pool
import numpy as np


class NegativeSampling:
    def __init__(self, dataset, num_neg, batch_size=64, seed=42, replacement_sampling=True):
        self.dataset = dataset
        self.num_neg = num_neg
        self.batch_size = batch_size
        self.seed = seed
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
        elif mode == "test":
            user_indices = self.dataset.test_user_indices
            item_indices = self.dataset.test_item_indices
            label_indices = self.dataset.test_labels

        '''
        user, item, label = [], [], []
        for i, u in enumerate(user_indices):
            user.append(user_indices[i])
            item.append(item_indices[i])
            label.append(label_indices[i])
            for _ in range(self.num_neg):
                item_neg = np.random.randint(0, self.dataset.n_items)
                while item_neg in self.dataset.train_user[u]:
                    item_neg = np.random.randint(0, self.dataset.n_items)

                user.append(u)
                item.append(item_neg)
                label.append(0.0)
        return np.array(user), np.array(item), np.array(label)
        '''


        user_implicit = np.tile(user_indices, self.num_neg + 1)
        label_negative = np.zeros(len(user_indices) * self.num_neg, dtype=np.float32)
        label_implicit = np.concatenate([label_indices, label_negative])
        item_negative = []
        for u, i in zip(user_indices, item_indices):
            for _ in range(self.num_neg):
                item_neg = np.random.randint(0, self.dataset.n_items)
                while item_neg in self.dataset.train_user[u]:
                    item_neg = np.random.randint(0, self.dataset.n_items)

                item_negative.append(item_neg)
        item_implicit = np.concatenate([item_indices, item_negative])
        return user_implicit, item_implicit, label_implicit


    def next_batch_without_replacement(self):
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
        indices = np.random.permutation(len(batch_user_indices))
        return np.array(batch_user_indices)[indices], \
               np.array(batch_item_indices)[indices], \
               np.array(batch_label_indices)[indices]

    def next_batch(self):
        batch_size = int(self.batch_size / (self.num_neg + 1))
        end = min(len(self.dataset.train_user_indices), (self.i + 1) * batch_size)
        batch_pos_user = self.dataset.train_user_indices[self.i * batch_size: end]
        batch_pos_item = self.dataset.train_item_indices[self.i * batch_size: end]
        batch_pos_label = self.dataset.train_labels[self.i * batch_size: end]

    #    end = min(len(self.dataset.train_user_indices), (self.i + 1) * self.batch_size)
    #    batch_pos_user = self.dataset.train_user_indices[self.i * self.batch_size: end]
    #    batch_pos_item = self.dataset.train_item_indices[self.i * self.batch_size: end]
    #    batch_pos_label = self.dataset.train_labels[self.i * self.batch_size: end]

        batch_user_indices, batch_item_indices, batch_label_indices = [], [], []
        for i, u in enumerate(batch_pos_user):
            batch_user_indices.append(batch_pos_user[i])
            batch_item_indices.append(batch_pos_item[i])
            batch_label_indices.append(batch_pos_label[i])

            for _ in range(self.num_neg):
                item_neg = np.random.randint(0, self.dataset.n_items)
        #        while item_neg in self.dataset.train_user[u]:
                while self.dataset.train_user[u].__contains__(item_neg):
                    item_neg = np.random.randint(0, self.dataset.n_items)  # resample

                batch_user_indices.append(u)
                batch_item_indices.append(item_neg)
                batch_label_indices.append(0.0)

        self.i += 1
        indices = np.random.permutation(len(batch_user_indices))
        return np.array(batch_user_indices)[indices], \
               np.array(batch_item_indices)[indices], \
               np.array(batch_label_indices)[indices]


class NegativeSamplingFeat_67876:
    def __init__(self, dataset, num_neg, batch_size=64, seed=42, replacement_sampling=True, item_cols=None):
        self.dataset = dataset
        self.num_neg = num_neg
        self.batch_size = batch_size
        self.seed = seed
        self.i = 0
        self.item_pool = defaultdict(set)
        self.item_cols = item_cols
        if not replacement_sampling:
            self.__init_sampling()

    def __init_sampling(self):
        self.user_negative_pool = {}
        for u in self.dataset.train_user:
            self.user_negative_pool[u] = list(set(range(self.dataset.n_items)) - set(self.dataset.train_user[u]))

    def __call__(self, mode):
        if mode == "train":
            feat_indices = self.dataset.train_feat_indices
            feat_values = self.dataset.train_feat_values
            feat_labels = self.dataset.train_labels
        elif mode == "test":
            feat_indices = self.dataset.test_feat_indices
            feat_values = self.dataset.test_feat_values
            feat_labels = self.dataset.test_labels

        indices, values, labels = [], [], []

        for i, sample in enumerate(feat_indices):
            u = sample.copy()
            user = u[-2] - self.dataset.user_offset
            indices.append(feat_indices[i])
            values.append(feat_values[i])
            labels.append(feat_labels[i])
            for _ in range(self.num_neg):
                item_neg = np.random.randint(0, self.dataset.n_items)
                while item_neg in self.dataset.train_user[user]:
                    item_neg = np.random.randint(0, self.dataset.n_items)

                item_neg += (self.dataset.user_offset + self.dataset.n_users)

                dt = self.dataset.train_feat_indices.copy()
                item_cols = dt[dt[:, -1] == item_neg][0, self.dataset.item_feature_cols]
                ss = sample.copy()
                ss[-1] = item_neg
                ss[self.dataset.item_feature_cols] = item_cols.copy()

            #    ss = sample.copy()
            #    ss[-1] = item_neg
                indices.append(ss)
                values.append(feat_values[i])
                labels.append(0.0)
        '''
        for i, sample in enumerate(feat_indices):
            ss = sample.tolist()
            user = ss[-2] - self.dataset.user_offset
            indices.append(feat_indices[i])
            values.append(feat_values[i])
            labels.append(feat_labels[i])
            for _ in range(self.num_neg):
                item_neg = np.random.randint(0, self.dataset.n_items)
                while item_neg in self.dataset.train_user[user]:
                    item_neg = np.random.randint(0, self.dataset.n_items)

                item_neg += (self.dataset.user_offset + self.dataset.n_users)

                dt = self.dataset.train_feat_indices[self.dataset.train_feat_indices[:, -1] == item_neg].tolist()[0]
                for col in self.dataset.item_feature_cols:
                    ss[col] = dt[col]

            #    item_cols = dt[dt[:, -1] == item_neg][0, self.dataset.item_feature_cols]
            #    ss = sample.copy()
            #    ss[-1] = item_neg
            #    ss[self.dataset.item_feature_cols] = item_cols.copy()

            #    ss = sample.copy()
            #    ss[-1] = item_neg
                indices.append(ss)
                values.append(feat_values[i])
                labels.append(0.0)
        '''
        return np.array(indices), np.array(values), np.array(labels)

    def next_batch_without_replacement(self):  # change to feat version
        # {i: list(train_user[i].keys()) for i in train_user}
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
        indices = np.random.permutation(len(batch_user_indices))
        return np.array(batch_user_indices)[indices], \
               np.array(batch_item_indices)[indices], \
               np.array(batch_label_indices)[indices]

    def next_batch(self):
        batch_size = int(self.batch_size / (self.num_neg + 1))
        end = min(len(self.dataset.train_feat_indices), (self.i + 1) * batch_size)
        batch_feat_indices = self.dataset.train_feat_indices[self.i * batch_size: end]
        batch_feat_values = self.dataset.train_feat_values[self.i * batch_size: end]
        batch_feat_labels = self.dataset.train_labels[self.i * batch_size: end]

    #    end = min(len(self.dataset.train_feat_indices), (self.i + 1) * self.batch_size)  # bs = self.batch_size // 2
    #    batch_feat_indices = self.dataset.train_feat_indices[self.i * self.batch_size: end]  # train_indices_implicit
    #    batch_feat_values = self.dataset.train_feat_values[self.i * self.batch_size: end]  # train_values_implicit
    #    batch_feat_labels = self.dataset.train_labels[self.i * self.batch_size: end]  # train_labels_implicit

        indices, values, labels = [], [], []
        for i, sample in enumerate(batch_feat_indices):
        #    user = sample[-2] - self.dataset.user_offset
            u = sample.copy()
            user = u[-2] - self.dataset.user_offset
            indices.append(batch_feat_indices[i])
            values.append(batch_feat_values[i])
            labels.append(batch_feat_labels[i])
            for _ in range(self.num_neg):
                item_neg = np.random.randint(0, self.dataset.n_items)
        #        while item_neg in self.dataset.train_user[u]:
                while item_neg in self.dataset.train_user[user]:
                    item_neg = np.random.randint(0, self.dataset.n_items)  # resample

                item_neg += (self.dataset.user_offset + self.dataset.n_users)  # item offset
                dt = self.dataset.train_feat_indices.copy()
                item_cols = dt[dt[:, -1] == item_neg][0, self.dataset.item_feature_cols]
                ss = sample.copy()
                ss[-1] = item_neg
                ss[self.dataset.item_feature_cols] = item_cols.copy()
                indices.append(ss)
                values.append(batch_feat_values[i])
                labels.append(0.0)

        self.i += 1
        random_mask = np.random.permutation(len(batch_feat_indices))
        return np.array(indices)[random_mask], np.array(values)[random_mask], np.array(labels)[random_mask]


class NegativeSamplingFeat:
    def __init__(self, dataset, num_neg, batch_size=64, seed=42, replacement_sampling=True,
                 pre_sampling=False):
        self.dataset = dataset
        self.num_neg = num_neg
        self.batch_size = batch_size
        self.seed = seed
        self.i = 0
        self.item_pool = defaultdict(set)
        self.neg_indices_dict, self.neg_values_dict = self.__neg_feat_dict()
        self.pre_sampling = pre_sampling
        if not replacement_sampling:
            self.__init_sampling()
        if not pre_sampling:
            random_mask = np.random.permutation(range(len(dataset.train_feat_indices)))
            dataset.train_feat_indices = dataset.train_feat_indices[random_mask]
            dataset.train_feat_values = dataset.train_feat_values[random_mask]
            dataset.train_labels = dataset.train_labels[random_mask]

    def __init_sampling(self):
        self.user_negative_pool = {}
        for u in self.dataset.train_user:
            self.user_negative_pool[u] = list(set(range(self.dataset.n_items)) - set(self.dataset.train_user[u]))

    def __neg_feat_dict(self):
        neg_indices_dict = dict()
        total_items_col = [-1]  # last col is item
        total_items_col.extend(self.dataset.item_feature_cols)  # if item_feature_cols is None ###########
        total_items_unique = np.unique(self.dataset.train_feat_indices[:, total_items_col], axis=0)
        total_items = total_items_unique[:, 0]
        total_items_feat_col = np.delete(total_items_unique, 0, axis=1)

    #    item_num_cols = [-1]
    #    item_num_cols.extend(np.arange(len(self.dataset.item_numerical_cols)) + len(self.dataset.user_numerical_cols))
    #    items_num_unique = np.unique(self.dataset.train_feat_values[:, item_num_cols], axis=0)
    #    total_items_value_col = np.delete(items_num_unique, 0, axis=1)

        for item, item_feat_col in zip(total_items, total_items_feat_col):
            neg_indices_dict[item] = item_feat_col.tolist()
    #    for item, item_value_col in zip(total_items, total_items_value_col):
    #        neg_values_dict[item] = item_value_col.tolist()
    #    print(neg_indices_dict)
        neg_values_dict = dict()
        for item in range(self.dataset.n_items):
            item_repr = item + self.dataset.user_offset + self.dataset.n_users
            item_values = []
            for num_col in self.dataset.item_numerical_cols:
                item_indices = np.where(self.dataset.train_feat_indices[:, -1] == item_repr)[0][0]
                item_values.append(self.dataset.train_feat_values[item_indices, num_col])
            neg_values_dict[item_repr] = item_values
    #    print(neg_values_dict[94833], neg_values_dict[103266])
    #    print(neg_values_dict)
        return neg_indices_dict, neg_values_dict

    def __call__(self, mode):
        if mode == "train":
            feat_indices = self.dataset.train_feat_indices
            feat_values = self.dataset.train_feat_values
            feat_labels = self.dataset.train_labels
        elif mode == "test":
            feat_indices = self.dataset.test_feat_indices
            feat_values = self.dataset.test_feat_values
            feat_labels = self.dataset.test_labels

        indices, values, labels = [], [], []
        for i, sample in enumerate(feat_indices):
            ss = sample.tolist()
            user = ss[-2] - self.dataset.user_offset
            indices.append(feat_indices[i])
            values.append(feat_values[i])
            labels.append(feat_labels[i])

            for _ in range(self.num_neg):
                item_neg = np.random.randint(0, self.dataset.n_items)
                while item_neg in self.dataset.train_user[user]:
                    item_neg = np.random.randint(0, self.dataset.n_items)

                item_neg += (self.dataset.user_offset + self.dataset.n_users)

            #    dt = self.dataset.train_feat_indices[self.dataset.train_feat_indices[:, -1] == item_neg][0]
                dt = self.neg_indices_dict[item_neg]
                for col, orig_col in enumerate(self.dataset.item_feature_cols):
                    ss[orig_col] = dt[col]
                ss[-1] = item_neg

                indices.append(ss)
                vv = feat_values[i]
                dv = self.neg_values_dict[item_neg]
                for col, orig_col in enumerate(self.dataset.item_numerical_cols):
                    vv[orig_col] = dv[col]
                values.append(vv)
                labels.append(0.0)
        random_mask = np.random.permutation(range(len(indices)))
        return np.array(indices)[random_mask], np.array(values)[random_mask], np.array(labels)[random_mask]

    def next_batch(self):
        if self.pre_sampling:
            end = min(len(self.dataset.train_indices_implicit), (self.i + 1) * self.batch_size)
            batch_feat_indices = self.dataset.train_indices_implicit[self.i * self.batch_size: end]
            batch_feat_values = self.dataset.train_values_implicit[self.i * self.batch_size: end]
            batch_labels = self.dataset.train_labels_implicit[self.i * self.batch_size: end]
            self.i += 1
            return batch_feat_indices, batch_feat_values, batch_labels
        else:
            batch_size = int(self.batch_size / (self.num_neg + 1))  # positive samples in one batch
            end = min(len(self.dataset.train_feat_indices), (self.i + 1) * batch_size)
            batch_feat_indices = self.dataset.train_feat_indices[self.i * batch_size: end]
            batch_feat_values = self.dataset.train_feat_values[self.i * batch_size: end]
            batch_feat_labels = self.dataset.train_labels[self.i * batch_size: end]

        #    end = min(len(self.dataset.train_feat_indices), (self.i + 1) * self.batch_size)  # bs = self.batch_size // 2
        #    batch_feat_indices = self.dataset.train_feat_indices[self.i * self.batch_size: end]  # train_indices_implicit
        #    batch_feat_values = self.dataset.train_feat_values[self.i * self.batch_size: end]  # train_values_implicit
        #    batch_feat_labels = self.dataset.train_labels[self.i * self.batch_size: end]  # train_labels_implicit

            indices, values, labels = [], [], []
            for i, sample in enumerate(batch_feat_indices):
            #    user = sample[-2] - self.dataset.user_offset
                ss = sample.tolist()
                user = ss[-2] - self.dataset.user_offset
                indices.append(batch_feat_indices[i])
                values.append(batch_feat_values[i])
                labels.append(batch_feat_labels[i])
                for _ in range(self.num_neg):
                    item_neg = np.random.randint(0, self.dataset.n_items)
            #        while item_neg in self.dataset.train_user[u]:
                    while item_neg in self.dataset.train_user[user]:
                        item_neg = np.random.randint(0, self.dataset.n_items)  # resample

                    item_neg += (self.dataset.user_offset + self.dataset.n_users)  # item offset
                #    dt = self.dataset.train_feat_indices[self.dataset.train_feat_indices[:, -1] == item_neg].tolist()[0]
                    dt = self.neg_indices_dict[item_neg]
                    for c, col in enumerate(self.dataset.item_feature_cols):
                        ss[col] = dt[c]
                    ss[-1] = item_neg

                    indices.append(ss)

                    vv = batch_feat_values[i]
                    dv = self.neg_values_dict[item_neg]
                    for col, orig_col in enumerate(self.dataset.item_numerical_cols):
                        vv[orig_col] = dv[col]
                    values.append(vv)

            #        values.append(batch_feat_values[i])
                    labels.append(0.0)

            self.i += 1
            random_mask = np.random.permutation(len(indices))
            return np.array(indices)[random_mask], np.array(values)[random_mask], np.array(labels)[random_mask]


class PairwiseSampling:
    def __init__(self, dataset, batch_size=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.i = 0

    def __call__(self, mode):
        if mode == "train":
            user_indices = self.dataset.train_user_indices
            item_indices = self.dataset.train_item_indices
            label_indices = self.dataset.train_labels
        elif mode == "test":
            user_indices = self.dataset.test_user_indices
            item_indices = self.dataset.test_item_indices
            label_indices = self.dataset.test_labels

        user, item, label = [], [], []
        for i, u in enumerate(user_indices):
            user.append(user_indices[i])
            item.append(item_indices[i])
            label.append(label_indices[i])

            item_neg = np.random.randint(0, self.dataset.n_items)
            while item_neg in self.dataset.train_user[u]:
                item_neg = np.random.randint(0, self.dataset.n_items)
            user.append(u)
            item.append(item_neg)
            label.append(0.0)
        return np.array(user), \
               np.array(item), \
               np.array(label)

    def next_mf(self, user_factors, item_factors, bootstrap=False):
        if bootstrap:
            random_i = np.random.choice(len(self.dataset.train_user_indices), 1, replace=True)
        #    random_i = np.random.randint(0, self.dataset.n_users)
            user, item_i = self.dataset.train_user_indices[random_i][0], \
                           self.dataset.train_item_indices[random_i][0]
            item_j = np.random.randint(0, self.dataset.n_items)  # self.dataset.n_items - 1
            while item_j in self.dataset.train_user[user]:
                item_j = np.random.randint(0, self.dataset.n_items)

            x_ui = np.dot(user_factors[user], item_factors[item_i])
            x_uj = np.dot(user_factors[user], item_factors[item_j])
            x_uij = x_ui - x_uj
            return user, item_i, item_j, x_uij

        elif self.batch_size == 1:
            user, item_i = self.dataset.train_user_indices[self.i], \
                           self.dataset.train_item_indices[self.i]
            item_j = np.random.randint(0, self.dataset.n_items)
            while item_j in self.dataset.train_user[user]:
                item_j = np.random.randint(0, self.dataset.n_items)

            x_ui = np.dot(user_factors[user], item_factors[item_i])
            x_uj = np.dot(user_factors[user], item_factors[item_j])
            x_uij = x_ui - x_uj
            self.i += 1
            return user, item_i, item_j, x_uij

        elif self.batch_size > 1:
            batch_item_j, batch_x_uij = [], []
            end = min(len(self.dataset.train_user_indices), (self.i + 1) * self.batch_size)
            batch_user = self.dataset.train_user_indices[self.i * self.batch_size: end]
            batch_item_i = self.dataset.train_item_indices[self.i * self.batch_size: end]
            for user, item_i in zip(batch_user, batch_item_i):
                item_j = np.random.randint(0, self.dataset.n_items)
                while item_j in self.dataset.train_user[user]:
                    item_j = np.random.randint(0, self.dataset.n_items)
                batch_item_j.append(item_j)

                x_ui = np.dot(user_factors[user], item_factors[item_i])
                x_uj = np.dot(user_factors[user], item_factors[item_j])
                x_uij = x_ui - x_uj
                batch_x_uij.append(x_uij)
            self.i += 1
            return batch_user, batch_item_i, np.array(batch_item_j), np.array(batch_x_uij)

        else:
            raise ValueError("either use bootstrap or batch size must be positive integer.")

    def next_mf_tf(self):
        batch_item_j, batch_x_uij = [], []
        end = min(len(self.dataset.train_user_indices), (self.i + 1) * self.batch_size)
        batch_user = self.dataset.train_user_indices[self.i * self.batch_size: end]
        batch_item_i = self.dataset.train_item_indices[self.i * self.batch_size: end]
        for user, item_i in zip(batch_user, batch_item_i):
            item_j = np.random.randint(0, self.dataset.n_items)
            while item_j in self.dataset.train_user[user]:
                item_j = np.random.randint(0, self.dataset.n_items)
            batch_item_j.append(item_j)
        self.i += 1
        return batch_user, batch_item_i, np.array(batch_item_j)

    def next_knn(self, sim_matrix, k=20):
        user, item_i = self.dataset.train_user_indices[self.i], \
                       self.dataset.train_item_indices[self.i]
        item_j = np.random.randint(0, self.dataset.n_items)
        while item_j in self.dataset.train_user[user]:
            item_j = np.random.randint(0, self.dataset.n_items)

        u_items = np.array(list(self.dataset.train_user[user]))
        item_i_neighbors_sim = sim_matrix[item_i, u_items]
        indices = np.argsort(item_i_neighbors_sim)[::-1][:k]
        i_k_neighbors = u_items[indices]
        x_ui = np.sum(item_i_neighbors_sim[indices])

        item_j_neighbors_sim = sim_matrix[item_j, u_items]
        indices = np.argsort(item_j_neighbors_sim)[::-1][:k]
        j_k_neighbors = u_items[indices]
        x_uj = np.sum(item_j_neighbors_sim[indices])

        x_uij = x_ui - x_uj
        self.i += 1
        return user, item_i, i_k_neighbors, item_j, j_k_neighbors, x_uij













