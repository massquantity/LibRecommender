import random
from operator import itemgetter
from itertools import islice, takewhile
from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix, issparse
from .base import Base
from ..utils.similarities import cosine_sim, pearson_sim, jaccard_sim
from ..utils.timing import time_block
from ..evaluate.evaluate import EvalMixin


class UserCF(Base, EvalMixin):
    def __init__(self, data_info, task="rating", sim_type="pearson", k=50,
                 lower_upper_bound=None):
        self.k = k
        self.task = task
        self.default_prediction = data_info.global_mean if task == "rating" else 0.0
        self.n_users = data_info.n_users
        self.n_items = data_info.n_items
        self.sim_type = sim_type
        self.interaction_user = None   # matrix that user as row and item as column
        self.interaction_item = None
        self.sim_matrix = None
        Base.__init__(self, data_info, task, lower_upper_bound)
        EvalMixin.__init__(self, task)

    def fit(self, train_data, block_size=None, num_threads=1, min_support=1,
            mode="invert", verbose=1, eval_data=None, metrics=None):
        self.interaction_user = train_data.sparse_interaction
        self.interaction_item = self.interaction_user.tocsc()

        with time_block("sim_matrix", verbose=1):
            if self.sim_type == "cosine":
                self.sim_matrix = cosine_sim(self.interaction_user, self.interaction_item,
                                             self.n_users, self.n_items, block_size,
                                             num_threads, min_support, mode)
            elif self.sim_type == "pearson":
                self.sim_matrix = pearson_sim(self.interaction_user, self.interaction_item,
                                              self.n_users, self.n_items, block_size,
                                              num_threads, min_support, mode)
            elif self.sim_type == "jaccard":
                self.sim_matrix = jaccard_sim(self.interaction_user, self.interaction_item,
                                              self.n_users, self.n_items, block_size,
                                              num_threads, min_support, mode)

        assert self.sim_matrix.has_sorted_indices
        if issparse(self.sim_matrix):
            n_elements = self.sim_matrix.getnnz()
            sparsity_ratio = 100 * n_elements / (self.n_users*self.n_users)
            print(f"sim_matrix, shape: {self.sim_matrix.shape}, "
                  f"num_elements: {n_elements}, "
                  f"data sparsity: {sparsity_ratio:5.2f} %")

        if verbose > 1:
            self.print_metrics(eval_data=eval_data, metrics=metrics)

    def predict(self, user, item):
        user = [user] if isinstance(user, int) else user
        item = [item] if isinstance(item, int) else item
        sim_matrix = self.sim_matrix
        interaction = self.interaction_item
        preds = []
        for u, i in zip(user, item):
            if u >= self.n_users or i >= self.n_items:  # TODO: cold user or item
                preds.append(self.default_prediction)
                continue

            user_slice = slice(sim_matrix.indptr[u], sim_matrix.indptr[u+1])
            sim_users = sim_matrix.indices[user_slice]
            sim_values = sim_matrix.data[user_slice]
            item_slice = slice(interaction.indptr[i], interaction.indptr[i+1])
            item_interacted_u = interaction.indices[item_slice]
            item_interacted_values = interaction.data[item_slice]
            common_users, indices_in_u, indices_in_i = np.intersect1d(sim_users,
                                                                      item_interacted_u,
                                                                      assume_unique=True,
                                                                      return_indices=True)

            common_sims = sim_values[indices_in_u]
            common_labels = item_interacted_values[indices_in_i]
            if common_users.size == 0 or np.all(common_sims <= 0.0):
                print(f"no common interaction or no similar neighbor for user {u} and item {i}, "
                      f"got default prediction")
                preds.append(self.default_prediction)

            else:
                k_neighbor_labels, k_neighbor_sims = zip(*islice(
                    takewhile(lambda x: x[1] > 0,
                              sorted(zip(common_labels, common_sims),
                                     key=itemgetter(1),
                                     reverse=True)
                              ),
                    self.k))

                if self.task == "rating":
                    k_neighbor_sims = k_neighbor_sims / np.sum(k_neighbor_sims)
                    weighted_pred = np.average(k_neighbor_labels, weights=k_neighbor_sims)
                    preds.append(np.clip(weighted_pred, self.lower_bound, self.upper_bound))
                elif self.task == "ranking":
                    preds.append(np.mean(k_neighbor_sims))

        return preds[0] if len(user) == 1 else preds

    def recommend_user(self, user, n_rec, random_rec=False):
        user_slice = slice(self.sim_matrix.indptr[user], self.sim_matrix.indptr[user+1])
        sim_users = self.sim_matrix.indices[user_slice]
        sim_values = self.sim_matrix.data[user_slice]
        all_item_indices = self.interaction_user.indices
        all_item_indptr = self.interaction_user.indptr
        all_item_values = self.interaction_user.data
        if sim_users.size == 0 or np.all(sim_values <= 0):     # TODO: return popular items
            print(f"no similar neighbor for user {user}, return default recommendation")
            return -1

        result = defaultdict(lambda: [0, 0])   # [sim, count]
        u_slices = slice(all_item_indptr[user], all_item_indptr[user+1])
        u_consumed = set(all_item_indices[u_slices])
        k_neighbors = islice(
            sorted(zip(sim_users, sim_values), key=itemgetter(1), reverse=True),
            self.k)

        for n, n_sim in k_neighbors:
            item_slices = slice(all_item_indptr[n], all_item_indptr[n+1])
            n_interacted_items = all_item_indices[item_slices]
            n_interacted_values = all_item_values[item_slices]
            for i, v in zip(n_interacted_items, n_interacted_values):
                if i in u_consumed:
                    continue
                result[i][0] += n_sim * v
                result[i][1] += n_sim

        rank_items = [(k, round(v[0] / v[1], 2)) for k, v in result.items()]
        rank_items.sort(key=lambda x: -x[1])
        if random_rec:
            if len(rank_items) < n_rec:
                item_candidates = rank_items
            else:
                item_candidates = random.sample(rank_items, k=n_rec)
            return item_candidates
        else:
            return rank_items[:n_rec]

