import os
import random
from operator import itemgetter
from itertools import islice, takewhile
from collections import defaultdict
import numpy as np
from scipy.sparse import (
    issparse,
    save_npz as save_sparse,
    load_npz as load_sparse
)
from tqdm import tqdm
from .base import Base
from ..utils.similarities import cosine_sim, pearson_sim, jaccard_sim
from ..utils.misc import time_block, colorize
from ..evaluation.evaluate import EvalMixin


class UserCF(Base, EvalMixin):
    def __init__(
            self,
            task,
            data_info,
            sim_type="cosine",
            k=20,
            lower_upper_bound=None
    ):
        Base.__init__(self, task, data_info, lower_upper_bound)
        EvalMixin.__init__(self, task, data_info)

        self.task = task
        self.data_info = data_info
        self.k = k
        self.n_users = data_info.n_users
        self.n_items = data_info.n_items
        self.sim_type = sim_type
        self.user_consumed = data_info.user_consumed
        # sparse matrix, user as row and item as column
        self.user_interaction = None
        # sparse matrix, item as row and user as column
        self.item_interaction = None
        # sparse similarity matrix
        self.sim_matrix = None
        self.topk_sim = None
        self.print_count = 0
        self._caution_sim_type()
        self.all_args = locals()

    def fit(self, train_data, block_size=None, num_threads=1, min_common=1,
            mode="invert", verbose=1, eval_data=None, metrics=None,
            store_top_k=True):
        self.show_start_time()
        self.user_interaction = train_data.sparse_interaction
        self.item_interaction = self.user_interaction.T.tocsr()

        with time_block("sim_matrix", verbose=1):
            if self.sim_type == "cosine":
                sim_func = cosine_sim
            elif self.sim_type == "pearson":
                sim_func = pearson_sim
            elif self.sim_type == "jaccard":
                sim_func = jaccard_sim
            else:
                raise ValueError("sim_type must be one of "
                                 "('cosine', 'pearson', 'jaccard')")

            self.sim_matrix = sim_func(
                self.user_interaction, self.item_interaction, self.n_users,
                self.n_items, block_size, num_threads, min_common, mode)

        assert self.sim_matrix.has_sorted_indices
        if issparse(self.sim_matrix):
            n_elements = self.sim_matrix.getnnz()
            sparsity_ratio = 100*n_elements / (self.n_users*self.n_users)
            print(f"sim_matrix, shape: {self.sim_matrix.shape}, "
                  f"num_elements: {n_elements}, "
                  f"sparsity: {sparsity_ratio:5.4f} %")
        if store_top_k:
            self.compute_top_k()

        if verbose > 1:
            self.print_metrics(eval_data=eval_data, metrics=metrics)
            print("=" * 30)

    def predict(self, user, item, cold="popular", inner_id=False):
        user, item = self.convert_id(user, item, inner_id)
        unknown_num, unknown_index, user, item = self._check_unknown(user, item)
        if unknown_num > 0 and cold != "popular":
            raise ValueError("UserCF only supports popular strategy")

        preds = []
        sim_matrix = self.sim_matrix
        interaction = self.item_interaction
        for u, i in zip(user, item):
            if u == self.n_users or i == self.n_items:
                preds.append(self.default_prediction)
                continue

            user_slice = slice(sim_matrix.indptr[u], sim_matrix.indptr[u+1])
            sim_users = sim_matrix.indices[user_slice]
            sim_values = sim_matrix.data[user_slice]

            item_slice = slice(interaction.indptr[i], interaction.indptr[i+1])
            item_interacted_u = interaction.indices[item_slice]
            item_interacted_values = interaction.data[item_slice]
            common_users, indices_in_u, indices_in_i = np.intersect1d(
                sim_users, item_interacted_u,
                assume_unique=True, return_indices=True)

            common_sims = sim_values[indices_in_u]
            common_labels = item_interacted_values[indices_in_i]
            if common_users.size == 0 or np.all(common_sims <= 0.0):
                self.print_count += 1
                no_str = (f"No common interaction or similar neighbor "
                          f"for user {u} and item {i}, "
                          f"proceed with default prediction")
                if self.print_count < 7:
                    print(f"{colorize(no_str, 'red')}")
                preds.append(self.default_prediction)
            else:
                k_neighbor_labels, k_neighbor_sims = zip(
                    *islice(
                        takewhile(
                            lambda x: x[1] > 0,
                            sorted(zip(common_labels, common_sims),
                                   key=itemgetter(1), reverse=True),
                        ),
                        self.k,
                    )
                )

                if self.task == "rating":
                    sims_distribution = (
                            k_neighbor_sims / np.sum(k_neighbor_sims)
                    )
                    weighted_pred = np.average(
                        k_neighbor_labels, weights=sims_distribution
                    )
                    preds.append(
                        np.clip(weighted_pred, self.lower_bound,
                                self.upper_bound)
                    )
                elif self.task == "ranking":
                    preds.append(np.mean(k_neighbor_sims))

        return preds[0] if len(user) == 1 else preds

    def recommend_user(self, user, n_rec, random_rec=False,
                       cold_start="popular", inner_id=False):
        user_id = self._check_unknown_user(user)
        if user_id is None:
            if cold_start == "popular":
                return self.data_info.popular_items[:n_rec]
            elif cold_start != "popular":
                raise ValueError("UserCF only supports popular strategy")
            else:
                raise ValueError(user)

        user_slice = slice(self.sim_matrix.indptr[user],
                           self.sim_matrix.indptr[user+1])
        sim_users = self.sim_matrix.indices[user_slice]
        sim_values = self.sim_matrix.data[user_slice]

        if sim_users.size == 0 or np.all(sim_values <= 0):
            self.print_count += 1
            no_str = (f"no similar neighbor for user {user}, "
                      f"return default recommendation")
            if self.print_count < 11:
                print(f"{colorize(no_str, 'red')}")
            return self.data_info.popular_items[:n_rec]

        if self.topk_sim is not None:
            k_nbs_and_sims = self.topk_sim[user]
        else:
            k_nbs_and_sims = islice(
                sorted(
                    zip(sim_users, sim_values),
                    key=itemgetter(1),
                    reverse=True
                ),
                self.k
            )
        u_consumed = set(self.user_consumed[user])

        all_item_indices = self.user_interaction.indices
        all_item_indptr = self.user_interaction.indptr
        all_item_values = self.user_interaction.data

        result = defaultdict(lambda: [0.0, 0])  # [sim, count]
        for n, n_sim in k_nbs_and_sims:
            item_slices = slice(all_item_indptr[n], all_item_indptr[n+1])
            n_interacted_items = all_item_indices[item_slices]
            n_interacted_values = all_item_values[item_slices]
            for i, v in zip(n_interacted_items, n_interacted_values):
                if i in u_consumed:
                    continue
                result[i][0] += n_sim * v
                result[i][1] += n_sim

        rank_items = [(k, round(v[0] / v[1], 4)) for k, v in result.items()]
        rank_items.sort(key=lambda x: -x[1])
        if random_rec:
            if len(rank_items) < n_rec:
                item_candidates = rank_items
            else:
                item_candidates = random.sample(rank_items, k=n_rec)
            return item_candidates
        else:
            return rank_items[:n_rec]

    def _caution_sim_type(self):
        caution_str = (f"Warning: {self.sim_type} is not suitable "
                       f"for implicit data")
        caution_str2 = (f"Warning: {self.sim_type} is not suitable "
                        f"for explicit data")
        if self.task == "ranking" and self.sim_type == "pearson":
            print(f"{colorize(caution_str, 'red')}")
        if self.task == "rating" and self.sim_type == "jaccard":
            print(f"{colorize(caution_str2, 'red')}")

    def compute_top_k(self):
        top_k = dict()
        for u in tqdm(range(self.n_users), desc="top_k"):
            user_slice = slice(self.sim_matrix.indptr[u],
                               self.sim_matrix.indptr[u+1])
            sim_users = self.sim_matrix.indices[user_slice].tolist()
            sim_values = self.sim_matrix.data[user_slice].tolist()

            top_k[u] = sorted(
                zip(sim_users, sim_values),
                key=itemgetter(1),
                reverse=True
            )[:self.k]
        self.topk_sim = top_k

    def save(self, path, model_name, **kwargs):
        if not os.path.isdir(path):
            print(f"file folder {path} doesn't exists, creating a new one...")
            os.makedirs(path)
        self.save_params(path)
        model_path = os.path.join(path, model_name)
        save_sparse(f"{model_path}_sim_matrix", self.sim_matrix)
        save_sparse(f"{model_path}_user_inter", self.user_interaction)
        save_sparse(f"{model_path}_item_inter", self.item_interaction)

    @classmethod
    def load(cls, path, model_name, data_info, **kwargs):
        hparams = cls.load_params(path, data_info)
        model = cls(**hparams)
        model_path = os.path.join(path, model_name)
        model.sim_matrix = load_sparse(f"{model_path}_sim_matrix.npz")
        model.user_interaction = load_sparse(f"{model_path}_user_inter.npz")
        model.item_interaction = load_sparse(f"{model_path}_item_inter.npz")
        return model

    def rebuild_graph(self, path, model_name, full_assign=False):
        raise NotImplementedError("UserCF doesn't support model retraining")
