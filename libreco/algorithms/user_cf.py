from collections import defaultdict
from itertools import islice
from operator import itemgetter

import numpy as np
from tqdm import tqdm

from ..bases import CfBase
from ..recommendation import popular_recommendations
from ..utils.misc import colorize


class UserCF(CfBase):
    def __init__(
        self,
        task,
        data_info,
        sim_type="cosine",
        k_sim=20,
        store_top_k=True,
        block_size=None,
        num_threads=1,
        min_common=1,
        mode="invert",
        seed=42,
        lower_upper_bound=None,
    ):
        super().__init__(
            task,
            data_info,
            "user_cf",
            sim_type,
            k_sim,
            store_top_k,
            block_size,
            num_threads,
            min_common,
            mode,
            seed,
            lower_upper_bound,
        )
        self.all_args = locals()

    def predict(self, user, item, cold_start="popular", inner_id=False):
        user_arr, item_arr = self.pre_predict_check(user, item, inner_id, cold_start)
        preds = []
        sim_matrix = self.sim_matrix
        interaction = self.item_interaction
        for u, i in zip(user_arr, item_arr):
            if u == self.n_users or i == self.n_items:
                preds.append(self.default_pred)
                continue
            user_slice = slice(sim_matrix.indptr[u], sim_matrix.indptr[u + 1])
            sim_users = sim_matrix.indices[user_slice]
            sim_values = sim_matrix.data[user_slice]

            item_slice = slice(interaction.indptr[i], interaction.indptr[i + 1])
            item_interacted_u = interaction.indices[item_slice]
            item_interacted_values = interaction.data[item_slice]
            common_users, indices_in_u, indices_in_i = np.intersect1d(
                sim_users, item_interacted_u, assume_unique=True, return_indices=True
            )

            common_sims = sim_values[indices_in_u]
            common_labels = item_interacted_values[indices_in_i]
            pred = self.compute_pred(
                u, i, common_users.size, common_sims, common_labels
            )
            preds.append(pred)
        return preds[0] if len(user_arr) == 1 else preds

    # all the items returned by this function will be inner_ids
    def recommend_one(self, user_id, n_rec, filter_consumed, random_rec):
        user_slice = slice(
            self.sim_matrix.indptr[user_id], self.sim_matrix.indptr[user_id + 1]
        )
        sim_users = self.sim_matrix.indices[user_slice]
        sim_values = self.sim_matrix.data[user_slice]
        if sim_users.size == 0 or np.all(sim_values <= 0):
            self.print_count += 1
            no_str = (
                f"no similar neighbor for user {user_id}, "
                f"return default recommendation"
            )
            if self.print_count < 7:
                print(f"{colorize(no_str, 'red')}")
            return popular_recommendations(self.data_info, inner_id=True, n_rec=n_rec)

        all_item_indices = self.user_interaction.indices
        all_item_indptr = self.user_interaction.indptr
        all_item_values = self.user_interaction.data
        if self.topk_sim is not None:
            k_nbs_and_sims = self.topk_sim[user_id]
        else:
            k_nbs_and_sims = islice(
                sorted(zip(sim_users, sim_values), key=itemgetter(1), reverse=True),
                self.k_sim,
            )
        item_sims = defaultdict(lambda: 0.0)
        item_scores = defaultdict(lambda: 0.0)
        for v, u_v_sim in k_nbs_and_sims:
            item_slices = slice(all_item_indptr[v], all_item_indptr[v + 1])
            v_interacted_items = all_item_indices[item_slices]
            v_interacted_values = all_item_values[item_slices]
            for i, v_i_score in zip(v_interacted_items, v_interacted_values):
                item_sims[i] += u_v_sim
                item_scores[i] += u_v_sim * v_i_score

        ids = np.array(list(item_sims))
        preds = np.array([item_scores[i] / item_sims[i] for i in ids])
        return self.rank_recommendations(
            user_id,
            ids,
            preds,
            n_rec,
            self.user_consumed[user_id],
            filter_consumed,
            random_rec,
        )

    def compute_top_k(self):
        top_k = dict()
        for u in tqdm(range(self.n_users), desc="top_k"):
            user_slice = slice(self.sim_matrix.indptr[u], self.sim_matrix.indptr[u + 1])
            sim_users = self.sim_matrix.indices[user_slice].tolist()
            sim_values = self.sim_matrix.data[user_slice].tolist()

            top_k[u] = sorted(
                zip(sim_users, sim_values), key=itemgetter(1), reverse=True
            )[: self.k_sim]
        self.topk_sim = top_k

    def rebuild_model(self, path, model_name, **kwargs):
        raise NotImplementedError(f"{self.model_name} doesn't support model retraining")
