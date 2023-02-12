"""Implementation of ItemCF."""
from collections import defaultdict
from operator import itemgetter

import numpy as np
from tqdm import tqdm

from ..bases import CfBase


class ItemCF(CfBase):
    """*Item Collaborative Filtering* algorithm.

    Parameters
    ----------
    task : {'rating', 'ranking'}
        Recommendation task. See :ref:`Task`.
    data_info : :class:`~libreco.data.DataInfo` object
        Object that contains useful information for training and inference.
    sim_type : {'cosine', 'pearson', 'jaccard'}, default: 'cosine'
        Types for computing similarities.
    k_sim : int, default: 20
        Number of similar items to use.
    store_top_k : bool, default: True
        Whether to store top k similar items after training.
    block_size : int or None, default: None
        Block size for computing similarity matrix. Large block size makes computation
        faster, but may cause memory issue.
    num_threads : int, default: 1
        Number of threads to use.
    min_common : int, default: 1
        Number of minimum common items to consider when computing similarities.
    mode : {'forward', 'invert'}, default: 'invert'
        Whether to use forward index or invert index.
    seed : int, default: 42
        Random seed.
    lower_upper_bound : tuple or None, default: None
        Lower and upper score bound for `rating` task.
    """

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
            "item_cf",
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
        """Make prediction(s) on given user(s) and item(s).

        Parameters
        ----------
        user : int or str or array_like
            User id or batch of user ids.
        item : int or str or array_like
            Item id or batch of item ids.
        cold_start : {'popular'}, default: 'popular'
            Cold start strategy, ItemCF can only use 'popular' strategy.
        inner_id : bool, default: False
            Whether to use inner_id defined in `libreco`. For library users inner_id
            may never be used.

        Returns
        -------
        prediction : float or array_like
            Predicted scores for each user-item pair.
        """
        user_arr, item_arr = self.pre_predict_check(user, item, inner_id, cold_start)
        preds = []
        sim_matrix = self.sim_matrix
        interaction = self.user_interaction
        for u, i in zip(user_arr, item_arr):
            if u == self.n_users or i == self.n_items:
                preds.append(self.default_pred)
                continue
            item_slice = slice(sim_matrix.indptr[i], sim_matrix.indptr[i + 1])
            sim_items = sim_matrix.indices[item_slice]
            sim_values = sim_matrix.data[item_slice]

            user_slice = slice(interaction.indptr[u], interaction.indptr[u + 1])
            user_interacted_i = interaction.indices[user_slice]
            user_interacted_values = interaction.data[user_slice]
            common_items, indices_in_i, indices_in_u = np.intersect1d(
                sim_items, user_interacted_i, assume_unique=True, return_indices=True
            )

            common_sims = sim_values[indices_in_i]
            common_labels = user_interacted_values[indices_in_u]
            pred = self.compute_pred(
                u, i, common_items.size, common_sims, common_labels
            )
            preds.append(pred)
        return preds[0] if len(user_arr) == 1 else preds

    def recommend_one(self, user_id, n_rec, filter_consumed, random_rec):
        user_slice = slice(
            self.user_interaction.indptr[user_id],
            self.user_interaction.indptr[user_id + 1],
        )
        user_interacted_i = self.user_interaction.indices[user_slice]
        user_interacted_labels = self.user_interaction.data[user_slice]

        result = defaultdict(lambda: 0.0)
        for i, i_label in zip(user_interacted_i, user_interacted_labels):
            if self.topk_sim is not None:
                item_sim_topk = self.topk_sim[i]
            else:
                item_slice = slice(
                    self.sim_matrix.indptr[i], self.sim_matrix.indptr[i + 1]
                )
                sim_items = self.sim_matrix.indices[item_slice]
                sim_values = self.sim_matrix.data[item_slice]
                item_sim_topk = sorted(
                    zip(sim_items, sim_values), key=itemgetter(1), reverse=True
                )[: self.k_sim]

            for j, sim in item_sim_topk:
                result[j] += sim * i_label

        result = list(zip(*result.items()))
        ids = np.array(result[0])
        preds = np.array(result[1])
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
        for i in tqdm(range(self.n_items), desc="top_k"):
            item_slice = slice(self.sim_matrix.indptr[i], self.sim_matrix.indptr[i + 1])
            sim_items = self.sim_matrix.indices[item_slice].tolist()
            sim_values = self.sim_matrix.data[item_slice].tolist()
            top_k[i] = sorted(
                zip(sim_items, sim_values), key=itemgetter(1), reverse=True
            )[: self.k_sim]
        self.topk_sim = top_k

    def rebuild_model(self, path, model_name, **kwargs):
        raise NotImplementedError("`ItemCF` doesn't support model retraining")
