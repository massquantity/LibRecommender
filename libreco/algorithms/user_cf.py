"""Implementation of UserCF."""
from collections import defaultdict

import numpy as np

from ..bases import CfBase
from ..recommendation import popular_recommendations


class UserCF(CfBase):
    """*User Collaborative Filtering* algorithm. See :ref:`UserCF / ItemCF` for more details.

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
        Whether to store top k similar users after training.
    block_size : int or None, default: None
        Block size for computing similarity matrix. Large block size makes computation
        faster, but may cause memory issue.
    num_threads : int, default: 1
        Number of threads to use.
    min_common : int, default: 1
        Number of minimum common users to consider when computing similarities.
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
        prediction : float or numpy.ndarray
            Predicted scores for each user-item pair.
        """
        user_arr, item_arr = self.pre_predict_check(user, item, inner_id, cold_start)
        preds = []
        sim_matrix = self.sim_matrix
        interaction = self.item_interaction
        for u, i in zip(user_arr, item_arr):
            if u == self.n_users or i == self.n_items:
                preds.append(self.default_pred)
                continue
            user_slice = slice(sim_matrix.indptr[u], sim_matrix.indptr[u + 1])
            sim_users = sim_matrix.indices[user_slice][: self.k_sim]
            sim_values = sim_matrix.data[user_slice][: self.k_sim]

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

    def recommend_one(self, user_id, n_rec, filter_consumed, random_rec):
        if self.topk_sim is not None:
            user_sim_topk = self.topk_sim[user_id]
        else:
            user_sim_topk = self.get_top_k_sims(user_id)
        if not user_sim_topk:
            return popular_recommendations(self.data_info, inner_id=True, n_rec=n_rec)

        all_item_indices = self.user_interaction.indices
        all_item_indptr = self.user_interaction.indptr
        all_item_values = self.user_interaction.data
        item_scores = defaultdict(lambda: 0.0)
        for v, u_v_sim in user_sim_topk:
            item_slices = slice(all_item_indptr[v], all_item_indptr[v + 1])
            v_interacted_items = all_item_indices[item_slices]
            v_interacted_values = all_item_values[item_slices]
            for i, v_i_score in zip(v_interacted_items, v_interacted_values):
                item_scores[i] += u_v_sim * v_i_score

        item_scores = list(zip(*item_scores.items()))
        ids = np.array(item_scores[0])
        preds = np.array(item_scores[1])
        return self.rank_recommendations(
            user_id,
            ids,
            preds,
            n_rec,
            self.user_consumed[user_id],
            filter_consumed,
            random_rec,
        )

    def rebuild_model(self, path, model_name, **kwargs):
        raise NotImplementedError("`UserCF` doesn't support model retraining")
