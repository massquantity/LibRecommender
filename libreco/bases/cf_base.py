import os
import random
from functools import partial
from itertools import islice, takewhile
from operator import itemgetter

import numpy as np
from scipy.sparse import issparse, save_npz as save_sparse, load_npz as load_sparse

from ..bases import Base
from ..evaluation import print_metrics
from ..recommendation import popular_recommendations
from ..recommendation.ranking import filter_items
from ..utils.misc import time_block, colorize
from ..utils.save_load import load_params, save_params
from ..utils.similarities import cosine_sim, jaccard_sim, pearson_sim
from ..utils.validate import convert_id, check_unknown


class CfBase(Base):
    def __init__(
        self,
        task,
        data_info,
        cf_type=None,
        sim_type="cosine",
        k_sim=20,
        store_top_k=True,
        block_size=None,
        num_threads=1,
        min_common=1,
        mode="invert",
        seed=42,
        k=10,
        eval_batch_size=8192,
        eval_user_num=None,
        lower_upper_bound=None,
    ):
        super().__init__(task, data_info, lower_upper_bound)

        assert cf_type in ("user_cf", "item_cf")
        self.all_args = locals()
        self.cf_type = cf_type
        self.k_sim = k_sim
        self.sim_type = sim_type
        self.store_top_k = store_top_k
        self.block_size = block_size
        self.num_threads = num_threads
        self.min_common = min_common
        self.mode = mode
        self.seed = seed
        self.k = k
        self.eval_batch_size = eval_batch_size
        self.eval_user_num = eval_user_num
        # sparse matrix, user as row and item as column
        self.user_interaction = None
        # sparse matrix, item as row and user as column
        self.item_interaction = None
        # sparse similarity matrix
        self.sim_matrix = None
        self.topk_sim = None
        self.print_count = 0
        self._caution_sim_type()

    def _caution_sim_type(self):
        if self.task == "ranking" and self.sim_type == "pearson":
            caution_str = (
                f"Warning: {self.sim_type} is not suitable " f"for implicit data"
            )
            print(f"{colorize(caution_str, 'red')}")
        if self.task == "rating" and self.sim_type == "jaccard":
            caution_str = (
                f"Warning: {self.sim_type} is not suitable " f"for explicit data"
            )
            print(f"{colorize(caution_str, 'red')}")

    def fit(self, train_data, verbose=1, eval_data=None, metrics=None):
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
                raise ValueError(
                    "sim_type must be one of (`cosine`, `pearson`, `jaccard`)"
                )
            sim_func = partial(
                sim_func,
                block_size=self.block_size,
                num_threads=self.num_threads,
                min_common=self.min_common,
                mode=self.mode,
            )

            if self.cf_type == "user_cf":
                self.sim_matrix = sim_func(
                    self.user_interaction,
                    self.item_interaction,
                    self.n_users,
                    self.n_items,
                )
            else:
                self.sim_matrix = sim_func(
                    self.item_interaction,
                    self.user_interaction,
                    self.n_items,
                    self.n_users,
                )

        assert self.sim_matrix.has_sorted_indices
        if issparse(self.sim_matrix):
            n_elements = self.sim_matrix.getnnz()
            density_ratio = 100 * n_elements / (self.n_users * self.n_users)
            print(
                f"sim_matrix, shape: {self.sim_matrix.shape}, "
                f"num_elements: {n_elements}, "
                f"density: {density_ratio:5.4f} %"
            )
        if self.store_top_k:
            self.compute_top_k()

        if verbose > 1:
            print_metrics(
                model=self,
                eval_data=eval_data,
                metrics=metrics,
                eval_batch_size=self.eval_batch_size,
                k=self.k,
                sample_user_num=self.eval_user_num,
                seed=self.seed,
            )
            print("=" * 30)

    def compute_top_k(self):
        ...

    def predict(self, user, item, **kwargs):
        ...

    def recommend_user(self, user, n_rec, **kwargs):
        ...

    def pre_predict_check(self, user, item, inner_id, cold_start):
        user_arr, item_arr = convert_id(self, user, item, inner_id)
        unknown_num, _, user_arr, item_arr = check_unknown(self, user_arr, item_arr)
        if unknown_num > 0 and cold_start != "popular":
            raise ValueError(f"{self.model_name} only supports popular strategy")
        return user_arr, item_arr

    def compute_pred(self, user, item, common_size, common_sims, common_labels):
        if common_size == 0 or np.all(common_sims <= 0.0):
            self.print_count += 1
            no_str = (
                f"No common interaction or similar neighbor "
                f"for user {user} and item {item}, "
                f"proceed with default prediction"
            )
            if self.print_count < 7:
                print(f"{colorize(no_str, 'red')}")
            return self.default_prediction
        else:
            k_neighbor_labels, k_neighbor_sims = zip(
                *islice(
                    takewhile(
                        lambda x: x[1] > 0,
                        sorted(
                            zip(common_labels, common_sims),
                            key=itemgetter(1),
                            reverse=True,
                        ),
                    ),
                    self.k_sim,
                )
            )

            if self.task == "rating":
                sims_distribution = k_neighbor_sims / np.sum(k_neighbor_sims)
                weighted_pred = np.average(k_neighbor_labels, weights=sims_distribution)
                return np.clip(weighted_pred, self.lower_bound, self.upper_bound)
            elif self.task == "ranking":
                return np.mean(k_neighbor_sims)

    def rank_recommendations(
        self,
        user,
        ids,
        preds,
        n_rec,
        consumed,
        inner_id,
        filter_consumed,
        random_rec,
    ):
        if filter_consumed:
            ids, preds = filter_items(ids, preds, consumed)
        if len(ids) == 0:
            self.print_count += 1
            no_str = (
                f"no suitable recommendation for user {user}, "
                f"return default recommendation"
            )
            if self.print_count < 11:
                print(f"{colorize(no_str, 'red')}")
            return popular_recommendations(self.data_info, inner_id, n_rec)

        if random_rec and len(ids) > n_rec:
            ids = random.sample(list(ids), k=n_rec)
        else:
            indices = np.argsort(preds)[::-1]
            ids = ids[indices][:n_rec]
        if not inner_id:
            ids = [self.data_info.id2item[i] for i in ids]
        return np.asarray(ids)

    def save(self, path, model_name, **kwargs):
        if not os.path.isdir(path):
            print(f"file folder {path} doesn't exists, creating a new one...")
            os.makedirs(path)
        save_params(self, path, model_name)
        model_path = os.path.join(path, model_name)
        save_sparse(f"{model_path}_sim_matrix", self.sim_matrix)
        save_sparse(f"{model_path}_user_inter", self.user_interaction)
        save_sparse(f"{model_path}_item_inter", self.item_interaction)

    @classmethod
    def load(cls, path, model_name, data_info, **kwargs):
        hparams = load_params(cls, path, data_info, model_name)
        model = cls(**hparams)
        model_path = os.path.join(path, model_name)
        model.sim_matrix = load_sparse(f"{model_path}_sim_matrix.npz")
        model.user_interaction = load_sparse(f"{model_path}_user_inter.npz")
        model.item_interaction = load_sparse(f"{model_path}_item_inter.npz")
        return model

    def rebuild_model(self, path, model_name, **kwargs):
        raise NotImplementedError(f"{self.model_name} doesn't support model retraining")
