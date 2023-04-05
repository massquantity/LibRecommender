"""CF model base class."""
import abc
import os
import random
from functools import partial
from itertools import islice, takewhile
from operator import itemgetter

import numpy as np
from scipy.sparse import issparse
from scipy.sparse import load_npz as load_sparse
from scipy.sparse import save_npz as save_sparse

from .base import Base
from ..evaluation import print_metrics
from ..prediction.preprocess import convert_id
from ..recommendation import construct_rec, popular_recommendations
from ..recommendation.ranking import filter_items
from ..utils.misc import colorize, time_block
from ..utils.save_load import load_params, save_params
from ..utils.similarities import cosine_sim, jaccard_sim, pearson_sim
from ..utils.validate import check_fitting, check_unknown, check_unknown_user


class CfBase(Base):
    """Base class for CF models.

    Parameters
    ----------
    task : {'rating', 'ranking'}
        Recommendation task. See :ref:`Task`.
    data_info : :class:`~libreco.data.DataInfo` object
        Object that contains useful information for training and inference.
    cf_type : {'user_cf', 'item_cf'}
        Specific CF type.
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

    See Also
    --------
    ~libreco.algorithms.UserCF
    ~libreco.algorithms.ItemCF
    """

    def __init__(
        self,
        task,
        data_info,
        cf_type,
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
            caution_str = "Warning: pearson is not suitable for implicit data"
            print(f"{colorize(caution_str, 'red')}")
        if self.task == "rating" and self.sim_type == "jaccard":
            caution_str = "Warning: jaccard is not suitable for explicit data"
            print(f"{colorize(caution_str, 'red')}")

    def fit(
        self,
        train_data,
        neg_sampling,
        verbose=1,
        eval_data=None,
        metrics=None,
        k=10,
        eval_batch_size=8192,
        eval_user_num=None,
    ):
        """Fit CF model on the training data.

        Parameters
        ----------
        train_data : :class:`~libreco.data.TransformedSet` object
            Data object used for training.
        neg_sampling : bool
            Whether to perform negative sampling for evaluating data.

            .. versionadded:: 1.1.0

        verbose : int, default: 1
            Print verbosity. If `eval_data` is provided, setting it to higher than 1
            will print evaluation metrics during training.
        eval_data : :class:`~libreco.data.TransformedSet` object, default: None
            Data object used for evaluating.
        metrics : list or None, default: None
            List of metrics for evaluating.
        k : int, default: 10
            Parameter of metrics, e.g. recall at k, ndcg at k
        eval_batch_size : int, default: 8192
            Batch size for evaluating.
        eval_user_num : int or None, default: None
            Number of users for evaluating. Setting it to a positive number will sample
            users randomly from eval data.
        """
        check_fitting(self, train_data, eval_data, neg_sampling, k)
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
                neg_sampling=neg_sampling,
                eval_data=eval_data,
                metrics=metrics,
                eval_batch_size=eval_batch_size,
                k=k,
                sample_user_num=eval_user_num,
                seed=self.seed,
            )
            print("=" * 30)

    @abc.abstractmethod
    def compute_top_k(self):
        ...

    # all the items returned by this function will be inner_ids
    @abc.abstractmethod
    def recommend_one(self, user_id, n_rec, filter_consumed, random_rec):
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
            return self.default_pred
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

    def recommend_user(
        self,
        user,
        n_rec,
        cold_start="popular",
        inner_id=False,
        filter_consumed=True,
        random_rec=False,
    ):
        """Recommend a list of items for given user(s).

        Parameters
        ----------
        user : int or str or array_like
            User id or batch of user ids to recommend.
        n_rec : int
            Number of recommendations to return.
        cold_start : {'popular'}, default: 'popular'
            Cold start strategy, CF models can only use 'popular' strategy.
        inner_id : bool, default: False
            Whether to use inner_id defined in `libreco`. For library users inner_id
            may never be used.
        filter_consumed : bool, default: True
            Whether to filter out items that a user has previously consumed.
        random_rec : bool, default: False
            Whether to choose items for recommendation based on their prediction scores.

        Returns
        -------
        recommendation : dict[Union[int, str, array_like], numpy.ndarray]
            Recommendation result with user ids as keys
            and array_like recommended items as values.
        """
        result_recs = dict()
        user_ids, unknown_users = check_unknown_user(self.data_info, user, inner_id)
        if unknown_users:
            if cold_start != "popular":
                raise ValueError(
                    f"{self.model_name} only supports `popular` cold start strategy"
                )
            for u in unknown_users:
                result_recs[u] = popular_recommendations(
                    self.data_info, inner_id, n_rec
                )
        if user_ids:
            computed_recs = [
                self.recommend_one(u, n_rec, filter_consumed, random_rec)
                for u in user_ids
            ]
            user_recs = construct_rec(self.data_info, user_ids, computed_recs, inner_id)
            result_recs.update(user_recs)
        return result_recs

    def rank_recommendations(
        self,
        user,
        ids,
        preds,
        n_rec,
        consumed,
        filter_consumed,
        random_rec,
    ):
        if filter_consumed:
            ids, preds = filter_items(ids, preds, consumed)
        if len(ids) == 0:  # pragma: no cover
            self.print_count += 1
            no_str = (
                f"no suitable recommendation for user {user}, "
                f"return default recommendation"
            )
            if self.print_count < 11:
                print(f"{colorize(no_str, 'red')}")
            return popular_recommendations(self.data_info, inner_id=True, n_rec=n_rec)

        if random_rec and len(ids) > n_rec:
            ids = random.sample(list(ids), k=n_rec)
        else:
            indices = np.argsort(preds)[::-1]
            ids = ids[indices][:n_rec]
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
        hparams = load_params(path, data_info, model_name)
        model = cls(**hparams)
        model_path = os.path.join(path, model_name)
        model.sim_matrix = load_sparse(f"{model_path}_sim_matrix.npz")
        model.user_interaction = load_sparse(f"{model_path}_user_inter.npz")
        model.item_interaction = load_sparse(f"{model_path}_item_inter.npz")
        return model
