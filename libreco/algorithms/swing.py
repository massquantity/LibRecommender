"""Implementation of Swing."""
import pathlib

from ..bases import Base
from ..evaluation import print_metrics
from ..prediction.preprocess import convert_id
from ..recommendation import construct_rec, popular_recommendations
from ..utils.misc import time_block
from ..utils.save_load import load_params, save_params
from ..utils.sparse import build_sparse
from ..utils.validate import check_fitting, check_unknown, check_unknown_user


class Swing(Base):
    """*Swing* algorithm.

    .. CAUTION::
        + Swing can only be used in ``ranking`` task.

    Parameters
    ----------
    task : {'ranking'}
        Recommendation task. See :ref:`Task`.
    data_info : :class:`~libreco.data.DataInfo` object
        Object that contains useful information for training and inference.
    top_k : int, default: 20
        Number of items to consider during recommendation.
    alpha : float, default: 1.0
        Smoothing coefficient.
    max_cache_num : int, default: 100,000,000
        Maximum cached item number during swing score computing.
    num_threads : int, default: 1
        Number of threads to use.
    seed : int, default: 42
        Random seed.

    References
    ----------
    *Xiaoyong Yang et al.* `Large Scale Product Graph Construction for Recommendation in E-commerce
    <https://arxiv.org/pdf/2010.05525>`_.
    """

    def __init__(
        self,
        task,
        data_info,
        top_k=20,
        alpha=1.0,
        max_cache_num=100_000_000,
        num_threads=1,
        seed=42,
    ):
        super().__init__(task, data_info, lower_upper_bound=None)

        assert task == "ranking", "`Swing` is only suitable for ranking task."
        self.all_args = locals()
        self.top_k = top_k
        self.alpha = alpha
        self.max_cache_num = max_cache_num
        self.num_threads = num_threads
        self.seed = seed
        self.rs_model = None
        self.incremental = False

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
        import recfarm

        check_fitting(self, train_data, eval_data, neg_sampling, k)
        self.show_start_time()
        user_interacts = build_sparse(train_data.sparse_interaction)
        item_interacts = build_sparse(train_data.sparse_interaction, transpose=True)
        self.rs_model = recfarm.Swing(
            self.task,
            self.top_k,
            self.alpha,
            self.max_cache_num,
            self.n_users,
            self.n_items,
            user_interacts,
            item_interacts,
            self.user_consumed,
            self.default_pred,
        )
        with time_block("swing computing", verbose=1):
            self.rs_model.compute_swing(self.num_threads, self.incremental)

        num = self.rs_model.num_swing_elements()
        density_ratio = 100 * num / (self.n_items * self.n_items)
        print(f"swing num_elements: {num}, density: {density_ratio:5.4f} %")
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

    def predict(self, user, item, cold_start="popular", inner_id=False):
        user_arr, item_arr = convert_id(self, user, item, inner_id)
        unknown_num, _, user_arr, item_arr = check_unknown(self, user_arr, item_arr)
        if unknown_num > 0 and cold_start != "popular":
            raise ValueError(f"{self.model_name} only supports popular strategy")
        preds = self.rs_model.predict(user_arr.tolist(), item_arr.tolist())
        return preds[0] if len(user_arr) == 1 else preds

    def recommend_user(
        self,
        user,
        n_rec,
        cold_start="popular",
        inner_id=False,
        filter_consumed=True,
        random_rec=False,
    ):
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
            computed_recs, no_rec_indices = self.rs_model.recommend(
                user_ids,
                n_rec,
                filter_consumed,
                random_rec,
            )
            for i in no_rec_indices:
                computed_recs[i] = popular_recommendations(
                    self.data_info, inner_id=True, n_rec=n_rec
                )
            user_recs = construct_rec(self.data_info, user_ids, computed_recs, inner_id)
            result_recs.update(user_recs)
        return result_recs

    def save(self, path, model_name, **kwargs):
        import recfarm

        path_obj = pathlib.Path(path)
        if not path_obj.is_dir():
            print(f"file folder {path} doesn't exists, creating a new one...")
            path_obj.mkdir(parents=True, exist_ok=False)
        save_params(self, path, model_name)
        recfarm.save_swing(self.rs_model, path, model_name)

    @classmethod
    def load(cls, path, model_name, data_info, **kwargs):
        import recfarm

        hparams = load_params(path, data_info, model_name)
        model = cls(**hparams)
        model.rs_model = recfarm.load_swing(path, model_name)
        return model

    def rebuild_model(self, path, model_name):
        """Assign the saved model variables to the newly initialized model.

        This method is used before retraining the new model, in order to avoid training
        from scratch every time we get some new data.

        Parameters
        ----------
        path : str
            File folder path for the saved model variables.
        model_name : str
            Name of the saved model file.
        """
        import recfarm

        self.rs_model = recfarm.load_swing(path, model_name)
        self.rs_model.n_users = self.n_users
        self.rs_model.n_items = self.n_items
        self.rs_model.user_consumed = self.user_consumed
        self.incremental = True
