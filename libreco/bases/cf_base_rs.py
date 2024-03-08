"""Rust CF model base class."""
import os.path

from .base import Base
from ..evaluation import print_metrics
from ..prediction.preprocess import convert_id
from ..recommendation import construct_rec, popular_recommendations
from ..utils.misc import time_block
from ..utils.save_load import load_params, save_params
from ..utils.sparse import build_sparse
from ..utils.validate import check_fitting, check_unknown, check_unknown_user


class RsCfBase(Base):
    def __init__(
        self,
        task,
        data_info,
        k_sim=20,
        num_threads=1,
        min_common=1,
        mode="invert",
        seed=42,
        lower_upper_bound=None,
    ):
        super().__init__(task, data_info, lower_upper_bound)

        self.k_sim = k_sim
        self.num_threads = num_threads
        self.min_common = min_common
        self.mode = mode
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
        rs_model_cls = (
            recfarm.UserCF if "user" in self.model_name.lower() else recfarm.ItemCF
        )
        if self.incremental:
            assert isinstance(self.rs_model, rs_model_cls)
            with time_block("update similarity", verbose=1):
                self.rs_model.update_similarities(user_interacts, item_interacts)
        else:
            self.rs_model = rs_model_cls(
                self.task,
                self.k_sim,
                self.n_users,
                self.n_items,
                self.min_common,
                user_interacts,
                item_interacts,
                self.user_consumed,
                self.default_pred,
            )
            with time_block("similarity", verbose=1):
                self.rs_model.compute_similarities(
                    self.mode == "invert", self.num_threads
                )

        num = self.rs_model.num_sim_elements()
        if "user" in self.model_name.lower():
            density_ratio = 100 * num / (self.n_users * self.n_users)
        else:
            density_ratio = 100 * num / (self.n_items * self.n_items)
        print(f"similarity num_elements: {num}, density: {density_ratio:5.4f} %")
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

        if not os.path.isdir(path):
            print(f"file folder {path} doesn't exists, creating a new one...")
            os.makedirs(path)
        save_params(self, path, model_name)
        if "user" in self.model_name.lower():
            recfarm.save_user_cf(self.rs_model, path, model_name)
        else:
            recfarm.save_item_cf(self.rs_model, path, model_name)

    @classmethod
    def load(cls, path, model_name, data_info, **kwargs):
        import recfarm

        hparams = load_params(path, data_info, model_name)
        model = cls(**hparams)
        if "user" in cls.__name__.lower():
            model.rs_model = recfarm.load_user_cf(path, model_name)
        else:
            model.rs_model = recfarm.load_item_cf(path, model_name)
        return model

    def rebuild_model(self, path, model_name):
        import recfarm

        if "user" in self.model_name.lower():
            self.rs_model = recfarm.load_user_cf(path, model_name)
        else:
            self.rs_model = recfarm.load_item_cf(path, model_name)
        self.rs_model.n_users = self.n_users
        self.rs_model.n_items = self.n_items
        self.rs_model.user_consumed = self.user_consumed
        self.incremental = True
