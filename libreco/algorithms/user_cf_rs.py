"""Implementation of RsUserCF."""
import os

from ..bases import Base
from ..evaluation import print_metrics
from ..prediction.preprocess import convert_id
from ..recommendation import construct_rec, popular_recommendations
from ..utils.misc import time_block
from ..utils.save_load import load_params, save_params
from ..utils.sparse import build_sparse
from ..utils.validate import check_fitting, check_unknown, check_unknown_user


class RsUserCF(Base):
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

        self.all_args = locals()
        self.k_sim = k_sim
        self.num_threads = num_threads
        self.min_common = min_common
        self.mode = mode
        self.seed = seed
        self.incremental = False
        self.user_cf_rs = None

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
        if self.incremental:
            assert isinstance(self.user_cf_rs, recfarm.UserCF)
            with time_block("update similarity", verbose=1):
                self.user_cf_rs.update_similarities(user_interacts, item_interacts)
        else:
            self.user_cf_rs = recfarm.UserCF(
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
            with time_block("similarity computation", verbose=1):
                self.user_cf_rs.compute_similarities(
                    self.mode == "invert", self.num_threads
                )

        num = self.user_cf_rs.num_sim_elements()
        density_ratio = 100 * num / (self.n_users * self.n_users)
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
        preds = self.user_cf_rs.predict(user_arr.tolist(), item_arr.tolist())
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
            computed_recs, no_rec_indices = self.user_cf_rs.recommend(
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
        recfarm.save_user_cf(self.user_cf_rs, path, model_name)

    @classmethod
    def load(cls, path, model_name, data_info, **kwargs):
        import recfarm

        hparams = load_params(path, data_info, model_name)
        model = cls(**hparams)
        model.user_cf_rs = recfarm.load_user_cf(path, model_name)
        return model

    def rebuild_model(self, path, model_name):
        import recfarm

        self.user_cf_rs = recfarm.load_user_cf(path, model_name)
        self.user_cf_rs.n_users = self.n_users
        self.user_cf_rs.n_items = self.n_items
        self.user_cf_rs.user_consumed = self.user_consumed
        self.incremental = True
