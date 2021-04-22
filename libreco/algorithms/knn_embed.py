import os
from operator import itemgetter
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from .base import Base
from ..utils.misc import time_block, colorize
from ..evaluation.evaluate import EvalMixin
from ..embedding import Item2Vec


class KnnEmbedding(Base, EvalMixin):
    def __init__(
            self,
            task="ranking",
            data_info=None,
            embedding_method=None,
            embed_size=16,
            window_size=None,
            k=10,
            seed=42,
            lower_upper_bound=None
    ):
        Base.__init__(self, task, data_info, lower_upper_bound)
        EvalMixin.__init__(self, task, data_info)

        self.task = task
        self.data_info = data_info
        self.embed_size = embed_size
        self.window_size = window_size
        self.k = k
        self.seed = seed
        self.embed_algo = self._choose_embedding_algo(embedding_method)
        self.n_users = data_info.n_users
        self.n_items = data_info.n_items
        self.user_consumed = data_info.user_consumed
        self.topk_sim = None
        self.item_vectors = None
        self._item_norms = None
        self.print_count = 0
        self.all_args = locals()

    def fit(self, train_data, n_threads=0, verbose=1, eval_data=None,
            metrics=None, store_top_k=True):
        assert self.task == "ranking", (
            "KNNEmbedding model is only suitable for ranking"
        )
        self.show_start_time()
        self.embed_algo.fit(n_threads, verbose)
        self.item_vectors = self.embed_algo.item_vectors
        if store_top_k:
            self._compute_topk()

        if verbose > 1:
            self.print_metrics(eval_data=eval_data, metrics=metrics)
            print("=" * 30)

    def predict(self, user, item, cold="popular", inner_id=False):
        user, item = self.convert_id(user, item, inner_id)
        unknown_num, unknown_index, user, item = self._check_unknown(user, item)
        if unknown_num > 0 and cold != "popular":
            raise ValueError("KnnEmbedding only supports popular strategy.")

        preds = []
        for u, i in zip(user, item):
            user_interacted = self.user_consumed[u]
            num = (
                len(user_interacted)
                if len(user_interacted) < self.k
                else self.k
            )
            interacted_sims = self._compute_sim(i, user_interacted)
            k_sims = np.partition(interacted_sims, -num)[-num:]
            preds.append(np.mean(k_sims))   # max ?

        if unknown_num > 0:
            preds[unknown_index] = self.default_prediction

        return preds[0] if len(user) == 1 else preds

    def recommend_user(self, user, n_rec, random_rec=False,
                       cold_start="popular", inner_id=False):
        user_id = self._check_unknown_user(user)
        if user_id is None:
            if cold_start == "popular":
                return self.data_info.popular_items[:n_rec]
            elif cold_start != "popular":
                raise ValueError("KnnEmbedding only supports popular strategy.")
            else:
                raise ValueError(user)

        u_consumed = set(self.user_consumed[user])
        user_interacted = self.user_consumed[user]
        result = defaultdict(lambda: 0.)
        for i in user_interacted:
            item_sim_topk = (
                self.topk_sim[i]
                if self.topk_sim is not None
                else self.sort_topk_items(i)
            )

            for j, sim in item_sim_topk:
                if j in u_consumed:
                    continue
                result[j] += sim

        if len(result) == 0:
            self.print_count += 1
            no_str = (f"no suitable recommendation for user {user}, "
                      f"return default recommendation")
            if self.print_count < 7:
                print(f"{colorize(no_str, 'red')}")
            return self.data_info.popular_items[:n_rec]

        rank_items = [(k, v) for k, v in result.items()]
        rank_items.sort(key=lambda x: -x[1])
        return rank_items[:n_rec]

    def _compute_sim(self, item, u_interacted_items):
        # cosine similarity
        sim = self.item_vectors[item].dot(
            self.item_vectors[u_interacted_items].T
        ) / (self.item_norms[item] * self.item_norms[u_interacted_items])
        return sim

    def sort_topk_items(self, item):
        sim = self.item_vectors[item].dot(self.item_vectors.T) / (
                self.item_norms[item] * self.item_norms
        )
        ids = np.argpartition(sim, -self.k)[-self.k:]
        sorted_result = sorted(
            zip(ids, sim[ids]),
            key=itemgetter(1),
            reverse=True
        )
        return sorted_result

    def _compute_topk(self):
        top_k = []
        for i in tqdm(range(self.n_items), desc="top_k"):
            top_k.append(self.sort_topk_items(i))
        self.topk_sim = np.asarray(top_k)

    def _choose_embedding_algo(self, embedding_method):
        if embedding_method.lower().startswith("item2vec"):
            return Item2Vec(
                self.data_info, self.embed_size, self.window_size, self.seed
            )
        else:
            raise ValueError(f"{embedding_method} not implemented, yet.")

    @property
    def item_norms(self):
        if self._item_norms is None:
            self._item_norms = np.linalg.norm(self.item_vectors, axis=-1)
            self._item_norms[self._item_norms == 0] = 1e-10
        return self._item_norms

    def save(self, path, model_name, **kwargs):
        raise NotImplementedError("KnnEmbedding doesn't support model saving.")

    @classmethod
    def load(cls, path, model_name, data_info, **kwargs):
        raise NotImplementedError("KnnEmbedding doesn't support model loading.")

    def rebuild_graph(self, path, model_name, full_assign=False):
        raise NotImplementedError(
            "KnnEmbedding doesn't support model retraining")


class KnnEmbeddingApproximate(KnnEmbedding):
    def __init__(
            self,
            task="ranking",
            data_info=None,
            embedding_method=None,
            embed_size=16,
            window_size=None,
            k=10,
            seed=42,
            lower_upper_bound=None
    ):
        super(KnnEmbeddingApproximate, self).__init__(
            task,
            data_info,
            embedding_method,
            embed_size,
            window_size,
            k,
            seed,
            lower_upper_bound
        )
        self.approximate_algo = None

    def fit(self, train_data, n_threads=0, verbose=1, eval_data=None,
            metrics=None, store_top_k=True):
        assert self.task == "ranking", (
            "KNNEmbedding model is only suitable for ranking"
        )
        self.show_start_time()
        self.embed_algo.fit(n_threads, verbose)
        self.item_vectors = self.embed_algo.item_vectors
        self.build_approximate_search(n_threads, verbose)
        if store_top_k:
            self._compute_topk()

        if verbose > 1:
            self.print_metrics(eval_data=eval_data, metrics=metrics)
            print("=" * 30)

    def build_approximate_search(self, n_threads, verbose):
        try:
            import hnswlib
        except ModuleNotFoundError:
            print_str = "hnswlib is needed when using approximate_search..."
            print(f"{colorize(print_str, 'red')}")
            raise

        data_labels = np.arange(self.n_items)
        self.approximate_algo = hnswlib.Index(
            space="cosine", dim=self.embed_size
        )
        self.approximate_algo.init_index(
            max_elements=self.n_items, ef_construction=200, M=32
        )
        with time_block("approximate search init", verbose):
            self.approximate_algo.add_items(
                data=self.item_vectors,
                ids=data_labels,
                num_threads=os.cpu_count() if not n_threads else n_threads
            )
            self.approximate_algo.set_ef(64)

    # def _compute_sim(self, item):
    #    ids, sim = self.approximate_algo.knn_query(
    #        self.item_vectors[item], k=self.n_items
    #    )
    #    return sim[0][np.argsort(ids[0])]

    def sort_topk_items(self, item):
        ids, sim = self.approximate_algo.knn_query(
            self.item_vectors[item], k=self.k
        )
        return list(zip(ids[0], sim[0]))

    def _compute_topk(self):
        top_k = self.approximate_algo.knn_query(self.item_vectors, k=self.k)
        top_k = np.stack(top_k, axis=0)
        self.topk_sim = np.transpose(top_k, [1, 2, 0])

    def save(self, path, model_name, **kwargs):
        raise NotImplementedError("KnnEmbedding doesn't support model saving.")

    @classmethod
    def load(cls, path, model_name, data_info, **kwargs):
        raise NotImplementedError("KnnEmbedding doesn't support model loading.")

    def rebuild_graph(self, path, model_name, full_assign=False):
        raise NotImplementedError(
            "KnnEmbedding doesn't support model retraining")
