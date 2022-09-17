import abc
import os
from operator import itemgetter

import numpy as np

from .base import Base
from ..prediction import predict_from_embedding
from ..recommendation import recommend_from_embedding
from ..utils.misc import colorize
from ..utils.save_load import load_params, save_params, save_tf_variables


class EmbedBase(Base):
    def __init__(self, task, data_info, embed_size, lower_upper_bound=None):
        super().__init__(task, data_info, lower_upper_bound)
        self.user_embed = None
        self.item_embed = None
        self.embed_size = embed_size
        self.num_threads = os.cpu_count()
        self.trainer = None
        self.user_index = None
        self.item_index = None
        self.user_norm = None
        self.item_norm = None
        self.sim_type = None
        self.approximate = False

    def fit(
        self,
        train_data,
        verbose=1,
        shuffle=True,
        eval_data=None,
        metrics=None,
        **kwargs,
    ):
        assert (
            self.trainer is not None
        ), "loaded model doesn't support retraining, use `rebuild_model` instead."
        self.show_start_time()
        # self._check_has_sampled(train_data, verbose)
        self.trainer.run(train_data, verbose, shuffle, eval_data, metrics)
        self.set_embeddings()
        self.assign_embedding_oov()

    def predict(self, user, item, cold_start="average", inner_id=False):
        return predict_from_embedding(self, user, item, cold_start, inner_id)

    def recommend_user(self, user, n_rec, cold_start="average", inner_id=False):
        return recommend_from_embedding(self, user, n_rec, cold_start, inner_id)

    @abc.abstractmethod
    def set_embeddings(self, *args, **kwargs):
        pass

    def assign_embedding_oov(self):
        for v_name in ("user_embed", "item_embed"):
            embed = getattr(self, v_name)
            if embed.ndim == 1:
                new_embed = np.append(embed, np.mean(embed))
                setattr(self, v_name, new_embed)
            else:
                new_embed = np.vstack([embed, np.mean(embed, axis=0)])
                setattr(self, v_name, new_embed)

    def save(self, path, model_name, inference_only=False, **kwargs):
        if not os.path.isdir(path):
            print(f"file folder {path} doesn't exists, creating a new one...")
            os.makedirs(path)
        save_params(self, path, model_name)
        if inference_only:
            variable_path = os.path.join(path, model_name)
            np.savez_compressed(
                file=variable_path,
                user_embed=self.user_embed,
                item_embed=self.item_embed,
            )
        elif hasattr(self, "sess"):
            save_tf_variables(self.sess, path, model_name, inference_only=False)
        else:
            pass  # todo: torch

    @classmethod
    def load(cls, path, model_name, data_info, **kwargs):
        variable_path = os.path.join(path, f"{model_name}.npz")
        variables = np.load(variable_path)
        hparams = load_params(cls, path, data_info, model_name)
        model = cls(**hparams)
        setattr(model, "user_embed", variables["user_embed"])
        setattr(model, "item_embed", variables["item_embed"])
        return model

    def get_user_id(self, user):
        if user not in self.data_info.user2id:
            raise ValueError(f"unknown user: {user}")
        return self.data_info.user2id[user]

    def get_item_id(self, item):
        if item not in self.data_info.item2id:
            raise ValueError(f"unknown item: {item}")
        return self.data_info.item2id[item]

    def get_user_embedding(self, user=None):
        assert (
            self.user_embed is not None
        ), "call `model.fit()` before getting user embeddings"
        if user is None:
            return self.user_embed[:-1, : self.embed_size]  # remove oov
        user_id = self.get_user_id(user)
        return self.user_embed[user_id, : self.embed_size]

    def get_item_embedding(self, item=None):
        assert (
            self.item_embed is not None
        ), "call `model.fit()` before getting item embeddings"
        if item is None:
            return self.item_embed[:-1, : self.embed_size]
        item_id = self.get_item_id(item)
        return self.item_embed[item_id, : self.embed_size]

    def init_knn(
        self,
        approximate,
        sim_type,
        M=100,
        ef_construction=200,
        ef_search=200
    ):
        if sim_type == "cosine":
            space = "cosinesimil"
        elif sim_type == "inner-product":
            space = "negdotprod"
        else:
            raise ValueError(
                f"unknown sim_type: {sim_type}, "
                f"only `cosine` and `inner-product` are supported"
            )

        def _create_index(data):
            index = nmslib.init(
                method="hnsw", space=space, data_type=nmslib.DataType.DENSE_VECTOR
            )
            index.addDataPointBatch(data)
            index.createIndex(
                {
                    "M": M,
                    "indexThreadQty": self.num_threads,
                    "efConstruction": ef_construction,
                }
            )
            index.setQueryTimeParams({"efSearch": ef_search})
            return index

        if approximate:
            try:
                import nmslib
            except (ImportError, ModuleNotFoundError):
                print_str = "`nmslib` is needed when using approximate_search..."
                print(f"{colorize(print_str, 'red')}")
                raise
            else:
                print("using approximate searching mode...")
            self.user_index = _create_index(self.get_user_embedding())
            self.item_index = _create_index(self.get_item_embedding())
        elif sim_type == "cosine":
            self.user_norm = np.linalg.norm(self.get_user_embedding(), axis=1)
            self.user_norm[self.user_norm == 0] = 1e-10
            self.item_norm = np.linalg.norm(self.get_item_embedding(), axis=1)
            self.item_norm[self.item_norm == 0] = 1e-10
        self.approximate = approximate
        self.sim_type = sim_type

    def search_knn_users(self, user, k):
        query = self.get_user_embedding(user)
        if self.approximate:
            ids, _ = self.user_index.knnQuery(query, k)
            return [self.data_info.id2user[i] for i in ids]

        embeds = self.get_user_embedding()
        sim = query.dot(embeds.T)
        if self.sim_type == "cosine":
            user_id = self.get_user_id(user)
            norm = self.user_norm[user_id] * self.user_norm
            sim /= norm
        ids = np.argpartition(sim, -k)[-k:]
        sorted_result = sorted(zip(ids, sim[ids]), key=itemgetter(1), reverse=True)
        return [self.data_info.id2user[i[0]] for i in sorted_result]

    def search_knn_items(self, item, k):
        query = self.get_item_embedding(item)
        if self.approximate:
            ids, _ = self.item_index.knnQuery(query, k)
            return [self.data_info.id2item[i] for i in ids]

        embeds = self.get_item_embedding()
        sim = query.dot(embeds.T)
        if self.sim_type == "cosine":
            item_id = self.get_item_id(item)
            norm = self.item_norm[item_id] * self.item_norm
            sim /= norm
        ids = np.argpartition(sim, -k)[-k:]
        sorted_result = sorted(zip(ids, sim[ids]), key=itemgetter(1), reverse=True)
        return [self.data_info.id2item[i[0]] for i in sorted_result]
