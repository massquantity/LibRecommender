"""

Reference: Oren Barkan and Noam Koenigstein. "Item2Vec: Neural Item Embedding for Collaborative Filtering"
           (https://arxiv.org/pdf/1603.04259.pdf)

author: massquantity

"""
import os
from itertools import islice

import numpy as np
from gensim.models import Word2Vec
from tqdm import tqdm

from .base import Base
from ..evaluation.evaluate import EvalMixin
from ..utils.misc import assign_oov_vector, colorize, time_block


class Item2Vec(Base, EvalMixin):
    user_variables_np = ["user_embed"]
    item_variables_np = ["item_embed"]

    def __init__(
            self,
            task,
            data_info=None,
            embed_size=16,
            norm_embed=False,
            window_size=None,
            n_epochs=5,
            n_threads=0,
            seed=42,
    ):
        Base.__init__(self, task, data_info)
        EvalMixin.__init__(self, task, data_info)

        self.task = task
        self.data_info = data_info
        self.embed_size = embed_size
        self.norm_embed = norm_embed
        self.window_size = self._decide_window_size(
            window_size, self.data_info.user_consumed
        )
        self.n_epochs = n_epochs
        self.n_threads = n_threads
        self.seed = seed
        self.n_users = data_info.n_users
        self.n_items = data_info.n_items
        self.user_consumed = data_info.user_consumed
        self.user_embed = None
        self.item_embed = None
        self.graph = None
        self.all_args = locals()
        print_str = (f"window size: {self.window_size}, "
                     f"using too large window size may slow down training.")
        print(f"{colorize(print_str, 'red')}")

    @staticmethod
    def _decide_window_size(window_size, user_consumed):
        if not window_size:
            return max(len(seq) for seq in user_consumed.values()) + 5
        else:
            return window_size

    def fit(self, train_data, verbose=1, eval_data=None, metrics=None, **kwargs):
        assert self.task == "ranking", "Item2Vec is only suitable for ranking"
        self.show_start_time()
        data = ItemCorpus(self.data_info.user_consumed)
        with time_block("gensim word2vec training", verbose):
            workers = os.cpu_count() if not self.n_threads else self.n_threads
            model = Word2Vec(
                sentences=data,
                vector_size=self.embed_size,
                window=self.window_size,
                sg=1,
                hs=0,
                negative=5,
                seed=self.seed,
                epochs=self.n_epochs,
                min_count=1,
                workers=workers,
                sorted_vocab=0
            )

        self._set_latent_vectors(model)
        assign_oov_vector(self)
        if verbose > 1:
            self.print_metrics(eval_data=eval_data, metrics=metrics, **kwargs)
            print("=" * 30)

    def predict(self, user, item, cold_start="average", inner_id=False):
        user, item = self.convert_id(user, item, inner_id)
        unknown_num, unknown_index, user, item = self._check_unknown(user, item)
        preds = np.sum(
            np.multiply(self.user_embed[user], self.item_embed[item]),
            axis=1
        )
        preds = 1 / (1 + np.exp(-preds))
        if unknown_num > 0 and cold_start == "popular":
            preds[unknown_index] = self.default_prediction
        return preds

    def recommend_user(self, user, n_rec, cold_start="average", inner_id=False):
        user_id = self._check_unknown_user(user, inner_id)
        if user_id is None:
            if cold_start == "average":
                user_id = self.n_users
            elif cold_start == "popular":
                return self.popular_recommends(inner_id, n_rec)
            else:
                raise ValueError(user)

        consumed = set(self.user_consumed[user_id])
        count = n_rec + len(consumed)
        recos = self.user_embed[user_id] @ self.item_embed.T
        recos = 1 / (1 + np.exp(-recos))

        ids = np.argpartition(recos, -count)[-count:]
        rank = sorted(zip(ids, recos[ids]), key=lambda x: -x[1])
        recs_and_scores = islice(
            (rec if inner_id else (self.data_info.id2item[rec[0]], rec[1])
             for rec in rank if rec[0] not in consumed),
            n_rec
        )
        return list(recs_and_scores)

    def _set_latent_vectors(self, model):
        self.item_embed = np.array(
            [model.wv.get_vector(str(i)) for i in range(self.n_items)]
        )
        user_embed = []
        for u in range(self.n_users):
            items = self.user_consumed[u]
            user_embed.append(np.mean(self.item_embed[items], axis=0))
        self.user_embed = np.array(user_embed)

        if self.norm_embed:
            user_norms = np.linalg.norm(self.user_embed, axis=1, keepdims=True)
            user_norms[user_norms == 0] = 1e-10
            self.user_embed /= user_norms
            item_norms = np.linalg.norm(self.item_embed, axis=1, keepdims=True)
            item_norms[item_norms == 0] = 1e-10
            self.item_embed /= item_norms

    def save(self, path, model_name, manual=True, inference_only=False):
        if not os.path.isdir(path):
            print(f"file folder {path} doesn't exists, creating a new one...")
            os.makedirs(path)
        self.save_params(path)
        variable_path = os.path.join(path, model_name)
        np.savez_compressed(variable_path,
                            user_embed=self.user_embed,
                            item_embed=self.item_embed)

    @classmethod
    def load(cls, path, model_name, data_info, manual=True):
        variable_path = os.path.join(path, f"{model_name}.npz")
        variables = np.load(variable_path)
        hparams = cls.load_params(path, data_info)
        model = cls(**hparams)
        model.user_embed = variables["user_embed"]
        model.item_embed = variables["item_embed"]
        return model


class ItemCorpus(object):
    def __init__(self, user_consumed):
        self.item_seqs = user_consumed.values()
        self.i = 0

    def __iter__(self):
        for items in tqdm(self.item_seqs, desc=f"Item2vec iter{self.i}"):
            yield list(map(str, items))
        self.i += 1
