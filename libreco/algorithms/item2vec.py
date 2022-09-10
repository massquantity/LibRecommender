"""

Reference: Oren Barkan and Noam Koenigstein. "Item2Vec: Neural Item Embedding for Collaborative Filtering"
           (https://arxiv.org/pdf/1603.04259.pdf)

author: massquantity

"""
import os

import numpy as np
from gensim.models import Word2Vec
from tqdm import tqdm

from ..bases import EmbedBase
from ..evaluation import print_metrics
from ..utils.misc import colorize, time_block
from ..utils.save_load import save_params


class Item2Vec(EmbedBase):
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
        k=10,
        eval_batch_size=8192,
        eval_user_num=None,
        lower_upper_bound=None,
    ):
        super().__init__(task, data_info, embed_size, lower_upper_bound)

        assert self.task == "ranking", "Item2Vec is only suitable for ranking"
        self.all_args = locals()
        self.norm_embed = norm_embed
        self.window_size = self._decide_window_size(
            window_size, self.data_info.user_consumed
        )
        self.n_epochs = n_epochs
        self.n_threads = n_threads
        self.seed = seed
        self.k = k
        self.eval_batch_size = eval_batch_size
        self.eval_user_num = eval_user_num

    @staticmethod
    def _decide_window_size(window_size, user_consumed):
        if not window_size:
            result = max(len(seq) for seq in user_consumed.values()) + 5
        else:
            result = window_size
        print_str = (
            f"window size: {result}, "
            f"using too large window size may slow down training."
        )
        print(f"{colorize(print_str, 'red')}")
        return result

    def fit(self, train_data, verbose=1, eval_data=None, metrics=None, **kwargs):
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
                sorted_vocab=0,
            )

        self.set_embeddings(model)
        self.assign_embedding_oov()
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

    def set_embeddings(self, model):
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
        save_params(self, path, model_name)
        variable_path = os.path.join(path, model_name)
        np.savez_compressed(
            variable_path, user_embed=self.user_embed, item_embed=self.item_embed
        )


class ItemCorpus(object):
    def __init__(self, user_consumed):
        self.item_seqs = user_consumed.values()
        self.i = 0

    def __iter__(self):
        for items in tqdm(self.item_seqs, desc=f"Item2vec iter{self.i}"):
            yield list(map(str, items))
        self.i += 1
