"""

Reference: Oren Barkan and Noam Koenigstein. "Item2Vec: Neural Item Embedding for Collaborative Filtering"
           (https://arxiv.org/pdf/1603.04259.pdf)

author: massquantity

"""
import os
import warnings
warnings.filterwarnings("ignore")
from gensim.models import Word2Vec
from tqdm import tqdm
import numpy as np
from ..utils.misc import time_block, colorize


class Item2Vec(object):
    def __init__(
            self,
            data_info=None,
            embed_size=16,
            window_size=None,
            seed=42,
    ):
        self.data_info = data_info
        self.embed_size = embed_size
        self.window_size = self._decide_window_size(
            window_size, self.data_info.user_consumed
        )
        self.seed = seed
        self.item_vectors = None
        print_str = (f"window size: {self.window_size}, "
                     f"using too large window size may slow down training.")
        print(f"{colorize(print_str, 'red')}")

    def fit(self, n_threads=0, verbose=1):
        data = ItemCorpus(self.data_info.user_consumed)
        with time_block(f"gensim word2vec training", verbose):
            model = Word2Vec(
                sentences=data,
                size=self.embed_size,
                window=self.window_size,
                sg=1,
                hs=0,
                negative=5,
                seed=self.seed,
                iter=5,
                min_count=1,
                workers=os.cpu_count() if not n_threads else n_threads,
                sorted_vocab=0
            )

        self.item_vectors = np.array(
            [model.wv.get_vector(str(i)) for i in range(self.data_info.n_items)]
        )

    def get_item_vec(self, item):
        assert self.item_vectors is not None, "must fit the model first..."
        return self.item_vectors[item]

    @staticmethod
    def _decide_window_size(window_size, user_consumed):
        if not window_size:
            return max(len(seq) for seq in user_consumed.values()) + 5
        else:
            return window_size


class ItemCorpus(object):
    def __init__(self, user_consumed):
        self.item_seqs = user_consumed.values()
        self.i = 0

    def __iter__(self):
        for items in tqdm(self.item_seqs, desc=f"Item2vec iter{self.i}"):
            yield list(map(str, items))
        self.i += 1
