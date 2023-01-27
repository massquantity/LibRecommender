"""

Reference: Oren Barkan and Noam Koenigstein. "Item2Vec: Neural Item Embedding for Collaborative Filtering"
           (https://arxiv.org/pdf/1603.04259.pdf)

author: massquantity

"""
from gensim.models import Word2Vec
from tqdm import tqdm

from ..bases import GensimBase


class Item2Vec(GensimBase):
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
        lower_upper_bound=None,
        with_training=True,
    ):
        super().__init__(
            task,
            data_info,
            embed_size,
            norm_embed,
            window_size,
            n_epochs,
            n_threads,
            seed,
            lower_upper_bound,
        )
        assert task == "ranking", "Item2Vec is only suitable for ranking"
        self.all_args = locals()
        if with_training:
            self.data = self.get_data()

    def get_data(self):
        return ItemCorpus(self.user_consumed)

    def build_model(self):
        model = Word2Vec(
            vector_size=self.embed_size,
            window=self.window_size,
            sg=1,
            hs=0,
            negative=5,
            seed=self.seed,
            min_count=1,
            workers=self.workers,
            sorted_vocab=0,
        )
        model.build_vocab(self.data, update=False)
        return model


class ItemCorpus:
    def __init__(self, user_consumed):
        self.item_seqs = user_consumed.values()
        self.i = 0

    def __iter__(self):
        for items in tqdm(self.item_seqs, desc=f"Item2vec iter{self.i}"):
            yield list(map(str, items))
        self.i += 1
