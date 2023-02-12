"""Implementation of Item2Vec."""
from gensim.models import Word2Vec
from tqdm import tqdm

from ..bases import GensimBase


class Item2Vec(GensimBase):
    """*Item2Vec* algorithm.

    .. WARNING::
        Item2Vec can only use in ``ranking`` task.

    Parameters
    ----------
    task : {'ranking'}
        Recommendation task. See :ref:`Task`.
    data_info : :class:`~libreco.data.DataInfo` object
        Object that contains useful information for training and inference.
    embed_size: int, default: 16
        Vector size of embeddings.
    norm_embed : bool, default: False
        Whether to normalize output embeddings.
    window_size : int, default: 5
        Maximum item distance within a sequence during training.
    n_epochs: int, default: 10
        Number of epochs for training.
    n_threads : int, default: 0
        Number of threads to use, `0` will use all cores.
    seed : int, default: 42
        Random seed.
    lower_upper_bound : tuple or None, default: None
        Lower and upper score bound for `rating` task.

    References
    ----------
    *Oren Barkan and Noam Koenigstein.* `Item2Vec: Neural Item Embedding for Collaborative Filtering
    <https://arxiv.org/pdf/1603.04259.pdf>`_.
    """

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

    def get_data(self):
        return _ItemCorpus(self.user_consumed)

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


class _ItemCorpus:
    def __init__(self, user_consumed):
        self.item_seqs = user_consumed.values()
        self.i = 0

    def __iter__(self):
        for items in tqdm(self.item_seqs, desc=f"Item2vec iter{self.i}"):
            yield list(map(str, items))
        self.i += 1
