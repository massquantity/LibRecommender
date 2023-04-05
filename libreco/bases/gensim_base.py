import abc
import os

import numpy as np
from gensim.models import Word2Vec

from .embed_base import EmbedBase
from ..evaluation import print_metrics
from ..recommendation import recommend_from_embedding
from ..utils.misc import time_block
from ..utils.save_load import save_default_recs, save_params
from ..utils.validate import check_fitting


class GensimBase(EmbedBase):
    """Base class for models that use Gensim for training.

    Including Item2Vec and Deepwalk.
    """

    def __init__(
        self,
        task,
        data_info,
        embed_size=16,
        norm_embed=False,
        window_size=5,
        n_epochs=5,
        n_threads=0,
        seed=42,
        lower_upper_bound=None,
    ):
        super().__init__(task, data_info, embed_size, lower_upper_bound)
        self.norm_embed = norm_embed
        self.window_size = 5 if not window_size else window_size
        self.n_epochs = n_epochs
        self.workers = os.cpu_count() if not n_threads else n_threads
        self.seed = seed
        self.gensim_model = None
        self.data = None

    @abc.abstractmethod
    def get_data(self):
        raise NotImplementedError

    def fit(
        self,
        train_data,
        neg_sampling,
        verbose=1,
        shuffle=True,
        eval_data=None,
        metrics=None,
        k=10,
        eval_batch_size=8192,
        eval_user_num=None,
        **kwargs,
    ):
        check_fitting(self, train_data, eval_data, neg_sampling, k)
        self.show_start_time()
        if self.data is None:
            self.data = self.get_data()
        if self.gensim_model is None:
            self.gensim_model = self.build_model()
        with time_block("gensim word2vec training", verbose):
            self.gensim_model.train(
                self.data,
                total_examples=self.gensim_model.corpus_count,
                epochs=self.n_epochs,
            )
        self.set_embeddings()
        self.assign_embedding_oov()
        self.default_recs = recommend_from_embedding(
            model=self,
            user_ids=[self.n_users],
            n_rec=min(2000, self.n_items),
            user_embeddings=self.user_embed,
            item_embeddings=self.item_embed,
            seq=None,
            filter_consumed=False,
            random_rec=False,
        ).flatten()

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

    def set_embeddings(self):
        self.item_embed = np.array(
            [self.gensim_model.wv.get_vector(str(i)) for i in range(self.n_items)]
        )
        user_embed = []
        for u in range(self.n_users):
            items = self.user_consumed[u]
            user_embed.append(np.mean(self.item_embed[items], axis=0))
            # user_embed.append(self.item_embed[items[-1]])
        self.user_embed = np.array(user_embed)

        if self.norm_embed:
            user_norms = np.linalg.norm(self.user_embed, axis=1, keepdims=True)
            user_norms[user_norms == 0] = 1.0
            self.user_embed /= user_norms
            item_norms = np.linalg.norm(self.item_embed, axis=1, keepdims=True)
            item_norms[item_norms == 0] = 1.0
            self.item_embed /= item_norms

    def save(self, path, model_name, inference_only=False, **_):
        if not os.path.isdir(path):
            print(f"file folder {path} doesn't exists, creating a new one...")
            os.makedirs(path)
        save_params(self, path, model_name)
        save_default_recs(self, path, model_name)
        if inference_only:
            variable_path = os.path.join(path, model_name)
            np.savez_compressed(
                variable_path, user_embed=self.user_embed, item_embed=self.item_embed
            )
        else:
            model_path = os.path.join(path, f"{model_name}_gensim.pkl")
            self.gensim_model.save(model_path)

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
        self.data = self.get_data()
        model_path = os.path.join(path, f"{model_name}_gensim.pkl")
        self.gensim_model = Word2Vec.load(model_path)
        self.gensim_model.build_vocab(self.data, update=True)
