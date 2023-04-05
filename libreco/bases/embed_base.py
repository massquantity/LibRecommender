"""Embed model base class."""
import abc
import os
from operator import itemgetter

import numpy as np

from .base import Base
from ..prediction import predict_from_embedding
from ..recommendation import cold_start_rec, construct_rec, recommend_from_embedding
from ..training.dispatch import get_trainer
from ..utils.constants import SEQUENCE_RECOMMEND_MODELS
from ..utils.misc import colorize
from ..utils.save_load import (
    load_default_recs,
    load_params,
    save_default_recs,
    save_params,
    save_tf_variables,
    save_torch_state_dict,
)
from ..utils.validate import check_fitting, check_unknown_user


class EmbedBase(Base):
    """Base class for embed models.

    Models that can generate user and item embeddings for inference.
    See `algorithm list <https://github.com/massquantity/LibRecommender#references>`_.

    Parameters
    ----------
    task : {'rating', 'ranking'}
        Recommendation task. See :ref:`Task`.
    data_info : :class:`~libreco.data.DataInfo` object
        Object that contains useful information for training and inference.
    embed_size: int
        Vector size of embeddings.
    lower_upper_bound : tuple or None, default: None
        Lower and upper score bound for `rating` task.
    """

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
        self.model_built = False
        self.trainer = None
        self.loaded = False

    @abc.abstractmethod
    def build_model(self):
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
        num_workers=0,
    ):
        """Fit embed model on the training data.

        Parameters
        ----------
        train_data : :class:`~libreco.data.TransformedSet` object
            Data object used for training.
        neg_sampling : bool
            Whether to perform negative sampling for training or evaluating data.

            .. versionadded:: 1.1.0

            .. NOTE::
               Negative sampling is needed if your data is implicit(i.e., `task` is ranking)
               and ONLY contains positive labels. Otherwise, it should be False.

        verbose : int, default: 1
            Print verbosity. If `eval_data` is provided, setting it to higher than 1
            will print evaluation metrics during training.
        shuffle : bool, default: True
            Whether to shuffle the training data.
        eval_data : :class:`~libreco.data.TransformedSet` object, default: None
            Data object used for evaluating.
        metrics : list or None, default: None
            List of metrics for evaluating.
        k : int, default: 10
            Parameter of metrics, e.g. recall at k, ndcg at k
        eval_batch_size : int, default: 8192
            Batch size for evaluating.
        eval_user_num : int or None, default: None
            Number of users for evaluating. Setting it to a positive number will sample
            users randomly from eval data.
        num_workers : int, default: 0
            How many subprocesses to use for training data loading.
            0 means that the data will be loaded in the main process,
            which is slower than multiprocessing.

            .. versionadded:: 1.1.0

            .. CAUTION::
               Using multiprocessing(``num_workers`` > 0) may consume more memory than
               single processing. See `Multi-process data loading <https://pytorch.org/docs/stable/data.html#multi-process-data-loading>`_.

        Raises
        ------
        RuntimeError
            If :py:func:`fit` is called from a loaded model(:py:func:`load`).
        AssertionError
            If ``neg_sampling`` parameter is not bool type.
        """
        check_fitting(self, train_data, eval_data, neg_sampling, k)
        self.show_start_time()
        if not self.model_built:
            self.build_model()
            self.model_built = True
        if self.trainer is None:
            self.trainer = get_trainer(self)
        self.trainer.run(
            train_data,
            neg_sampling,
            verbose,
            shuffle,
            eval_data,
            metrics,
            k,
            eval_batch_size,
            eval_user_num,
            num_workers,
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

    def predict(self, user, item, cold_start="average", inner_id=False):
        """Make prediction(s) on given user(s) and item(s).

        Parameters
        ----------
        user : int or str or array_like
            User id or batch of user ids.
        item : int or str or array_like
            Item id or batch of item ids.
        cold_start : {'popular', 'average'}, default: 'average'
            Cold start strategy.

            - 'popular' will sample from popular items.
            - 'average' will use the average of all the user/item embeddings as the
              representation of the cold-start user/item.

        inner_id : bool, default: False
            Whether to use inner_id defined in `libreco`. For library users inner_id
            may never be used.

        Returns
        -------
        prediction : float or numpy.ndarray
            Predicted scores for each user-item pair.
        """
        return predict_from_embedding(self, user, item, cold_start, inner_id)

    def recommend_user(
        self,
        user,
        n_rec,
        seq=None,
        cold_start="average",
        inner_id=False,
        filter_consumed=True,
        random_rec=False,
    ):
        """Recommend a list of items for given user(s).

        Parameters
        ----------
        user : int or str or array_like
            User id or batch of user ids to recommend.
        n_rec : int
            Number of recommendations to return.
        seq : list or numpy.ndarray
            Extra item sequence for recommendation. If the sequence length is larger than
            `recent_num` hyperparameter specified in the model, it will be truncated.
            If it is smaller, it will be padded.

            .. versionadded:: 1.1.0

        cold_start : {'popular', 'average'}, default: 'average'
            Cold start strategy.

            - 'popular' will sample from popular items.
            - 'average' will use the average of all the user/item embeddings as the
              representation of the cold-start user/item.

        inner_id : bool, default: False
            Whether to use inner_id defined in `libreco`. For library users inner_id
            may never be used.
        filter_consumed : bool, default: True
            Whether to filter out items that a user has previously consumed.
        random_rec : bool, default: False
            Whether to choose items for recommendation based on their prediction scores.

        Returns
        -------
        recommendation : dict of {Union[int, str, array_like] : numpy.ndarray}
            Recommendation result with user ids as keys and array_like recommended items as values.
        """
        if seq is not None:
            if self.model_name not in SEQUENCE_RECOMMEND_MODELS:
                raise ValueError(
                    f"`{self.model_name}` doesn't support arbitrary seq recommendation."
                )
            if not np.isscalar(user) and len(user) > 1:
                raise ValueError(
                    f"Batch recommend doesn't support arbitrary item sequence: {user}"
                )

        result_recs = dict()
        user_ids, unknown_users = check_unknown_user(self.data_info, user, inner_id)
        if unknown_users:
            cold_recs = cold_start_rec(
                self.data_info,
                self.default_recs,
                cold_start,
                unknown_users,
                n_rec,
                inner_id,
            )
            result_recs.update(cold_recs)
        if user_ids:
            computed_recs = recommend_from_embedding(
                self,
                user_ids,
                n_rec,
                self.user_embed,
                self.item_embed,
                seq,
                filter_consumed,
                random_rec,
                inner_id,
            )
            user_recs = construct_rec(self.data_info, user_ids, computed_recs, inner_id)
            result_recs.update(user_recs)
        return result_recs

    @abc.abstractmethod
    def set_embeddings(self):
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
        """Save embed model for inference or retraining.

        Parameters
        ----------
        path : str
            File folder path to save model.
        model_name : str
            Name of the saved model file.
        inference_only : bool, default: False
            Whether to save model only for inference. If it is True, only embeddings
            will be saved. Otherwise, model variables will be saved.

        See Also
        --------
        load
        """
        if not os.path.isdir(path):
            print(f"file folder {path} doesn't exists, creating a new one...")
            os.makedirs(path)
        save_params(self, path, model_name)
        save_default_recs(self, path, model_name)
        if inference_only:
            variable_path = os.path.join(path, model_name)
            np.savez_compressed(
                file=variable_path,
                user_embed=self.user_embed,
                item_embed=self.item_embed,
            )
        elif hasattr(self, "sess"):
            save_tf_variables(self.sess, path, model_name, inference_only=False)
        elif hasattr(self, "torch_model"):
            save_torch_state_dict(self, path, model_name)

    @classmethod
    def load(cls, path, model_name, data_info, **kwargs):
        """Load saved embed model for inference.

        Parameters
        ----------
        path : str
            File folder path to save model.
        model_name : str
            Name of the saved model file.
        data_info : :class:`~libreco.data.DataInfo` object
            Object that contains some useful information.

        Returns
        -------
        model : type(cls)
            Loaded embed model.

        See Also
        --------
        save
        """
        variable_path = os.path.join(path, f"{model_name}.npz")
        variables = np.load(variable_path)
        hparams = load_params(path, data_info, model_name)
        model = cls(**hparams)
        model.loaded = True
        model.default_recs = load_default_recs(path, model_name)
        model.user_embed = variables["user_embed"]
        model.item_embed = variables["item_embed"]
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
        """Get user embedding(s) from the model.

        Parameters
        ----------
        user : int or str or None
            Query user id. If it is None, all user embeddings will be returned.

        Returns
        -------
        user_embedding : numpy.ndarray
            Returned user embeddings.

        Raises
        ------
        ValueError
            If the user does not appear in the training data.
        AssertionError
            If the model has not been trained.
        """
        assert (
            self.user_embed is not None
        ), "call `model.fit()` before getting user embeddings"
        if user is None:
            return self.user_embed[:-1, : self.embed_size]  # remove oov
        user_id = self.get_user_id(user)
        return self.user_embed[user_id, : self.embed_size]

    def get_item_embedding(self, item=None):
        """Get item embedding(s) from the model.

        Parameters
        ----------
        item : int or str or None
            Query item id. If it is None, all item embeddings will be returned.

        Returns
        -------
        item_embedding : numpy.ndarray
            Returned item embeddings.

        Raises
        ------
        ValueError
            If the item does not appear in the training data.
        AssertionError
            If the model has not been trained.
        """
        assert (
            self.item_embed is not None
        ), "call `model.fit()` before getting item embeddings"
        if item is None:
            return self.item_embed[:-1, : self.embed_size]
        item_id = self.get_item_id(item)
        return self.item_embed[item_id, : self.embed_size]

    def init_knn(
        self, approximate, sim_type, M=100, ef_construction=200, ef_search=200
    ):
        """Initialize k-nearest-search model.

        Parameters
        ----------
        approximate : bool
            Whether to use approximate nearest neighbor search.
            If it is True, `nmslib <https://github.com/nmslib/nmslib>`_ must be installed.
            The `HNSW` method in `nmslib` is used.
        sim_type : {'cosine', 'inner-product'}
            Similarity space type.
        M : int, default: 100
            Parameter in `HNSW`, refer to `nmslib doc
            <https://github.com/nmslib/nmslib/blob/master/manual/methods.md>`_.
        ef_construction : int, default: 200
            Parameter in `HNSW`, refer to `nmslib doc
            <https://github.com/nmslib/nmslib/blob/master/manual/methods.md>`_.
        ef_search : int, default: 200
            Parameter in `HNSW`, refer to `nmslib doc
            <https://github.com/nmslib/nmslib/blob/master/manual/methods.md>`_.

        Raises
        ------
        ValueError
            If sim_type is not one of ('cosine', 'inner-product').
        ModuleNotFoundError
            If `approximate=True` and `nmslib` is not installed.
        """
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
            self.user_norm[self.user_norm == 0] = 1.0
            self.item_norm = np.linalg.norm(self.get_item_embedding(), axis=1)
            self.item_norm[self.item_norm == 0] = 1.0
        self.approximate = approximate
        self.sim_type = sim_type

    def search_knn_users(self, user, k):
        """Search most similar k users.

        Parameters
        ----------
        user : int or str
            Query user id.
        k : int
            Number of similar users.

        Returns
        -------
        similar users : list
            A list of k similar users.
        """
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
        """Search most similar k items.

        Parameters
        ----------
        item : int or str
            Query item id.
        k : int
            Number of similar items.

        Returns
        -------
        similar items : list
            A list of k similar items.
        """
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
