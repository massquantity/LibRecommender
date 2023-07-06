"""TF model base class."""
import abc
import os

import numpy as np

from .base import Base
from ..prediction import predict_tf_feat
from ..recommendation import (
    check_dynamic_rec_feats,
    cold_start_rec,
    construct_rec,
    recommend_tf_feat,
)
from ..tfops import modify_variable_names, sess_config, tf
from ..training.dispatch import get_trainer
from ..utils.save_load import (
    load_tf_model,
    load_tf_variables,
    save_default_recs,
    save_params,
    save_tf_model,
    save_tf_variables,
)
from ..utils.validate import check_fitting, check_unknown_user


class TfBase(Base):
    """Base class for TF models.

    Models that relies on TensorFlow graph for inference. Although some models such as
    `RNN4Rec`, `SVD` etc., are trained using TensorFlow, they don't inherit from this
    base class since their inference only uses embeddings.

    Parameters
    ----------
    task : {'rating', 'ranking'}
        Recommendation task. See :ref:`Task`.
    data_info : :class:`~libreco.data.DataInfo` object
        Object that contains useful information for training and inference.
    lower_upper_bound : tuple or None
        Lower and upper score bound for `rating` task.
    tf_sess_config : dict or None
        Optional TensorFlow session config, see `ConfigProto options
        <https://github.com/tensorflow/tensorflow/blob/v2.10.0/tensorflow/core/protobuf/config.proto#L431>`_.
    """

    def __init__(self, task, data_info, lower_upper_bound=None, tf_sess_config=None):
        super().__init__(task, data_info, lower_upper_bound)
        self.sess = sess_config(tf_sess_config)
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
        """Fit TF model on the training data.

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
            Print verbosity.

            - ``verbose <= 0``: Print nothing.
            - ``verbose == 1``: Print progress bar and training time.
            - ``verbose > 1`` : Print evaluation metrics if ``eval_data`` is provided.

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
        self.assign_tf_variables_oov()
        self.default_recs = recommend_tf_feat(
            model=self,
            user_ids=[self.n_users],
            n_rec=min(2000, self.n_items),
            user_feats=None,
            seq=None,
            filter_consumed=False,
            random_rec=False,
        ).flatten()

    def predict(self, user, item, feats=None, cold_start="average", inner_id=False):
        """Make prediction(s) on given user(s) and item(s).

        Parameters
        ----------
        user : int or str or array_like
            User id or batch of user ids.
        item : int or str or array_like
            Item id or batch of item ids.
        feats : dict or pandas.Series or None, default: None
            Extra features used in prediction.
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
        if self.model_name == "NCF" and feats is not None:
            raise ValueError("NCF can't use features.")
        return predict_tf_feat(self, user, item, feats, cold_start, inner_id)

    def recommend_user(
        self,
        user,
        n_rec,
        user_feats=None,
        seq=None,
        cold_start="average",
        inner_id=False,
        filter_consumed=True,
        random_rec=False,
    ):
        """Recommend a list of items for given user(s).

        If both ``user_feats`` and ``seq`` are ``None``, the model will use the stored features
        for recommendation, and the ``cold_start`` strategy will be used for unknown users.

        If either ``user_feats`` or ``seq`` is provided, the model will use them for recommendation.
        In this case, if the ``user`` is unknown, it will be set to padding id, which means
        the ``cold_start`` strategy will not be applied.
        This situation is common when one wants to recommend for an unknown user based on
        user features or behavior sequence.

        Parameters
        ----------
        user : int or str or array_like
            User id or batch of user ids to recommend.
        n_rec : int
            Number of recommendations to return.
        user_feats : dict or None, default: None
            Extra user features for recommendation.
        seq : list or numpy.ndarray or None, default: None
            Extra item sequence for recommendation. If the sequence length is larger than
            `recent_num` hyperparameter specified in the model, it will be truncated.
            If smaller, it will be padded.

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
        if self.model_name == "NCF" and user_feats is not None:
            raise ValueError("`NCF` can't use features.")

        if user_feats is None and seq is None:
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
                computed_recs = recommend_tf_feat(
                    self,
                    user_ids,
                    n_rec,
                    user_feats,
                    seq,
                    filter_consumed,
                    random_rec,
                    inner_id,
                )
                user_recs = construct_rec(
                    self.data_info, user_ids, computed_recs, inner_id
                )
                result_recs.update(user_recs)
        else:
            # must be a single user if `user_feats` or `seq` is provided
            check_dynamic_rec_feats(self.model_name, user, user_feats, seq)
            user_id = self._convert_id(user, inner_id)
            computed_recs = recommend_tf_feat(
                self,
                [user_id],
                n_rec,
                user_feats,
                seq,
                filter_consumed,
                random_rec,
                inner_id,
            )
            rec_items = (
                computed_recs[0]
                if inner_id
                else np.array([self.data_info.id2item[i] for i in computed_recs[0]])
            )
            result_recs = {user: rec_items}

        return result_recs

    def _convert_id(self, user, inner_id):
        """Convert a single user to inner user id.

        If the user doesn't exist, it will be converted to padding id.
        """
        assert np.isscalar(user), f"User to convert must be scalar, got: {user}"
        if inner_id:
            if not isinstance(user, (int, np.integer)):
                raise ValueError(f"`inner id` user must be int, got {user}")
            return user if 0 <= user < self.n_users else self.n_users
        else:
            return self.data_info.user2id.get(user, self.n_users)

    def assign_tf_variables_oov(self):
        (
            user_variables,
            item_variables,
            sparse_variables,
            dense_variables,
            _,
        ) = modify_variable_names(self, trainable=True)

        update_ops = []
        for v in tf.trainable_variables():
            if user_variables is not None and v.name in user_variables:
                # size = v.get_shape().as_list()[1]
                mean_op = tf.IndexedSlices(
                    tf.reduce_mean(
                        tf.gather(v, tf.range(self.n_users)), axis=0, keepdims=True
                    ),
                    [self.n_users],
                )
                update_ops.append(v.scatter_update(mean_op))

            if item_variables is not None and v.name in item_variables:
                mean_op = tf.IndexedSlices(
                    tf.reduce_mean(
                        tf.gather(v, tf.range(self.n_items)), axis=0, keepdims=True
                    ),
                    [self.n_items],
                )
                update_ops.append(v.scatter_update(mean_op))

            if sparse_variables is not None and v.name in sparse_variables:
                sparse_oovs = self.data_info.sparse_oov
                start = 0
                for oov in sparse_oovs:
                    # multi_sparse case
                    if start >= oov:
                        continue
                    mean_tensor = tf.reduce_mean(
                        tf.gather(v, tf.range(start, oov)), axis=0, keepdims=True
                    )
                    update_ops.append(v.scatter_nd_update([[oov]], mean_tensor))
                    start = oov + 1

        self.sess.run(update_ops)

    def build_topk(self, outputs):
        self.k = tf.placeholder(tf.int32, shape=())
        _, indices = tf.math.top_k(outputs, self.k, sorted=True)
        return indices

    def save(self, path, model_name, manual=True, inference_only=False):
        """Save TF model for inference or retraining.

        Parameters
        ----------
        path : str
            File folder path to save model.
        model_name : str
            Name of the saved model file.
        manual : bool, default: True
            Whether to save model variables using numpy.
        inference_only : bool, default: False
            Whether to save model variables only for inference.

        See Also
        --------
        load
        """
        if not os.path.isdir(path):
            print(f"file folder {path} doesn't exists, creating a new one...")
            os.makedirs(path)
        save_params(self, path, model_name)
        save_default_recs(self, path, model_name)
        if manual:
            save_tf_variables(self.sess, path, model_name, inference_only)
        else:
            save_tf_model(self.sess, path, model_name)

    @classmethod
    def load(cls, path, model_name, data_info, manual=True):
        """Load saved TF model for inference.

        Parameters
        ----------
        path : str
            File folder path to save model.
        model_name : str
            Name of the saved model file.
        data_info : :class:`~libreco.data.DataInfo` object
            Object that contains some useful information.
        manual : bool, default: True
            Whether to load model variables using numpy. If you save the model using
            `manual`, you should also load the mode using `manual`.

        Returns
        -------
        model : type(cls)
            Loaded TF model.

        See Also
        --------
        save
        """
        if manual:
            return load_tf_variables(cls, path, model_name, data_info)
        else:
            return load_tf_model(cls, path, model_name, data_info)
