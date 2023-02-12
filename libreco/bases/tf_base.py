"""TF model base class."""
import abc
import os

import numpy as np

from ..prediction import predict_tf_feat
from ..recommendation import cold_start_rec, construct_rec, recommend_tf_feat
from ..tfops import modify_variable_names, sess_config, tf
from ..training import get_trainer
from ..utils.save_load import (
    load_tf_model,
    load_tf_variables,
    save_default_recs,
    save_params,
    save_tf_model,
    save_tf_variables,
)
from ..utils.validate import check_unknown_user
from .base import Base


class TfBase(Base):
    """Base class for TF models.

    Models that relies on TensorFlow graph for inference. Although some models such as
    `RNN4Rec`, `SVD` etc., are trained using TensorFlow, they don't belong to this
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
        verbose=1,
        shuffle=True,
        eval_data=None,
        metrics=None,
        k=10,
        eval_batch_size=8192,
        eval_user_num=None,
    ):
        """Fit TF model on the training data.

        Parameters
        ----------
        train_data : :class:`~libreco.data.TransformedSet` object
            Data object used for training.
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

        Raises
        ------
        RuntimeError
            If :py:func:`fit` is called from a loaded model(:py:func:`load`).
        """
        if self.loaded:
            raise RuntimeError(
                "Loaded model doesn't support retraining, use `rebuild_model` instead. "
                "Or constructing a new model from scratch."
            )
        if eval_data is not None and k > self.n_items:
            raise ValueError(f"eval `k` {k} exceeds num of items {self.n_items}")

        self.show_start_time()
        if not self.model_built:
            self.build_model()
            self.model_built = True
        if self.trainer is None:
            self.trainer = get_trainer(self)
        self.trainer.run(
            train_data,
            verbose,
            shuffle,
            eval_data,
            metrics,
            k,
            eval_batch_size,
            eval_user_num,
        )
        self.assign_tf_variables_oov()
        self.default_recs = recommend_tf_feat(
            model=self,
            user_ids=[self.n_users],
            n_rec=min(2000, self.n_items),
            user_feats=None,
            item_data=None,
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
        return predict_tf_feat(self, user, item, feats, cold_start, inner_id)

    def recommend_user(
        self,
        user,
        n_rec,
        user_feats=None,
        item_data=None,
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
        user_feats : dict or pandas.Series or None, default: None
            Extra user features for recommendation.
        item_data : pandas.DataFrame or None, default: None
            Extra item features for recommendation.
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
            Recommendation result with user ids as keys
            and array_like recommended items as values.
        """
        if (
            (user_feats is not None or item_data is not None)
            and not np.isscalar(user)
            and len(user) > 1
        ):
            raise ValueError(
                f"Batch recommend doesn't support assigning arbitrary features: {user}"
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
            computed_recs = recommend_tf_feat(
                self,
                user_ids,
                n_rec,
                user_feats,
                item_data,
                filter_consumed,
                random_rec,
            )
            user_recs = construct_rec(self.data_info, user_ids, computed_recs, inner_id)
            result_recs.update(user_recs)
        return result_recs

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
