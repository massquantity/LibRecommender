"""Implementation of SVD++."""
import numpy as np

from ..bases import EmbedBase
from ..layers import embedding_lookup, sparse_embeds_pooling
from ..tfops import rebuild_tf_model, reg_config, sess_config, tf


class SVDpp(EmbedBase):
    """*SVD++* algorithm.

    Parameters
    ----------
    task : {'rating', 'ranking'}
        Recommendation task. See :ref:`Task`.
    data_info : :class:`~libreco.data.DataInfo` object
        Object that contains useful information for training and inference.
    loss_type : {'cross_entropy', 'focal'}, default: 'cross_entropy'
        Loss for model training.
    embed_size: int, default: 16
        Vector size of embeddings.
    n_epochs: int, default: 10
        Number of epochs for training.
    lr : float, default 0.001
        Learning rate for training.
    lr_decay : bool, default: False
        Whether to use learning rate decay.
    epsilon : float, default: 1e-5
        A small constant added to the denominator to improve numerical stability in
        Adam optimizer.
        According to the `official comment <https://github.com/tensorflow/tensorflow/blob/v1.15.0/tensorflow/python/training/adam.py#L64>`_,
        default value of `1e-8` for `epsilon` is generally not good, so here we choose `1e-5`.
        Users can try tuning this hyperparameter if the training is unstable.
    reg : float or None, default: None
        Regularization parameter, must be non-negative or None.
    batch_size : int, default: 256
        Batch size for training.
    sampler : {'random', 'unconsumed', 'popular'}, default: 'random'
        Negative sampling strategy.

        - ``'random'`` means random sampling.
        - ``'unconsumed'`` samples items that the target user did not consume before.
        - ``'popular'`` has a higher probability to sample popular items as negative samples.

        .. versionadded:: 1.1.0

    num_neg : int, default: 1
        Number of negative samples for each positive sample, only used in `ranking` task.
    seed : int, default: 42
        Random seed.
    lower_upper_bound : tuple or None, default: None
        Lower and upper score bound for `rating` task.
    tf_sess_config : dict or None, default: None
        Optional TensorFlow session config, see `ConfigProto options
        <https://github.com/tensorflow/tensorflow/blob/v2.10.0/tensorflow/core/protobuf/config.proto#L431>`_.

    References
    ----------
    *Yehuda Koren* `Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model
    <https://dl.acm.org/citation.cfm?id=1401944>`_.
    """

    user_variables = ("embedding/bu_var", "embedding/pu_var", "embedding/yj_var")
    item_variables = ("embedding/bi_var", "embedding/qi_var")

    def __init__(
        self,
        task,
        data_info,
        loss_type="cross_entropy",
        embed_size=16,
        n_epochs=20,
        lr=0.001,
        lr_decay=False,
        epsilon=1e-5,
        reg=None,
        batch_size=256,
        sampler="random",
        num_neg=1,
        seed=42,
        recent_num=30,
        lower_upper_bound=None,
        tf_sess_config=None,
    ):
        super().__init__(task, data_info, embed_size, lower_upper_bound)

        self.all_args = locals()
        self.sess = sess_config(tf_sess_config)
        self.loss_type = loss_type
        self.n_epochs = n_epochs
        self.lr = lr
        self.lr_decay = lr_decay
        self.epsilon = epsilon
        self.reg = reg_config(reg)
        self.batch_size = batch_size
        self.sampler = sampler
        self.num_neg = num_neg
        self.recent_num = recent_num
        self.seed = seed
        self.sparse_interaction = None

    def build_model(self):
        tf.set_random_seed(self.seed)
        self.user_indices = tf.placeholder(tf.int32, shape=[None])
        self.item_indices = tf.placeholder(tf.int32, shape=[None])
        self.labels = tf.placeholder(tf.float32, shape=[None])

        bias_user = embedding_lookup(
            indices=self.user_indices,
            var_name="bu_var",
            var_shape=[self.n_users],
            initializer=tf.zeros_initializer(),
            regularizer=self.reg,
        )
        bias_item = embedding_lookup(
            indices=self.item_indices,
            var_name="bi_var",
            var_shape=[self.n_items],
            initializer=tf.zeros_initializer(),
            regularizer=self.reg,
        )

        self.all_user_embeds = self._compute_user_embeddings()
        embed_user = embedding_lookup(self.user_indices, embed_var=self.all_user_embeds)
        embed_item = embedding_lookup(
            indices=self.item_indices,
            var_name="qi_var",
            var_shape=(self.n_items, self.embed_size),
            initializer=tf.glorot_uniform_initializer(),
            regularizer=self.reg,
        )

        self.output = (
            bias_user + bias_item + tf.einsum("ij,ij->i", embed_user, embed_item)
        )

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
        if self.sparse_interaction is None:
            self.sparse_interaction = self._set_sparse_interaction()
        super().fit(
            train_data,
            neg_sampling,
            verbose,
            shuffle,
            eval_data,
            metrics,
            k,
            eval_batch_size,
            eval_user_num,
        )

    def set_embeddings(self):
        with tf.variable_scope("embedding", reuse=True):
            bu = self.sess.run(tf.get_variable("bu_var"))
            bi = self.sess.run(tf.get_variable("bi_var"))
            qi = self.sess.run(tf.get_variable("qi_var"))

        puj = self.sess.run(self.all_user_embeds)
        user_bias = np.ones([len(puj), 2], dtype=puj.dtype)
        user_bias[:, 0] = bu
        item_bias = np.ones([len(qi), 2], dtype=qi.dtype)
        item_bias[:, 1] = bi
        self.user_embeds_np = np.hstack([puj, user_bias])
        self.item_embeds_np = np.hstack([qi, item_bias])

    def _set_sparse_interaction(self):
        assert self.recent_num is None or (
            isinstance(self.recent_num, int) and self.recent_num > 0
        ), "`recent_num` must be None or positive int"
        indices = []
        values = []
        for u in range(self.n_users):
            items = self.user_consumed[u]
            u_data = items if self.recent_num is None else items[-self.recent_num :]
            indices.extend([u] * len(u_data))
            values.extend(u_data)
        indices = np.array(indices)[:, None]
        indices = np.concatenate([indices, np.zeros_like(indices)], axis=1)
        sparse_interaction = tf.SparseTensor(
            indices=indices, values=values, dense_shape=(self.n_users, self.n_items)
        )
        return sparse_interaction

    def _compute_user_embeddings(self):
        with tf.variable_scope("embedding", reuse=None):
            pu_var = tf.get_variable(
                name="pu_var",
                shape=(self.n_users, self.embed_size),
                initializer=tf.glorot_uniform_initializer(),
                regularizer=self.reg,
            )

        uj = sparse_embeds_pooling(
            self.sparse_interaction,
            var_name="yj_var",
            var_shape=(self.n_items, self.embed_size),
            initializer=tf.glorot_uniform_initializer(),
            regularizer=self.reg,
            reuse_layer=False,
            scope_name="embedding",
        )
        return pu_var + uj

    def rebuild_model(self, path, model_name, full_assign=False):
        self.sparse_interaction = self._set_sparse_interaction()
        rebuild_tf_model(self, path, model_name, full_assign)
