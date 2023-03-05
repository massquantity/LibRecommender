"""Implementation of SVD."""
import numpy as np

from ..bases import EmbedBase, ModelMeta
from ..tfops import reg_config, sess_config, tf


class SVD(EmbedBase, metaclass=ModelMeta, backend="tensorflow"):
    """*Singular Value Decomposition* algorithm.

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
    *Yehuda Koren* `Matrix Factorization Techniques for Recommender Systems
    <https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf>`_.
    """

    user_variables = ["bu_var", "pu_var"]
    item_variables = ["bi_var", "qi_var"]

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
        self.seed = seed

    def build_model(self):
        self.user_indices = tf.placeholder(tf.int32, shape=[None])
        self.item_indices = tf.placeholder(tf.int32, shape=[None])
        self.labels = tf.placeholder(tf.float32, shape=[None])

        self.bu_var = tf.get_variable(
            name="bu_var",
            shape=[self.n_users],
            initializer=tf.zeros_initializer(),
            regularizer=self.reg,
        )
        self.bi_var = tf.get_variable(
            name="bi_var",
            shape=[self.n_items],
            initializer=tf.zeros_initializer(),
            regularizer=self.reg,
        )
        self.pu_var = tf.get_variable(
            name="pu_var",
            shape=[self.n_users, self.embed_size],
            initializer=tf.glorot_uniform_initializer(),
            regularizer=self.reg,
        )
        self.qi_var = tf.get_variable(
            name="qi_var",
            shape=[self.n_items, self.embed_size],
            initializer=tf.glorot_uniform_initializer(),
            regularizer=self.reg,
        )

        bias_user = tf.nn.embedding_lookup(self.bu_var, self.user_indices)
        bias_item = tf.nn.embedding_lookup(self.bi_var, self.item_indices)
        embed_user = tf.nn.embedding_lookup(self.pu_var, self.user_indices)
        embed_item = tf.nn.embedding_lookup(self.qi_var, self.item_indices)
        self.output = (
            bias_user
            + bias_item
            + tf.reduce_sum(tf.multiply(embed_user, embed_item), axis=1)
        )

    def set_embeddings(self):
        bu, bi, pu, qi = self.sess.run(
            [self.bu_var, self.bi_var, self.pu_var, self.qi_var]
        )
        user_bias = np.ones([len(pu), 2], dtype=pu.dtype)
        user_bias[:, 0] = bu
        item_bias = np.ones([len(qi), 2], dtype=qi.dtype)
        item_bias[:, 1] = bi
        self.user_embed = np.hstack([pu, user_bias])
        self.item_embed = np.hstack([qi, item_bias])
