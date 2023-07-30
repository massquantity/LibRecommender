"""Implementation of SVD."""
import numpy as np

from ..bases import EmbedBase, ModelMeta
from ..layers import embedding_lookup, normalize_embeds
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
    norm_embed : bool, default: False
        Whether to l2 normalize output embeddings.
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

    user_variables = ("embedding/bu_var", "embedding/pu_var")
    item_variables = ("embedding/bi_var", "embedding/qi_var")

    def __init__(
        self,
        task,
        data_info,
        loss_type="cross_entropy",
        embed_size=16,
        norm_embed=False,
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
        self.norm_embed = norm_embed
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
        embed_user = embedding_lookup(
            indices=self.user_indices,
            var_name="pu_var",
            var_shape=(self.n_users, self.embed_size),
            initializer=tf.glorot_uniform_initializer(),
            regularizer=self.reg,
        )
        embed_item = embedding_lookup(
            indices=self.item_indices,
            var_name="qi_var",
            var_shape=(self.n_items, self.embed_size),
            initializer=tf.glorot_uniform_initializer(),
            regularizer=self.reg,
        )

        if self.norm_embed:
            embed_user, embed_item = normalize_embeds(
                embed_user, embed_item, backend="tf"
            )
        self.output = (
            bias_user + bias_item + tf.einsum("ij,ij->i", embed_user, embed_item)
        )

    def set_embeddings(self):
        with tf.variable_scope("embedding", reuse=True):
            bu = self.sess.run(tf.get_variable("bu_var"))
            bi = self.sess.run(tf.get_variable("bi_var"))
            pu = self.sess.run(tf.get_variable("pu_var"))
            qi = self.sess.run(tf.get_variable("qi_var"))

        user_bias = np.ones([len(pu), 2], dtype=pu.dtype)
        user_bias[:, 0] = bu
        item_bias = np.ones([len(qi), 2], dtype=qi.dtype)
        item_bias[:, 1] = bi
        if self.norm_embed:
            pu, qi = normalize_embeds(pu, qi, backend="np")
        self.user_embeds_np = np.hstack([pu, user_bias])
        self.item_embeds_np = np.hstack([qi, item_bias])
