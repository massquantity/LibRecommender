"""Implementation of NCF."""
from ..bases import ModelMeta, TfBase
from ..tfops import dense_nn, dropout_config, reg_config, tf, tf_dense
from ..torchops import hidden_units_config


class NCF(TfBase, metaclass=ModelMeta):
    """*Neural Collaborative Filtering* algorithm.

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
    use_bn : bool, default: True
        Whether to use batch normalization.
    dropout_rate : float or None, default: None
        Probability of an element to be zeroed. If it is None, dropout is not used.
    hidden_units : int, list of int or tuple of (int,), default: (128, 64, 32)
        Number of layers and corresponding layer size in MLP.

        .. versionchanged:: 1.0.0
           Accept type of ``int``, ``list`` or ``tuple``, instead of ``str``.

    seed : int, default: 42
        Random seed.
    lower_upper_bound : tuple or None, default: None
        Lower and upper score bound for `rating` task.
    tf_sess_config : dict or None, default: None
        Optional TensorFlow session config, see `ConfigProto options
        <https://github.com/tensorflow/tensorflow/blob/v2.10.0/tensorflow/core/protobuf/config.proto#L431>`_.

    References
    ----------
    *Xiangnan He et al.* `Neural Collaborative Filtering
    <https://arxiv.org/pdf/1708.05031.pdf>`_.
    """

    user_variables = ["user_gmf", "user_mlp"]
    item_variables = ["item_gmf", "item_mlp"]

    def __init__(
        self,
        task,
        data_info,
        loss_type="cross_entropy",
        embed_size=16,
        n_epochs=20,
        lr=0.01,
        lr_decay=False,
        epsilon=1e-5,
        reg=None,
        batch_size=256,
        sampler="random",
        num_neg=1,
        use_bn=True,
        dropout_rate=None,
        hidden_units=(128, 64, 32),
        seed=42,
        lower_upper_bound=None,
        tf_sess_config=None,
    ):
        super().__init__(task, data_info, lower_upper_bound, tf_sess_config)

        self.all_args = locals()
        self.loss_type = loss_type
        self.embed_size = embed_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.lr_decay = lr_decay
        self.epsilon = epsilon
        self.reg = reg_config(reg)
        self.batch_size = batch_size
        self.sampler = sampler
        self.num_neg = num_neg
        self.use_bn = use_bn
        self.dropout_rate = dropout_config(dropout_rate)
        self.hidden_units = hidden_units_config(hidden_units)
        self.seed = seed

    def build_model(self):
        self.user_indices = tf.placeholder(tf.int32, shape=[None])
        self.item_indices = tf.placeholder(tf.int32, shape=[None])
        self.labels = tf.placeholder(tf.float32, shape=[None])
        self.is_training = tf.placeholder_with_default(False, shape=[])

        user_gmf = tf.get_variable(
            name="user_gmf",
            shape=[self.n_users + 1, self.embed_size],
            initializer=tf.glorot_uniform_initializer(),
            regularizer=self.reg,
        )
        item_gmf = tf.get_variable(
            name="item_gmf",
            shape=[self.n_items + 1, self.embed_size],
            initializer=tf.glorot_uniform_initializer(),
            regularizer=self.reg,
        )
        user_mlp = tf.get_variable(
            name="user_mlp",
            shape=[self.n_users + 1, self.embed_size],
            initializer=tf.glorot_uniform_initializer(),
            regularizer=self.reg,
        )
        item_mlp = tf.get_variable(
            name="item_mlp",
            shape=[self.n_items + 1, self.embed_size],
            initializer=tf.glorot_uniform_initializer(),
            regularizer=self.reg,
        )

        user_gmf_embed = tf.nn.embedding_lookup(user_gmf, self.user_indices)
        item_gmf_embed = tf.nn.embedding_lookup(item_gmf, self.item_indices)
        user_mlp_embed = tf.nn.embedding_lookup(user_mlp, self.user_indices)
        item_mlp_embed = tf.nn.embedding_lookup(item_mlp, self.item_indices)

        gmf_layer = tf.multiply(user_gmf_embed, item_gmf_embed)
        mlp_input = tf.concat([user_mlp_embed, item_mlp_embed], axis=1)
        mlp_layer = dense_nn(
            mlp_input,
            self.hidden_units,
            use_bn=self.use_bn,
            dropout_rate=self.dropout_rate,
            is_training=self.is_training,
        )
        concat_layer = tf.concat([gmf_layer, mlp_layer], axis=1)
        self.output = tf.reshape(tf_dense(units=1)(concat_layer), [-1])
